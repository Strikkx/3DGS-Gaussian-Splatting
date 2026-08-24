"""Step 3: diffusion repair of sparse-view 3DGS renders. Self-contained.

Differences vs. the first draft (all of these are things you should be able to
defend in the KYC discussion):

  * get_combined_args(): ModelParams(sentinel=True) sets every dataset arg to
    None on purpose so they get read back from <model_path>/cfg_args. Plain
    parse_args() never does that merge, so source_path stayed None.
  * The VAE runs in fp32 in the conf_schedule path. The SD2 VAE is numerically
    unstable in fp16 and silently emits NaNs / black frames. The UNet stays in
    fp16 -- it is the expensive part and it is fine at half precision.
  * The binary-inpainting BASELINE now composites with the same binary mask it
    handed to the inpainter. Previously it inpainted with a hard mask but
    blended with the soft confidence, which meant the ablation was changing two
    things at once and could not isolate the schedule.
  * PSNR / SSIM / LPIPS against ground truth, per view, for both the raw render
    and the repaired image. Written into manifest.json and printed as a summary.
    This is the whole point: you cannot tune a threshold by squinting at PNGs.
  * Confidence can be dumped as .npy so step 4 can use it as a per-pixel loss
    weight rather than just a diagnostic image.
  * torch.quantile() refuses inputs over ~16M elements, so it is wrapped with a
    subsampling fallback for full-resolution renders.
  * Cue defaults are now alpha-only (w_grad=0, w_near=0). On BICYCLE the depth
    gradient cue fires on every real occlusion boundary -- bench slats, wheel
    spokes -- because a legitimate silhouette and a floater edge look identical
    to a Sobel filter. Turn them back on explicitly to ablate them.
"""
import os
import sys
import json
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from utils.loss_utils import ssim as _repo_ssim
    SSIM_AVAILABLE = True
except Exception:
    SSIM_AVAILABLE = False

DEFAULT_PROMPT = ("a sharp outdoor photograph of a bicycle leaning against a "
                  "bench on grass, trees and a lawn in the background, "
                  "natural daylight, high detail, photorealistic")


# --------------------------------------------------------------------------
# image IO (PIL only; torchvision is broken in this env)
# --------------------------------------------------------------------------
def _u8(x):
    import numpy as np
    x = x.detach().float().cpu().clamp(0, 1)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_img(path, x):
    from PIL import Image
    Image.fromarray(_u8(x)).save(path)


def save_debug_grid(path, rgb, alpha, inv_depth, confidence):
    from PIL import Image

    def to3(x):
        x = x.detach().float().cpu()
        return (x.repeat(3, 1, 1) if x.shape[0] == 1 else x).clamp(0, 1)

    if inv_depth is not None:
        d = inv_depth.detach().float().cpu()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    else:
        d = torch.zeros_like(alpha.detach().float().cpu())
    panels = torch.cat([to3(rgb), to3(alpha), to3(d), to3(confidence)], dim=2)
    Image.fromarray(_u8(panels)).save(path)


# --------------------------------------------------------------------------
# numerics helpers
# --------------------------------------------------------------------------
_QUANTILE_MAX = 8_000_000  # torch.quantile hard-fails somewhere above ~2**24


def _quantile(x, q):
    """torch.quantile with a subsampling fallback for large tensors."""
    x = x.flatten().float()
    if x.numel() > _QUANTILE_MAX:
        step = x.numel() // _QUANTILE_MAX + 1
        x = x[::step]
    return torch.quantile(x, q)


# --------------------------------------------------------------------------
# artifact localisation (no ground truth used here -- deliberately)
# --------------------------------------------------------------------------
# Alpha compositing: I = C + (1-a)*bg. Render white and black:
#   I_white - I_black = 1-a   =>   a = 1 - (I_white - I_black)
# The rasterizer never returns alpha, so we recover it with two renders.
def render_with_alpha(camera, gaussians, pipe_cfg):
    white = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    black = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    with torch.no_grad():
        # use_trained_exp=False: the learned per-image exposure is an affine
        # colour transform and would break the linearity above.
        pkg_b = render(camera, gaussians, pipe_cfg, black,
                       use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)
        pkg_w = render(camera, gaussians, pipe_cfg, white,
                       use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)
    rgb = pkg_b["render"].clamp(0, 1)
    alpha = (1.0 - (pkg_w["render"] - pkg_b["render"]).mean(0, keepdim=True)).clamp(0, 1)
    inv_d = pkg_b.get("depth", None)
    if inv_d is not None and inv_d.dim() == 2:
        inv_d = inv_d.unsqueeze(0)
    return rgb, alpha, inv_d


def _sobel(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    x4 = x.unsqueeze(0)
    gx = F.conv2d(x4, kx, padding=1)
    gy = F.conv2d(x4, kx.transpose(2, 3), padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0)
    p99 = _quantile(mag, 0.99).clamp(min=1e-6)
    return (mag / p99).clamp(0, 1)


def _dilate(m, k):
    if k <= 1:
        return m
    return F.max_pool2d(m.unsqueeze(0), k, 1, k // 2).squeeze(0)


def _blur(m, k):
    if k <= 1:
        return m
    sigma = max(k / 3.0, 1e-3)
    c = torch.arange(k, device=m.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    x = m.unsqueeze(0)
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    x = F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return x.squeeze(0).clamp(0, 1)


def compute_confidence(alpha, inv_depth, alpha_thresh=0.40,
                       depth_grad_thresh=0.25, near_pct=0.98,
                       w_alpha=1.0, w_grad=0.0, w_near=0.0,
                       dilate_k=5, blur_k=9):
    """Returns confidence [1,H,W]: 1 = trust, 0 = garbage.

    Three cues, each producing a "badness" in [0,1]:
      alpha  -- low accumulated opacity means no Gaussian covers this pixel,
                so the rasterizer composited straight to the background colour.
                On BICYCLE this is the dominant, and correct, signal.
      grad   -- large inverse-depth gradient. Intended to catch floater edges,
                but it cannot tell a floater edge from a real silhouette. Off
                by default; turn on only as an ablation.
      near   -- top near_pct of inverse depth, i.e. suspiciously close geometry.
                This is a PER-IMAGE percentile, so it unconditionally condemns
                2% of every frame whether or not floaters exist -- and in this
                scene the nearest object is the subject. Off by default.

    Cues are fused by max (soft OR: any single strong cue condemns a pixel),
    then dilated and blurred so the diffusion model gets a little context
    around each bad region rather than a pixel-tight hole. Confidence stays
    continuous; thresholding it recovers the binary-inpainting baseline.
    """
    a_bad = ((alpha_thresh - alpha) / max(alpha_thresh, 1e-6)).clamp(0, 1)

    if inv_depth is not None and (w_grad > 0 or w_near > 0):
        d = inv_depth.float()
        lo = _quantile(d, 0.01)
        hi = _quantile(d, 0.99).clamp(min=lo + 1e-6)
        dn = ((d - lo) / (hi - lo)).clamp(0, 1)

        g_bad = (_sobel(dn) - depth_grad_thresh).clamp(min=0.0)
        g_bad = (g_bad / max(1.0 - depth_grad_thresh, 1e-6)).clamp(0, 1)

        cut = _quantile(dn, near_pct)
        n_bad = ((dn - cut) / (1.0 - cut).clamp(min=1e-6)).clamp(0, 1)
    else:
        g_bad = n_bad = torch.zeros_like(alpha)

    bad = torch.maximum(torch.maximum(w_alpha * a_bad, w_grad * g_bad),
                        w_near * n_bad).clamp(0, 1)
    bad = _blur(_dilate(bad, dilate_k), blur_k)
    return (1.0 - bad).clamp(0, 1)


# --------------------------------------------------------------------------
# latent diffusion repair
# --------------------------------------------------------------------------
def load_repairer(mode, precision="fp16", vae_fp32=True, device="cuda"):
    """mode 'inpaint' -> SD2-inpainting (its UNet takes 9 input channels).
    mode 'conf_schedule' -> SD2.1-base, driven by our own sampling loop.

    NOTE FOR THE WRITE-UP: these are two different checkpoints. If you compare
    them head to head you are changing method AND backbone at once. Say so, or
    run conf_schedule against the inpainting checkpoint too.
    """
    # Workaround for a torch circular-import bug (hits torch ~2.5-2.7):
    # diffusers' first act is `from torch._dynamo import allow_in_graph`.
    # _dynamo's own __init__ transitively does `import torch.onnx.operators`,
    # which reaches torch._export -> torch.export._unlift, whose module body
    # applies `@torch._dynamo.disable` -- while _dynamo is still on line 2 of
    # its own __init__. Result: "partially initialized module 'torch._dynamo'".
    # Entering the cycle from torch._export instead lets it resolve in an
    # order where the decorator is available by the time it is needed.
    try:
        import torch._export  # noqa: F401
    except Exception:
        pass

    from diffusers import (StableDiffusionInpaintPipeline,
                           StableDiffusionPipeline, DDIMScheduler)
    dtype = torch.float16 if precision == "fp16" else torch.float32

    if mode == "inpaint":
        p = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
    else:
        p = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)

    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)  # deterministic
    p = p.to(device)
    p.set_progress_bar_config(disable=True)

    # Only safe to split precision in the custom loop, where we control every
    # cast ourselves. The high-level inpaint __call__ assumes one dtype.
    if vae_fp32 and mode != "inpaint" and dtype == torch.float16:
        p.vae = p.vae.to(torch.float32)

    try:
        p.enable_attention_slicing()
    except Exception:
        pass
    return p


def _fit(x, long=512, mode="bilinear"):
    """SD2.1-base is 512-native and the VAE needs dims divisible by 8.
    Downsample, repair, upsample. Real limitation worth stating out loud:
    detail below ~512px cannot be restored, only invented."""
    _, h, w = x.shape
    s = long / max(h, w)
    nh = max(int(round(h * s)) // 8 * 8, 8)
    nw = max(int(round(w * s)) // 8 * 8, 8)
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(x.unsqueeze(0), size=(nh, nw), mode=mode,
                         **kw).squeeze(0), (h, w)


def _back(x, hw):
    return F.interpolate(x.unsqueeze(0), size=hw, mode="bilinear",
                         align_corners=False).squeeze(0)


@torch.no_grad()
def repair_binary(pipe, rgb, conf, prompt, conf_thresh=0.5, steps=30,
                  guidance=7.5, seed=0, sd_res=512):
    """BASELINE: threshold confidence into a binary mask, standard inpainting.

    The composite uses the SAME binary mask that was fed to the inpainter.
    Blending with the soft confidence here would smuggle the proposed method's
    soft-edge behaviour into the baseline and invalidate the ablation.
    """
    from PIL import Image
    import numpy as np

    small, hw = _fit(rgb, sd_res)
    cs, _ = _fit(conf, sd_res, mode="area")
    mask_small = (cs < conf_thresh).float()

    img = Image.fromarray(_u8(small))
    msk = Image.fromarray(
        (mask_small[0].cpu().numpy() * 255).astype(np.uint8)).convert("L")

    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(prompt=prompt, image=img, mask_image=msk,
               num_inference_steps=steps, guidance_scale=guidance,
               generator=g, height=small.shape[1], width=small.shape[2]).images[0]

    t = torch.from_numpy(np.array(out).astype("float32") / 255.0)
    t = _back(t.permute(2, 0, 1).to(rgb.device), hw)

    # Composite against the ORIGINAL full-res render, not the VAE round-trip,
    # otherwise every trusted pixel pays a reconstruction penalty and PSNR
    # collapses for reasons that have nothing to do with the repair.
    mask_full = (conf < conf_thresh).float()
    return ((1 - mask_full) * rgb + mask_full * t).clamp(0, 1)


@torch.no_grad()
def repair_conf_schedule(pipe, rgb, conf, prompt, steps=30, guidance=6.0,
                         seed=0, max_strength=0.85, softness=1.5, sd_res=512):
    """PROPOSED: no threshold. Per-pixel release timestep t_start = T*(1-c).

    Each latent site stays pinned to its forward-noised original until the
    schedule reaches its release step, then evolves freely under the model.
    c=0 -> regenerated from the start, c=1 -> pinned throughout. Binary
    inpainting is the degenerate case where c only takes values 0 and 1.
    """
    device = pipe.device
    unet_dtype = pipe.unet.dtype
    vae, unet, sch = pipe.vae, pipe.unet, pipe.scheduler
    vae_dtype = next(vae.parameters()).dtype

    small, hw = _fit(rgb, sd_res)
    cs, _ = _fit(conf, sd_res, mode="area")

    x = small.unsqueeze(0).to(device=device, dtype=vae_dtype) * 2 - 1
    z0 = (vae.encode(x).latent_dist.mean * vae.config.scaling_factor).to(unet_dtype)

    # area-pool confidence to latent resolution, not nearest-neighbour: a
    # sub-8px floater occupies less than one latent cell and would vanish.
    c_lat = F.interpolate(cs.unsqueeze(0).to(device, torch.float32),
                          size=z0.shape[-2:], mode="area")

    sch.set_timesteps(steps, device=device)
    ts = sch.timesteps[int(steps * (1 - max_strength)):]
    N = len(ts)
    release = c_lat * (N - 1)

    # One fixed noise draw shared across all views. This is a cheap partial
    # consistency prior -- it is NOT a real multi-view consistency mechanism,
    # and you should say so rather than overclaim it.
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z0.shape, generator=g, device=device,
                        dtype=torch.float32).to(unet_dtype)

    if hasattr(pipe, "encode_prompt"):
        pos, neg = pipe.encode_prompt(prompt, device, 1, guidance > 1.0, "")
        emb = torch.cat([neg, pos]) if guidance > 1.0 else pos
    else:
        emb = pipe._encode_prompt(prompt, device, 1, guidance > 1.0, "")

    z = sch.add_noise(z0, noise, ts[0:1])
    for i, t in enumerate(ts):
        inp = torch.cat([z] * 2) if guidance > 1.0 else z
        eps = unet(sch.scale_model_input(inp, t), t,
                   encoder_hidden_states=emb).sample
        if guidance > 1.0:
            eu, ec = eps.chunk(2)
            eps = eu + guidance * (ec - eu)
        z_model = sch.step(eps, t, z).prev_sample
        if i < N - 1:
            # pin at t_next, not t: step() has already advanced released sites
            z_pin = sch.add_noise(z0, noise, ts[i + 1].view(1))
            # Edge behaviour to be aware of if asked: the sigmoid is soft, so
            # c=1 still reaches gate=0.5 at i=N-2 and c=0 starts at gate~0.66
            # rather than exactly 1. Neither matters in practice because the
            # final pixel-space composite is driven by conf, but do not claim
            # the pinning is exact. Shrink --softness to sharpen the gate.
            gate = torch.sigmoid(((i + 1) - release) / softness).to(unet_dtype)
            z = gate * z_model + (1 - gate) * z_pin
        else:
            z = z_model

    img = vae.decode((z / vae.config.scaling_factor).to(vae_dtype)).sample
    img = ((img.squeeze(0).float() + 1) / 2).clamp(0, 1)
    img = _back(img.to(rgb.device), hw)
    return (conf * rgb + (1 - conf) * img).clamp(0, 1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
class Metrics:
    """PSNR / SSIM / LPIPS against ground truth.

    Held-out views have ground truth -- we simply never trained on them. Using
    it for EVALUATION is legitimate; using it inside compute_confidence would
    not be, since the whole premise is localising artifacts blind.
    """

    def __init__(self, use_lpips=False, lpips_net="alex", device="cuda"):
        self.lpips = None
        if use_lpips:
            try:
                import lpips as _lpips
                self.lpips = _lpips.LPIPS(net=lpips_net).to(device).eval()
            except Exception as e:
                print(f"[warn] LPIPS unavailable ({e}); skipping it.")

    @staticmethod
    def psnr(a, b):
        mse = ((a - b) ** 2).mean()
        return float(10.0 * torch.log10(1.0 / mse.clamp(min=1e-12)))

    @staticmethod
    def ssim(a, b):
        if not SSIM_AVAILABLE:
            return None
        return float(_repo_ssim(a, b))

    def __call__(self, pred, gt):
        out = {"psnr": self.psnr(pred, gt), "ssim": self.ssim(pred, gt)}
        if self.lpips is not None:
            with torch.no_grad():
                d = self.lpips(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1)
            out["lpips"] = float(d.flatten()[0])
        return out


def get_gt(cam):
    gt = getattr(cam, "original_image", None)
    if gt is None:
        return None
    return gt.cuda().clamp(0, 1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser():
    parser = ArgumentParser("Step 3: diffusion repair of sparse-view 3DGS")
    lp_ = ModelParams(parser, sentinel=True)
    pp_ = PipelineParams(parser)

    # NOTE: every argument below must have a non-None default. get_combined_args
    # drops command-line values that are None, so a None-default arg the user
    # does not pass simply will not exist on the namespace.
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--view_set", default="test", choices=["test", "train"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)

    parser.add_argument("--repair_mode", default="conf_schedule",
                        choices=["conf_schedule", "inpaint", "none"])
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--diff_steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--diff_seed", type=int, default=0)
    parser.add_argument("--max_strength", type=float, default=0.85)
    parser.add_argument("--softness", type=float, default=1.5)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--sd_res", type=int, default=512)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--vae_fp16", action="store_true", default=False,
                        help="keep the VAE in fp16 (default is fp32: the SD2 "
                             "VAE emits NaNs/black frames at half precision)")

    # confidence cues -- alpha only by default, see compute_confidence docstring
    parser.add_argument("--alpha_thresh", type=float, default=0.40)
    parser.add_argument("--depth_grad_thresh", type=float, default=0.25)
    parser.add_argument("--near_pct", type=float, default=0.98)
    parser.add_argument("--w_alpha", type=float, default=1.0)
    parser.add_argument("--w_grad", type=float, default=0.0)
    parser.add_argument("--w_near", type=float, default=0.0)
    parser.add_argument("--dilate_k", type=int, default=5)
    parser.add_argument("--blur_k", type=int, default=9)

    parser.add_argument("--no_metrics", action="store_true", default=False)
    parser.add_argument("--lpips", action="store_true", default=False)
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    parser.add_argument("--save_conf_npy", action="store_true", default=False,
                        help="dump confidence as float32 .npy for use as a "
                             "per-pixel loss weight in step 4")
    parser.add_argument("--save_gt", action="store_true", default=False,
                        help="dump ground-truth PNGs alongside the renders so "
                             "a second environment can score results without "
                             "loading the Scene (needed if diffusion runs in a "
                             "separate env from the rasterizer)")
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser, lp_, pp_


def summarise(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    parser, lp_, pp_ = build_parser()
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset, pipe_cfg = lp_.extract(args), pp_.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.load_iteration,
                      shuffle=False)

    cams = scene.getTestCameras() if args.view_set == "test" \
        else scene.getTrainCameras()
    if len(cams) == 0:
        other = "train" if args.view_set == "test" else "test"
        sys.exit(f"No {args.view_set} cameras. Try --view_set {other}. "
                 f"(If both are empty, cfg_args probably has eval=False.)")
    cams = cams[::args.stride]
    if args.limit:
        cams = cams[:args.limit]

    root = os.path.expanduser(args.outdir) if args.outdir \
        else os.path.join(dataset.model_path, "repair")
    keys = ["renders", "confidence", "repaired", "debug"]
    dirs = {k: os.path.join(root, k) for k in keys}
    if args.save_conf_npy:
        dirs["conf_npy"] = os.path.join(root, "conf_npy")
    if args.save_gt:
        dirs["gt"] = os.path.join(root, "gt")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"model     : {dataset.model_path} @ iter {scene.loaded_iter}")
    print(f"gaussians : {gaussians.get_xyz.shape[0]}")
    print(f"views     : {len(cams)} ({args.view_set})")
    print(f"mode      : {args.repair_mode}")
    print(f"cues      : w_alpha={args.w_alpha} w_grad={args.w_grad} "
          f"w_near={args.w_near} alpha_thresh={args.alpha_thresh}")
    print(f"out       : {root}")

    pipe_sd = None
    if args.repair_mode != "none":
        print("loading latent diffusion model ...")
        pipe_sd = load_repairer(args.repair_mode, precision=args.precision,
                                vae_fp32=not args.vae_fp16)

    metric = None if args.no_metrics else Metrics(args.lpips, args.lpips_net)

    manifest = []
    for cam in tqdm(cams, desc=f"repair[{args.repair_mode}]"):
        name = cam.image_name
        out_png = os.path.join(dirs["repaired"], f"{name}.png")
        if args.skip_existing and os.path.exists(out_png):
            continue

        rgb, alpha, inv_d = render_with_alpha(cam, gaussians, pipe_cfg)
        conf = compute_confidence(
            alpha, inv_d,
            alpha_thresh=args.alpha_thresh,
            depth_grad_thresh=args.depth_grad_thresh,
            near_pct=args.near_pct,
            w_alpha=args.w_alpha, w_grad=args.w_grad, w_near=args.w_near,
            dilate_k=args.dilate_k, blur_k=args.blur_k)

        if args.repair_mode == "none":
            fixed = rgb.clone()
        elif args.repair_mode == "inpaint":
            fixed = repair_binary(pipe_sd, rgb, conf, args.prompt,
                                  conf_thresh=args.conf_thresh,
                                  steps=args.diff_steps, guidance=7.5,
                                  seed=args.diff_seed, sd_res=args.sd_res)
        else:
            fixed = repair_conf_schedule(pipe_sd, rgb, conf, args.prompt,
                                         steps=args.diff_steps,
                                         guidance=args.guidance,
                                         seed=args.diff_seed,
                                         max_strength=args.max_strength,
                                         softness=args.softness,
                                         sd_res=args.sd_res)

        row = {"name": name,
               "masked_fraction": float((1 - conf).mean().item()),
               "mean_alpha": float(alpha.mean().item())}

        if metric is not None or args.save_gt:
            gt = get_gt(cam)
            if gt is not None:
                if metric is not None:
                    for tag, im in (("render", rgb), ("repaired", fixed)):
                        for k, v in metric(im, gt).items():
                            row[f"{tag}_{k}"] = v
                if args.save_gt:
                    save_img(os.path.join(dirs["gt"], f"{name}.png"), gt)
                del gt

        save_img(os.path.join(dirs["renders"], f"{name}.png"), rgb)
        save_img(os.path.join(dirs["confidence"], f"{name}.png"), conf)
        save_img(out_png, fixed)
        save_debug_grid(os.path.join(dirs["debug"], f"{name}_cues.png"),
                        rgb, alpha, inv_d, conf)
        save_img(os.path.join(dirs["debug"], f"{name}_before_after.png"),
                 torch.cat([rgb, fixed], dim=2))
        if args.save_conf_npy:
            import numpy as np
            np.save(os.path.join(dirs["conf_npy"], f"{name}.npy"),
                    conf[0].detach().float().cpu().numpy())

        manifest.append(row)
        del rgb, alpha, inv_d, conf, fixed
        torch.cuda.empty_cache()

    summary = {"mean_masked_fraction": summarise(manifest, "masked_fraction")}
    for tag in ("render", "repaired"):
        for k in ("psnr", "ssim", "lpips"):
            v = summarise(manifest, f"{tag}_{k}")
            if v is not None:
                summary[f"{tag}_{k}"] = v

    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                   "iteration": scene.loaded_iter,
                   "n_views": len(manifest),
                   "summary": summary,
                   "views": manifest}, f, indent=2)

    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"{k:24s} {v:.4f}" if v is not None else f"{k:24s} n/a")
    if "render_psnr" in summary and "repaired_psnr" in summary:
        d = summary["repaired_psnr"] - summary["render_psnr"]
        if abs(d) < 1e-6:
            verdict = "NO-OP (expected for --repair_mode none)"
        else:
            verdict = "HELPED" if d > 0 else "HURT"
        print(f"{'delta_psnr':24s} {d:+.4f}   <- repair {verdict}")
    print(f"\nLOOK AT: {dirs['debug']}/*_cues.png -- is the mask on the floaters?")


if __name__ == "__main__":
    main()