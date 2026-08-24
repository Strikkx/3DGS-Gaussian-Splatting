"""Step 3: diffusion repair of sparse-view 3DGS renders. Self-contained."""
import os, sys, json
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

DEFAULT_PROMPT = ("a sharp outdoor photograph of a bicycle leaning against a "
                  "bench on grass, trees and a lawn in the background, "
                  "natural daylight, high detail, photorealistic")


# ---------- image IO (PIL only; torchvision is broken in this env) ----------
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


# ---------- artifact localisation (no ground truth available) ----------
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
    p99 = torch.quantile(mag.flatten().float(), 0.99).clamp(min=1e-6)
    return (mag / p99).clamp(0, 1)


def _dilate(m, k):
    return F.max_pool2d(m.unsqueeze(0), k, 1, k // 2).squeeze(0)


def _blur(m, k):
    sigma = max(k / 3.0, 1e-3)
    c = torch.arange(k, device=m.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    x = m.unsqueeze(0)
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    x = F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return x.squeeze(0).clamp(0, 1)


def compute_confidence(alpha, inv_depth, alpha_thresh=0.55,
                       depth_grad_thresh=0.25, near_pct=0.98,
                       w_alpha=1.0, w_grad=0.7, w_near=0.7,
                       dilate_k=9, blur_k=15):
    """Returns confidence [1,H,W]: 1 = trust, 0 = garbage. Three cues:
    low alpha (holes), high depth gradient (floater edges), near depth
    (floater interiors). Continuous, not binary -- thresholding it recovers
    the standard binary-inpainting baseline we ablate against."""
    a_bad = ((alpha_thresh - alpha) / max(alpha_thresh, 1e-6)).clamp(0, 1)
    if inv_depth is not None:
        d = inv_depth.float()
        lo = torch.quantile(d.flatten(), 0.01)
        hi = torch.quantile(d.flatten(), 0.99).clamp(min=lo + 1e-6)
        dn = ((d - lo) / (hi - lo)).clamp(0, 1)
        g_bad = (_sobel(dn) - depth_grad_thresh).clamp(min=0.0)
        g_bad = (g_bad / max(1.0 - depth_grad_thresh, 1e-6)).clamp(0, 1)
        cut = torch.quantile(dn.flatten(), near_pct)
        n_bad = ((dn - cut) / (1.0 - cut).clamp(min=1e-6)).clamp(0, 1)
    else:
        g_bad = n_bad = torch.zeros_like(alpha)
    # soft-OR by max: any single strong cue should condemn a pixel
    bad = torch.maximum(torch.maximum(w_alpha * a_bad, w_grad * g_bad),
                        w_near * n_bad).clamp(0, 1)
    bad = _blur(_dilate(bad, dilate_k), blur_k)
    return (1.0 - bad).clamp(0, 1)


# ---------- latent diffusion repair ----------
def load_repairer(mode, dtype=torch.float16, device="cuda"):
    from diffusers import (StableDiffusionInpaintPipeline,
                           StableDiffusionPipeline, DDIMScheduler)
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
    try:
        p.enable_attention_slicing()
    except Exception:
        pass
    return p


def _fit(x, long=512):
    """SD is 512-native, VAE needs dims divisible by 8. Downsample, repair,
    upsample. Real limitation: detail cannot be restored at 512."""
    _, h, w = x.shape
    s = long / max(h, w)
    nh = max(int(round(h * s)) // 8 * 8, 8)
    nw = max(int(round(w * s)) // 8 * 8, 8)
    return F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear",
                         align_corners=False).squeeze(0), (h, w)


def _back(x, hw):
    return F.interpolate(x.unsqueeze(0), size=hw, mode="bilinear",
                         align_corners=False).squeeze(0)


@torch.no_grad()
def repair_binary(pipe, rgb, conf, prompt, conf_thresh=0.5, steps=30,
                  guidance=7.5, seed=0, sd_res=512):
    """BASELINE: threshold confidence into a binary mask, standard inpainting."""
    from PIL import Image
    import numpy as np
    small, hw = _fit(rgb, sd_res)
    cs, _ = _fit(conf, sd_res)
    mask = (cs < conf_thresh).float()
    img = Image.fromarray(_u8(small))
    msk = Image.fromarray((mask[0].cpu().numpy() * 255).astype(np.uint8)).convert("L")
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(prompt=prompt, image=img, mask_image=msk,
               num_inference_steps=steps, guidance_scale=guidance,
               generator=g, height=small.shape[1], width=small.shape[2]).images[0]
    t = torch.from_numpy(np.array(out).astype("float32") / 255.0)
    t = _back(t.permute(2, 0, 1).to(rgb.device), hw)
    # composite in pixel space vs the ORIGINAL full-res render, else every
    # trusted pixel eats a VAE round-trip and PSNR collapses
    return (conf * rgb + (1 - conf) * t).clamp(0, 1)


@torch.no_grad()
def repair_conf_schedule(pipe, rgb, conf, prompt, steps=30, guidance=6.0,
                         seed=0, max_strength=0.85, softness=1.5, sd_res=512):
    """PROPOSED: no threshold. Per-pixel release timestep t_start = T*(1-c).
    Each latent site stays pinned to its forward-noised original until the
    schedule reaches its release step. c=0 -> fully regenerated, c=1 ->
    preserved. Binary inpainting is the c in {0,1} special case."""
    device, dtype = pipe.device, pipe.unet.dtype
    vae, unet, sch = pipe.vae, pipe.unet, pipe.scheduler
    small, hw = _fit(rgb, sd_res)
    cs, _ = _fit(conf, sd_res)
    x = small.unsqueeze(0).to(device, dtype) * 2 - 1
    z0 = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
    # area-pool to latent res, not nearest: a sub-8px floater would vanish
    c_lat = F.interpolate(cs.unsqueeze(0).to(device, torch.float32),
                          size=z0.shape[-2:], mode="area")
    sch.set_timesteps(steps, device=device)
    ts = sch.timesteps[int(steps * (1 - max_strength)):]
    N = len(ts)
    release = c_lat * (N - 1)
    # one fixed noise draw shared across views: cheap partial consistency prior
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z0.shape, generator=g, device=device).to(dtype)
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
            # pin at t_next, not t: step() already advanced released sites
            z_pin = sch.add_noise(z0, noise, ts[i + 1].view(1))
            gate = torch.sigmoid(((i + 1) - release) / softness).to(dtype)
            z = gate * z_model + (1 - gate) * z_pin
        else:
            z = z_model
    img = vae.decode(z / vae.config.scaling_factor).sample
    img = ((img.squeeze(0).float() + 1) / 2).clamp(0, 1)
    img = _back(img.to(rgb.device), hw)
    return (conf * rgb + (1 - conf) * img).clamp(0, 1)


# ---------- main ----------
def main():
    parser = ArgumentParser("Step 3: diffusion repair of sparse-view 3DGS")
    lp_ = ModelParams(parser, sentinel=True)
    pp_ = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--outdir", default="", type=str)
    parser.add_argument("--view_set", default="test", choices=["test", "train"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--repair_mode", default="conf_schedule",
                        choices=["conf_schedule", "inpaint", "none"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--diff_steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--diff_seed", type=int, default=0)
    parser.add_argument("--max_strength", type=float, default=0.85)
    parser.add_argument("--softness", type=float, default=1.5)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--sd_res", type=int, default=512)
    parser.add_argument("--alpha_thresh", type=float, default=0.55)
    parser.add_argument("--depth_grad_thresh", type=float, default=0.25)
    parser.add_argument("--w_alpha", type=float, default=1.0)
    parser.add_argument("--w_grad", type=float, default=0.7)
    parser.add_argument("--w_near", type=float, default=0.7)
    parser.add_argument("--dilate_k", type=int, default=9)
    parser.add_argument("--blur_k", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
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
        sys.exit(f"No {args.view_set} cameras. Try --view_set {other}.")
    cams = cams[::args.stride]
    if args.limit:
        cams = cams[:args.limit]

    root = os.path.expanduser(args.outdir) if args.outdir \
        else os.path.join(dataset.model_path, "repair")
    dirs = {k: os.path.join(root, k)
            for k in ["renders", "confidence", "repaired", "debug"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"model     : {dataset.model_path} @ iter {scene.loaded_iter}")
    print(f"gaussians : {gaussians.get_xyz.shape[0]}")
    print(f"views     : {len(cams)} ({args.view_set})")
    print(f"mode      : {args.repair_mode}")
    print(f"out       : {root}")

    pipe_sd = None
    if args.repair_mode != "none":
        print("loading latent diffusion model ...")
        pipe_sd = load_repairer(args.repair_mode)

    manifest = []
    for cam in tqdm(cams, desc=f"repair[{args.repair_mode}]"):
        name = cam.image_name
        out_png = os.path.join(dirs["repaired"], f"{name}.png")
        if args.skip_existing and os.path.exists(out_png):
            continue

        rgb, alpha, inv_d = render_with_alpha(cam, gaussians, pipe_cfg)
        conf = compute_confidence(
            alpha, inv_d, alpha_thresh=args.alpha_thresh,
            depth_grad_thresh=args.depth_grad_thresh, w_alpha=args.w_alpha,
            w_grad=args.w_grad, w_near=args.w_near,
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

        save_img(os.path.join(dirs["renders"], f"{name}.png"), rgb)
        save_img(os.path.join(dirs["confidence"], f"{name}.png"), conf)
        save_img(out_png, fixed)
        save_debug_grid(os.path.join(dirs["debug"], f"{name}_cues.png"),
                        rgb, alpha, inv_d, conf)
        save_img(os.path.join(dirs["debug"], f"{name}_before_after.png"),
                 torch.cat([rgb, fixed], dim=2))

        manifest.append({"name": name,
                         "masked_fraction": float((1 - conf).mean().item()),
                         "mean_alpha": float(alpha.mean().item())})
        del rgb, alpha, inv_d, conf, fixed
        torch.cuda.empty_cache()

    fr = [m["masked_fraction"] for m in manifest]
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                   "iteration": scene.loaded_iter, "n_views": len(manifest),
                   "mean_masked_fraction": sum(fr) / max(len(fr), 1),
                   "views": manifest}, f, indent=2)
    print(f"\nmean masked fraction = {sum(fr) / max(len(fr), 1):.3f}")
    print(f"LOOK AT: {dirs['debug']}/*_cues.png -- is the mask on the floaters?")


if __name__ == "__main__":
    main()