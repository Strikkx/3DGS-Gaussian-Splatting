"""Step 3b: diffusion repair that runs on ALREADY-RENDERED images.

Why this file exists
--------------------
repair_renders.py needs the 3DGS stack (diff_gaussian_rasterization, simple_knn)
which is compiled against one specific torch build. If that same environment
cannot import diffusers -- e.g. because site-packages has two torch versions
layered on top of each other -- you are stuck: the rasterizer pins you to the
broken torch, and fixing the torch risks the rasterizer.

So split the pipeline at the image boundary:

  env A (has the rasterizer, torch pinned, works today)
      python repair_renders.py -m <model> --repair_mode none \
          --save_conf_npy --save_gt --outdir <D>
      -> <D>/renders/*.png  <D>/confidence/*.png  <D>/conf_npy/*.npy  <D>/gt/*.png

  env B (clean: torch + diffusers + pillow, nothing compiled)
      python repair_images.py --indir <D> --repair_mode conf_schedule \
          --outdir <D>_sched

This file imports NOTHING from the gaussian-splatting repo. Torch, diffusers,
numpy and PIL only. SSIM is reimplemented here with the same 11x11 / sigma=1.5
Gaussian window the 3DGS repo uses, so the numbers stay comparable.

Fidelity note worth stating in the write-up: confidence read back from an 8-bit
PNG is quantised to 1/255. Pass --use_npy (default when conf_npy/ exists) to
read the float32 dump instead, so the schedule sees exactly the confidence the
renderer computed.
"""
import os
import sys
import json
import math
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from tqdm import tqdm

DEFAULT_PROMPT = ("a sharp outdoor photograph of a bicycle leaning against a "
                  "bench on grass, trees and a lawn in the background, "
                  "natural daylight, high detail, photorealistic")


# --------------------------------------------------------------------------
# image IO
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


def load_img(path, device="cuda", gray=False):
    from PIL import Image
    import numpy as np
    im = Image.open(path)
    im = im.convert("L" if gray else "RGB")
    a = np.array(im).astype("float32") / 255.0
    t = torch.from_numpy(a)
    t = t.unsqueeze(0) if gray else t.permute(2, 0, 1)
    return t.to(device).clamp(0, 1)


def load_npy(path, device="cuda"):
    import numpy as np
    a = np.load(path).astype("float32")
    t = torch.from_numpy(a)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.to(device).clamp(0, 1)


# --------------------------------------------------------------------------
# metrics (self-contained; matches the 3DGS repo's SSIM parameters)
# --------------------------------------------------------------------------
def _gauss_window(ws, sigma, channel, device, dtype):
    g = torch.tensor([math.exp(-((x - ws // 2) ** 2) / (2.0 * sigma ** 2))
                      for x in range(ws)], device=device, dtype=dtype)
    g = (g / g.sum()).unsqueeze(1)
    w2 = g.mm(g.t()).unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, ws, ws).contiguous()


def ssim(a, b, ws=11, sigma=1.5):
    """a, b: [C,H,W] in [0,1]."""
    a4, b4 = a.unsqueeze(0), b.unsqueeze(0)
    c = a4.shape[1]
    w = _gauss_window(ws, sigma, c, a.device, a.dtype)
    pad = ws // 2
    mu1 = F.conv2d(a4, w, padding=pad, groups=c)
    mu2 = F.conv2d(b4, w, padding=pad, groups=c)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a4 * a4, w, padding=pad, groups=c) - mu1s
    s2 = F.conv2d(b4 * b4, w, padding=pad, groups=c) - mu2s
    s12 = F.conv2d(a4 * b4, w, padding=pad, groups=c) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))
    return float(m.mean())


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(10.0 * torch.log10(1.0 / mse.clamp(min=1e-12)))


class Metrics:
    def __init__(self, use_lpips=False, net="alex", device="cuda"):
        self.lpips = None
        if use_lpips:
            try:
                import lpips as _lpips
                self.lpips = _lpips.LPIPS(net=net).to(device).eval()
            except Exception as e:
                print(f"[warn] LPIPS unavailable ({e}); continuing without it.")

    def __call__(self, pred, gt):
        out = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
        if self.lpips is not None:
            with torch.no_grad():
                out["lpips"] = float(self.lpips(pred.unsqueeze(0) * 2 - 1,
                                                gt.unsqueeze(0) * 2 - 1).flatten()[0])
        return out


# --------------------------------------------------------------------------
# diffusion (identical maths to repair_renders.py -- keep the two in sync)
# --------------------------------------------------------------------------
def color_match(img, rgb, conf, trust=0.9, eps=1e-5):
    """Affine-correct img's colour statistics to the render's, estimated ONLY
    over high-confidence pixels where the two should agree.

    Rationale: the trusted region gives paired samples of "what the renderer
    produced" vs "what the VAE round-trip produced". Any systematic gain/offset
    between them is a pipeline artifact, not scene content, so removing it is
    principled. It does NOT fix semantic colour errors -- if SD decides the sky
    is teal, this will not turn it grey, because the void has no paired data.
    """
    m = (conf > trust).float()
    if float(m.sum()) < 100:
        return img

    def stats(x):
        n = m.sum().clamp(min=1)
        mean = (x * m).sum(dim=(1, 2), keepdim=True) / n
        var = (((x - mean) ** 2) * m).sum(dim=(1, 2), keepdim=True) / n
        return mean, var.clamp(min=eps).sqrt()

    mi, si = stats(img)
    mr, sr = stats(rgb)
    return ((img - mi) / si * sr + mr).clamp(0, 1)


@torch.no_grad()
def repair_meanfill(rgb, conf, fill_color=None, trust=0.9):
    """TRIVIAL BASELINE: no diffusion at all. Fill the untrusted region with a
    single colour -- either an explicit one or the mean of the trusted pixels.

    Worth running before believing any diffusion result. PSNR is dominated by
    getting a region's MEAN right, and a flat fill gets that for free while a
    generative model can easily get it wrong in an interesting-looking way.
    If diffusion cannot beat this, say so.
    """
    if fill_color is not None:
        c = torch.tensor(fill_color, device=rgb.device,
                         dtype=rgb.dtype).view(3, 1, 1)
    else:
        m = (conf > trust).float()
        c = (rgb * m).sum(dim=(1, 2), keepdim=True) / m.sum().clamp(min=1)
    return (conf * rgb + (1 - conf) * c).clamp(0, 1)


def load_repairer(mode, precision="fp16", vae_fp32=True, device="cuda",
                  model_id=""):
    from diffusers import (StableDiffusionInpaintPipeline,
                           StableDiffusionPipeline, DDIMScheduler)
    dtype = torch.float16 if precision == "fp16" else torch.float32

    if mode == "inpaint":
        mid = model_id or "stabilityai/stable-diffusion-2-inpainting"
        p = StableDiffusionInpaintPipeline.from_pretrained(
            mid, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
    else:
        mid = model_id or "stabilityai/stable-diffusion-2-1-base"
        p = StableDiffusionPipeline.from_pretrained(
            mid, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
    print(f"checkpoint: {mid}")

    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    p = p.to(device)
    p.set_progress_bar_config(disable=True)
    # SD2's VAE is unstable in fp16 (NaNs / black frames). The UNet is fine.
    if vae_fp32 and mode != "inpaint" and dtype == torch.float16:
        p.vae = p.vae.to(torch.float32)
    try:
        p.enable_attention_slicing()
    except Exception:
        pass
    return p


def _fit(x, long=512, mode="bilinear"):
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
                  guidance=7.5, seed=0, sd_res=512, negative_prompt="",
                  do_color_match=False):
    """BASELINE: hard threshold -> standard inpainting. Composites with the
    same binary mask it inpainted with, so the ablation isolates one thing."""
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
               negative_prompt=negative_prompt or None,
               num_inference_steps=steps, guidance_scale=guidance,
               generator=g, height=small.shape[1], width=small.shape[2]).images[0]

    t = torch.from_numpy(np.array(out).astype("float32") / 255.0)
    t = _back(t.permute(2, 0, 1).to(rgb.device), hw)
    if do_color_match:
        t = color_match(t, rgb, conf)
    mask_full = (conf < conf_thresh).float()
    return ((1 - mask_full) * rgb + mask_full * t).clamp(0, 1)


@torch.no_grad()
def repair_conf_schedule(pipe, rgb, conf, prompt, steps=30, guidance=6.0,
                         seed=0, max_strength=0.85, softness=1.5, sd_res=512,
                         negative_prompt="", do_color_match=False):
    """PROPOSED: per-pixel release timestep t_start = T*(1-c). No threshold."""
    device = pipe.device
    unet_dtype = pipe.unet.dtype
    vae, unet, sch = pipe.vae, pipe.unet, pipe.scheduler
    vae_dtype = next(vae.parameters()).dtype

    small, hw = _fit(rgb, sd_res)
    cs, _ = _fit(conf, sd_res, mode="area")

    x = small.unsqueeze(0).to(device=device, dtype=vae_dtype) * 2 - 1
    z0 = (vae.encode(x).latent_dist.mean * vae.config.scaling_factor).to(unet_dtype)

    c_lat = F.interpolate(cs.unsqueeze(0).to(device, torch.float32),
                          size=z0.shape[-2:], mode="area")

    sch.set_timesteps(steps, device=device)
    ts = sch.timesteps[int(steps * (1 - max_strength)):]
    N = len(ts)
    release = c_lat * (N - 1)

    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z0.shape, generator=g, device=device,
                        dtype=torch.float32).to(unet_dtype)

    if hasattr(pipe, "encode_prompt"):
        pos, neg = pipe.encode_prompt(prompt, device, 1, guidance > 1.0,
                                      negative_prompt)
        emb = torch.cat([neg, pos]) if guidance > 1.0 else pos
    else:
        emb = pipe._encode_prompt(prompt, device, 1, guidance > 1.0,
                                  negative_prompt)

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
            z_pin = sch.add_noise(z0, noise, ts[i + 1].view(1))
            gate = torch.sigmoid(((i + 1) - release) / softness).to(unet_dtype)
            z = gate * z_model + (1 - gate) * z_pin
        else:
            z = z_model

    img = vae.decode((z / vae.config.scaling_factor).to(vae_dtype)).sample
    img = ((img.squeeze(0).float() + 1) / 2).clamp(0, 1)
    img = _back(img.to(rgb.device), hw)
    if do_color_match:
        img = color_match(img, rgb, conf)
    return (conf * rgb + (1 - conf) * img).clamp(0, 1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def summarise(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return sum(v) / len(v) if v else None


def main():
    ap = ArgumentParser("Step 3b: diffusion repair of pre-rendered images")
    ap.add_argument("--indir", required=True,
                    help="a repair_renders.py output dir (needs renders/ and "
                         "confidence/ or conf_npy/)")
    ap.add_argument("--outdir", default="",
                    help="defaults to <indir>_<repair_mode>")
    ap.add_argument("--gt_dir", default="",
                    help="defaults to <indir>/gt if it exists")
    ap.add_argument("--repair_mode", default="conf_schedule",
                    choices=["conf_schedule", "inpaint", "meanfill", "none"])
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative_prompt", default="",
                    help="e.g. 'bicycle, bench, object, dramatic sky, sunset, "
                         "teal, oversaturated' -- steers CFG away from the "
                         "caption content the model otherwise paints into holes")
    ap.add_argument("--color_match", action="store_true", default=False,
                    help="affine-match the diffusion output's colour stats to "
                         "the render's, estimated over trusted pixels only")
    ap.add_argument("--fill_color", default="",
                    help="meanfill only: 'R,G,B' in 0-1 (e.g. 0.75,0.76,0.78). "
                         "Default is the mean of the trusted pixels.")
    ap.add_argument("--diff_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--diff_seed", type=int, default=0)
    ap.add_argument("--max_strength", type=float, default=0.85)
    ap.add_argument("--softness", type=float, default=1.5)
    ap.add_argument("--conf_thresh", type=float, default=0.5)
    ap.add_argument("--sd_res", type=int, default=512)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--vae_fp16", action="store_true", default=False)
    ap.add_argument("--sd_model", default="",
                    help="HF repo id or local directory to load instead of "
                         "the default checkpoint for this --repair_mode")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--skip_existing", action="store_true", default=False)
    ap.add_argument("--use_png_conf", action="store_true", default=False,
                    help="force 8-bit PNG confidence even when conf_npy/ exists")
    ap.add_argument("--lpips", action="store_true", default=False)
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--no_metrics", action="store_true", default=False)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(sys.argv[1:])

    indir = os.path.expanduser(args.indir)
    rend_dir = os.path.join(indir, "renders")
    conf_png_dir = os.path.join(indir, "confidence")
    conf_npy_dir = os.path.join(indir, "conf_npy")
    if not os.path.isdir(rend_dir):
        sys.exit(f"{rend_dir} not found. Run repair_renders.py --repair_mode "
                 f"none --save_conf_npy --save_gt first.")

    use_npy = os.path.isdir(conf_npy_dir) and not args.use_png_conf
    if not use_npy and not os.path.isdir(conf_png_dir):
        sys.exit("Neither conf_npy/ nor confidence/ present in --indir.")

    gt_dir = os.path.expanduser(args.gt_dir) if args.gt_dir \
        else os.path.join(indir, "gt")
    has_gt = os.path.isdir(gt_dir)

    root = os.path.expanduser(args.outdir) if args.outdir \
        else f"{indir.rstrip('/')}_{args.repair_mode}"
    dirs = {k: os.path.join(root, k) for k in ("repaired", "debug")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(rend_dir)
                   if f.lower().endswith(".png"))
    names = names[::args.stride]
    if args.limit:
        names = names[:args.limit]
    if not names:
        sys.exit(f"No PNGs in {rend_dir}")

    print(f"in        : {indir}")
    print(f"views     : {len(names)}")
    print(f"confidence: {'conf_npy (float32)' if use_npy else 'confidence PNG (8-bit)'}")
    print(f"ground tr.: {gt_dir if has_gt else 'ABSENT -- metrics disabled'}")
    print(f"mode      : {args.repair_mode}")
    print(f"out       : {root}")

    fill_color = None
    if args.fill_color:
        fill_color = [float(v) for v in args.fill_color.split(",")]
        if len(fill_color) != 3:
            sys.exit("--fill_color needs three comma-separated values in 0-1")

    pipe = None
    if args.repair_mode in ("conf_schedule", "inpaint"):
        print("loading latent diffusion model ...")
        pipe = load_repairer(args.repair_mode, precision=args.precision,
                             vae_fp32=not args.vae_fp16, device=args.device,
                             model_id=args.sd_model)

    metric = None
    if has_gt and not args.no_metrics:
        metric = Metrics(args.lpips, args.lpips_net, args.device)

    manifest = []
    for name in tqdm(names, desc=f"repair[{args.repair_mode}]"):
        out_png = os.path.join(dirs["repaired"], f"{name}.png")
        if args.skip_existing and os.path.exists(out_png):
            continue

        rgb = load_img(os.path.join(rend_dir, f"{name}.png"), args.device)
        if use_npy:
            conf = load_npy(os.path.join(conf_npy_dir, f"{name}.npy"), args.device)
        else:
            conf = load_img(os.path.join(conf_png_dir, f"{name}.png"),
                            args.device, gray=True)
        if conf.shape[-2:] != rgb.shape[-2:]:
            conf = _back(conf, rgb.shape[-2:])

        if args.repair_mode == "none":
            fixed = rgb.clone()
        elif args.repair_mode == "meanfill":
            fixed = repair_meanfill(rgb, conf, fill_color=fill_color)
        elif args.repair_mode == "inpaint":
            fixed = repair_binary(pipe, rgb, conf, args.prompt,
                                  conf_thresh=args.conf_thresh,
                                  steps=args.diff_steps, guidance=7.5,
                                  seed=args.diff_seed, sd_res=args.sd_res,
                                  negative_prompt=args.negative_prompt,
                                  do_color_match=args.color_match)
        else:
            fixed = repair_conf_schedule(pipe, rgb, conf, args.prompt,
                                         steps=args.diff_steps,
                                         guidance=args.guidance,
                                         seed=args.diff_seed,
                                         max_strength=args.max_strength,
                                         softness=args.softness,
                                         sd_res=args.sd_res,
                                         negative_prompt=args.negative_prompt,
                                         do_color_match=args.color_match)

        row = {"name": name, "masked_fraction": float((1 - conf).mean().item())}
        if metric is not None:
            gt_path = os.path.join(gt_dir, f"{name}.png")
            if os.path.exists(gt_path):
                gt = load_img(gt_path, args.device)
                if gt.shape == rgb.shape:
                    for tag, im in (("render", rgb), ("repaired", fixed)):
                        for k, v in metric(im, gt).items():
                            row[f"{tag}_{k}"] = v
                del gt

        save_img(out_png, fixed)
        save_img(os.path.join(dirs["debug"], f"{name}_before_after.png"),
                 torch.cat([rgb, fixed], dim=2))
        manifest.append(row)
        del rgb, conf, fixed
        if args.device == "cuda":
            torch.cuda.empty_cache()

    summary = {"mean_masked_fraction": summarise(manifest, "masked_fraction")}
    for tag in ("render", "repaired"):
        for k in ("psnr", "ssim", "lpips"):
            v = summarise(manifest, f"{tag}_{k}")
            if v is not None:
                summary[f"{tag}_{k}"] = v

    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"args": vars(args), "n_views": len(manifest),
                   "summary": summary, "views": manifest}, f, indent=2)

    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"{k:24s} {v:.4f}" if v is not None else f"{k:24s} n/a")
    if "render_psnr" in summary and "repaired_psnr" in summary:
        d = summary["repaired_psnr"] - summary["render_psnr"]
        verdict = ("NO-OP" if abs(d) < 1e-6
                   else ("HELPED" if d > 0 else "HURT"))
        print(f"{'delta_psnr':24s} {d:+.4f}   <- repair {verdict}")
    print(f"\nLOOK AT: {dirs['debug']}/*_before_after.png")


if __name__ == "__main__":
    main()