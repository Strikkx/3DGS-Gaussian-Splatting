"""Step 4: fine-tune a sparse-view 3DGS model on diffusion-repaired pseudo-views.

Runs in the ENV WITH THE RASTERIZER (karthik_gs3d), not the diffusion env.

What this does
--------------
Resume from the 5-view checkpoint and keep optimising, but now with extra
supervision: the 189 held-out cameras are fed back in with the REPAIRED images
standing in for ground truth. Because those images are partly invented, every
pseudo-view pixel is weighted by the confidence map from step 3, so the
Gaussians are told exactly how much to trust each pixel.

The honest-evaluation problem
-----------------------------
If you supervise on all 189 views you no longer have a test set -- every view
with ground truth has been trained on. So the 189 are split deterministically:
`--n_holdout` of them are NEVER used for supervision and NEVER repaired, and
all reported numbers come from those. The split is seeded and written to
split.json so that every ablation run, and every later round of the
alternating loop, uses exactly the same one.

Design choices you should be able to defend
-------------------------------------------
* Real views are OVERSAMPLED (--real_ratio). 5 real views against 149 pseudo
  ones would otherwise be drowned out, and the pseudo views are the ones
  containing hallucinations. The real views are the only thing anchoring the
  model to the actual scene.
* Densification is OFF by default. Splitting and cloning Gaussians to fit
  invented sky texture is how you permanently bake a hallucination into the
  geometry. Turn it on with --densify only as a deliberate experiment.
* The learning rate continues the original schedule rather than restarting it:
  update_learning_rate(loaded_iter + i) returns position_lr_final, because
  get_expon_lr_func clamps t to 1. Restarting from position_lr_init would
  scatter the geometry you already paid 30k iterations for.
* Rendering uses use_trained_exp=False to match how the step-3 renders were
  produced. If you flip this, the pseudo targets no longer align with what the
  renderer outputs and the loss fights the exposure model.
"""
import os
import sys
import json
import random
from argparse import ArgumentParser, Namespace

import torch

# --- ENVIRONMENT WORKAROUND, not part of the method ------------------------
# site-packages reports torch 2.3.1 but holds 2.6-era torch/onnx files, so
# `import torch._dynamo` dies in a circular import. Adam.add_param_group is
# wrapped in torch._compile._disable_dynamo, which imports it lazily -- so
# constructing ANY optimizer crashes. _disable_dynamo only needs
# .disable(fn, recursive) to return a callable, and nothing here uses
# torch.compile, so a no-op stub restores the eager behaviour we want.
import types as _types
try:
    import torch._dynamo  # noqa: F401
except Exception:
    for _k in [k for k in list(sys.modules) if k.startswith("torch._dynamo")]:
        del sys.modules[_k]

    def _noop_disable(fn=None, recursive=True, *a, **kw):
        return (lambda f: f) if fn is None else fn

    def _noop(*a, **k):
        return a[0] if (a and callable(a[0])) else None

    class _DynamoStub(_types.ModuleType):
        # torch.optim reaches for several dynamo helpers besides .disable
        # (graph_break, is_compiling, ...) and the set grows between versions,
        # so answer anything with a no-op rather than enumerating them.
        def __getattr__(self, name):
            return _noop

    _stub = _DynamoStub("torch._dynamo")
    _stub.disable = _noop_disable
    _stub.is_compiling = lambda *a, **k: False
    _stub.graph_break = lambda *a, **k: None
    _stub.reset = lambda *a, **k: None
    sys.modules["torch._dynamo"] = _stub
    torch._dynamo = _stub
    print("[shim] torch._dynamo unusable in this env; installed a no-op stub.")
# ---------------------------------------------------------------------------

from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import (ModelParams, PipelineParams, OptimizationParams,
                       get_combined_args)
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


# --------------------------------------------------------------------------
# pseudo-ground-truth store
# --------------------------------------------------------------------------
class PseudoStore:
    """Repaired image + confidence map per view name.

    Images are cached on the CPU as uint8 and confidence as float16, then moved
    to the GPU one at a time. Holding 149 full-res float32 images on the GPU
    would cost several GB for no reason -- the training loop only ever touches
    one camera per iteration.
    """

    def __init__(self, img_dir, conf_dir, names, cache=True):
        self.img_dir, self.conf_dir, self.cache = img_dir, conf_dir, cache
        self.mem = {}
        self.names = []
        for n in names:
            if os.path.exists(os.path.join(img_dir, f"{n}.png")):
                self.names.append(n)
        if cache:
            for n in tqdm(self.names, desc="caching pseudo targets"):
                self.mem[n] = self._read(n)

    def _read(self, name):
        from PIL import Image
        import numpy as np
        a = np.array(Image.open(os.path.join(self.img_dir, f"{name}.png"))
                     .convert("RGB"))
        img = torch.from_numpy(a).permute(2, 0, 1).contiguous()  # uint8 [3,H,W]

        conf = None
        if self.conf_dir:
            npy = os.path.join(self.conf_dir, f"{name}.npy")
            png = os.path.join(self.conf_dir, f"{name}.png")
            if os.path.exists(npy):
                c = np.load(npy).astype("float32")
                conf = torch.from_numpy(c)
            elif os.path.exists(png):
                c = np.array(Image.open(png).convert("L")).astype("float32") / 255.0
                conf = torch.from_numpy(c)
            if conf is not None:
                if conf.dim() == 2:
                    conf = conf.unsqueeze(0)
                conf = conf.half()
        return img, conf

    def get(self, name):
        img, conf = self.mem[name] if self.cache else self._read(name)
        img = img.cuda().float() / 255.0
        if conf is None:
            conf = torch.ones((1,) + img.shape[1:], device="cuda")
        else:
            conf = conf.cuda().float()
            if conf.shape[-2:] != img.shape[-2:]:
                conf = torch.nn.functional.interpolate(
                    conf.unsqueeze(0), size=img.shape[-2:],
                    mode="bilinear", align_corners=False).squeeze(0)
        return img, conf


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def weighted_l1(pred, target, w):
    """Per-pixel confidence-weighted L1. w is [1,H,W], broadcast over channels.

    Normalised by w.sum() rather than by pixel count, so a view where most
    pixels are untrusted does not automatically produce a small loss and get
    quietly ignored.
    """
    num = ((pred - target).abs() * w).sum()
    den = w.sum() * pred.shape[0]
    return num / den.clamp(min=1e-8)


def ssim_term(pred, target):
    if FUSED_SSIM_AVAILABLE:
        return fused_ssim(pred.unsqueeze(0), target.unsqueeze(0))
    return ssim(pred, target)


# --------------------------------------------------------------------------
# evaluation on the untouched holdout
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(gaussians, cams, pipe_cfg, bg, label, lpips_fn=None):
    ps, ss, lp = [], [], []
    for cam in cams:
        img = render(cam, gaussians, pipe_cfg, bg, use_trained_exp=False,
                     separate_sh=SPARSE_ADAM_AVAILABLE)["render"].clamp(0, 1)
        gt = cam.original_image.cuda().clamp(0, 1)
        ps.append(float(psnr(img, gt).mean()))
        ss.append(float(ssim(img, gt)))
        if lpips_fn is not None:
            lp.append(float(lpips_fn(img.unsqueeze(0) * 2 - 1,
                                     gt.unsqueeze(0) * 2 - 1).flatten()[0]))
        del gt, img
    out = {"n": len(ps),
           "psnr": sum(ps) / max(len(ps), 1),
           "ssim": sum(ss) / max(len(ss), 1)}
    if lp:
        out["lpips"] = sum(lp) / len(lp)
    print(f"[{label}] n={out['n']} PSNR={out['psnr']:.3f} "
          f"SSIM={out['ssim']:.4f}"
          + (f" LPIPS={out['lpips']:.4f}" if "lpips" in out else ""))
    return out


# --------------------------------------------------------------------------
# saving in a layout repair_renders.py can load again
# --------------------------------------------------------------------------
def save_model(out_dir, gaussians, src_model_path, iteration, train_test_exp):
    pc = os.path.join(out_dir, "point_cloud", f"iteration_{iteration}")
    os.makedirs(pc, exist_ok=True)
    gaussians.save_ply(os.path.join(pc, "point_cloud.ply"))

    # cfg_args must exist and must point at the NEW model_path, otherwise
    # get_combined_args() will send the next round back to the old checkpoint.
    src_cfg = os.path.join(src_model_path, "cfg_args")
    if os.path.exists(src_cfg):
        ns = eval(open(src_cfg).read())          # noqa: S307 - 3DGS's own format
        ns.model_path = out_dir
        with open(os.path.join(out_dir, "cfg_args"), "w") as f:
            f.write(str(Namespace(**vars(ns))))
    for extra in ("cameras.json", "exposure.json", "input.ply"):
        s = os.path.join(src_model_path, extra)
        d = os.path.join(out_dir, extra)
        if os.path.exists(s) and not os.path.exists(d):
            try:
                import shutil
                shutil.copy2(s, d)
            except Exception:
                pass
    print(f"saved -> {pc}")


# --------------------------------------------------------------------------
def build_parser():
    p = ArgumentParser("Step 4: fine-tune 3DGS on repaired pseudo-views")
    lp_ = ModelParams(p, sentinel=True)
    op_ = OptimizationParams(p)
    pp_ = PipelineParams(p)

    # every custom arg needs a non-None default; get_combined_args drops Nones
    p.add_argument("--load_iteration", type=int, default=-1)
    p.add_argument("--repaired_dir", type=str, default="",
                   help="directory of repaired PNGs (e.g. <D>_conf_schedule/repaired)")
    p.add_argument("--conf_dir", type=str, default="",
                   help="conf_npy/ or confidence/ from stage 1; omit for uniform weights")
    p.add_argument("--out_model", type=str, default="",
                   help="new model directory to write (required)")

    p.add_argument("--ft_iterations", type=int, default=7000)
    p.add_argument("--n_holdout", type=int, default=40)
    p.add_argument("--split_seed", type=int, default=1234)
    p.add_argument("--split_json", type=str, default="",
                   help="reuse an existing split.json instead of regenerating")
    p.add_argument("--real_ratio", type=float, default=0.30,
                   help="fraction of iterations that sample a real training view")
    p.add_argument("--pseudo_weight", type=float, default=1.0,
                   help="global multiplier on the pseudo-view loss")
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="multiply all non-xyz learning rates by this")
    p.add_argument("--densify", action="store_true", default=False,
                   help="allow densification during fine-tuning (off by "
                        "default: densifying onto hallucinated texture bakes "
                        "it into the geometry)")
    p.add_argument("--no_cache", action="store_true", default=False)
    p.add_argument("--eval_every", type=int, default=0,
                   help="also evaluate the holdout every N iterations")
    p.add_argument("--lpips", action="store_true", default=False)
    p.add_argument("--quiet", action="store_true", default=False)
    return p, lp_, op_, pp_


def main():
    parser, lp_, op_, pp_ = build_parser()
    args = get_combined_args(parser)

    if not args.out_model:
        sys.exit("--out_model is required (do not overwrite your step-1 model)")
    if not args.repaired_dir:
        sys.exit("--repaired_dir is required")

    safe_state(args.quiet)
    dataset = lp_.extract(args)
    opt = op_.extract(args)
    pipe_cfg = pp_.extract(args)

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("sparse_adam requested but the accelerated rasterizer is absent.")

    out_model = os.path.expanduser(args.out_model)
    os.makedirs(out_model, exist_ok=True)

    # ---- load the step-1 checkpoint ----
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration,
                  shuffle=False)

    # CRITICAL: load_ply() never sets spatial_lr_scale, it is only set by
    # create_from_pcd(). Left at 0 the xyz learning rate is exactly 0 and the
    # Gaussians can change colour and opacity but can never move -- which looks
    # like "fine-tuning did almost nothing" rather than like a bug.
    gaussians.spatial_lr_scale = scene.cameras_extent

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    # Same family of resume bug as spatial_lr_scale: _exposure and its
    # name->index map are only built by create_from_pcd(); load_ply() does not
    # restore them, so training_setup() dies on Adam([self._exposure]).
    # We render with use_trained_exp=False throughout, so exposure never enters
    # the forward pass -- this only needs to exist for the optimizer to build.
    if getattr(gaussians, "_exposure", None) is None:
        import torch.nn as nn
        eye = torch.eye(3, 4, device="cuda")[None].repeat(len(train_cams), 1, 1)
        gaussians._exposure = nn.Parameter(eye.requires_grad_(True))
        gaussians.exposure_mapping = {c.image_name: i
                                      for i, c in enumerate(train_cams)}
        print(f"[fix] rebuilt identity exposure for {len(train_cams)} cameras")
    if not hasattr(gaussians, "pretrained_exposures"):
        gaussians.pretrained_exposures = None

    gaussians.training_setup(opt)

    if args.lr_scale != 1.0:
        for g in gaussians.optimizer.param_groups:
            if g.get("name") != "xyz":
                g["lr"] = g["lr"] * args.lr_scale
    print(f"loaded {gaussians.get_xyz.shape[0]} gaussians @ iter {scene.loaded_iter}")
    print(f"real train views: {len(train_cams)}   pseudo pool: {len(test_cams)}")

    # ---- deterministic holdout split ----
    if args.split_json and os.path.exists(os.path.expanduser(args.split_json)):
        split = json.load(open(os.path.expanduser(args.split_json)))
        print(f"reusing split from {args.split_json}")
    else:
        names = sorted(c.image_name for c in test_cams)
        rng = random.Random(args.split_seed)
        shuffled = names[:]
        rng.shuffle(shuffled)
        hold = sorted(shuffled[:args.n_holdout])
        fit = sorted(shuffled[args.n_holdout:])
        split = {"seed": args.split_seed, "holdout": hold, "fit": fit}
    hold_set, fit_set = set(split["holdout"]), set(split["fit"])
    with open(os.path.join(out_model, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    holdout_cams = [c for c in test_cams if c.image_name in hold_set]
    fit_cams = [c for c in test_cams if c.image_name in fit_set]
    print(f"holdout (never trained, never repaired): {len(holdout_cams)}")

    # ---- pseudo targets ----
    store = PseudoStore(os.path.expanduser(args.repaired_dir),
                        os.path.expanduser(args.conf_dir) if args.conf_dir else "",
                        [c.image_name for c in fit_cams],
                        cache=not args.no_cache)
    have = set(store.names)
    pseudo_cams = [c for c in fit_cams if c.image_name in have]
    missing = len(fit_cams) - len(pseudo_cams)
    print(f"pseudo-supervised views: {len(pseudo_cams)}"
          + (f"   ({missing} fit views had no repaired PNG and are skipped)"
             if missing else ""))
    if not pseudo_cams:
        sys.exit("No repaired images matched the fit split. Check --repaired_dir.")

    leaked = [n for n in store.names if n in hold_set]
    if leaked:
        print(f"[warn] {len(leaked)} repaired images correspond to HOLDOUT views. "
              f"They are excluded from training, but for a fully clean protocol "
              f"you should not have repaired them at all.")

    lpips_fn = None
    if args.lpips:
        try:
            import lpips as _lpips
            lpips_fn = _lpips.LPIPS(net="alex").cuda().eval()
        except Exception as e:
            print(f"[warn] LPIPS unavailable ({e})")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- before ----
    print("\n=== holdout BEFORE fine-tuning ===")
    before = evaluate(gaussians, holdout_cams, pipe_cfg, background,
                      "before", lpips_fn)

    # ---- fine-tune ----
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    rng = random.Random(args.split_seed + 1)
    ema = 0.0
    bar = tqdm(range(1, args.ft_iterations + 1), desc="finetune")
    history = []

    for i in bar:
        gaussians.update_learning_rate(scene.loaded_iter + i)

        is_real = (not pseudo_cams) or (rng.random() < args.real_ratio)
        if is_real:
            cam = rng.choice(train_cams)
            target = cam.original_image.cuda().clamp(0, 1)
            w = None
            scale = 1.0
        else:
            cam = rng.choice(pseudo_cams)
            target, w = store.get(cam.image_name)
            scale = args.pseudo_weight

        pkg = render(cam, gaussians, pipe_cfg, background,
                     use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = pkg["render"]
        radii, vpt, vis = pkg["radii"], pkg["viewspace_points"], pkg["visibility_filter"]

        if getattr(cam, "alpha_mask", None) is not None:
            am = cam.alpha_mask.cuda()
            image = image * am
            if w is not None:
                w = w * am

        if w is None:
            l1 = l1_loss(image, target)
        else:
            l1 = weighted_l1(image, target, w)
        s = ssim_term(image.clamp(0, 1), target)
        loss = scale * ((1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - s))

        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if i % 10 == 0:
                bar.set_postfix({"loss": f"{ema:.5f}",
                                 "src": "real" if is_real else "pseudo"})

            if args.densify and i < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(
                    gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(vpt, vis)
                if i > opt.densify_from_iter and i % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, 20, radii)

            # use_trained_exp=False means _exposure never enters the forward
            # pass and never gets a gradient, so only step it if one appeared.
            if (getattr(gaussians, "_exposure", None) is not None
                    and gaussians._exposure.grad is not None
                    and hasattr(gaussians, "exposure_optimizer")):
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            if use_sparse_adam:
                gaussians.optimizer.step(radii > 0, radii.shape[0])
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if args.eval_every and i % args.eval_every == 0:
                m = evaluate(gaussians, holdout_cams, pipe_cfg, background,
                             f"iter {i}", None)
                m["iteration"] = i
                history.append(m)

        del target
        if w is not None:
            del w

    bar.close()

    # ---- after ----
    print("\n=== holdout AFTER fine-tuning ===")
    after = evaluate(gaussians, holdout_cams, pipe_cfg, background,
                     "after", lpips_fn)

    final_iter = scene.loaded_iter + args.ft_iterations
    save_model(out_model, gaussians, dataset.model_path, final_iter,
               dataset.train_test_exp)

    report = {"source_model": dataset.model_path,
              "out_model": out_model,
              "repaired_dir": args.repaired_dir,
              "conf_dir": args.conf_dir,
              "ft_iterations": args.ft_iterations,
              "real_ratio": args.real_ratio,
              "pseudo_weight": args.pseudo_weight,
              "densify": bool(args.densify),
              "n_pseudo": len(pseudo_cams),
              "n_holdout": len(holdout_cams),
              "before": before, "after": after, "history": history}
    with open(os.path.join(out_model, "finetune_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- holdout delta ---")
    for k in ("psnr", "ssim", "lpips"):
        if k in before and k in after:
            d = after[k] - before[k]
            good = (d > 0) if k != "lpips" else (d < 0)
            print(f"{k:6s} {before[k]:.4f} -> {after[k]:.4f}   "
                  f"{d:+.4f}  {'OK' if good else 'WORSE'}")
    print(f"\nreport: {os.path.join(out_model, 'finetune_report.json')}")


if __name__ == "__main__":
    main()
