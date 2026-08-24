"""Score step-3 outputs against ground truth and print an ablation table.

Two modes, because the pipeline may be split across two environments:

  SCENE MODE (env A -- has the 3DGS stack)
      python eval_repair.py -m <model_path> --dirs <d1> <d2> ... [--lpips]
      Ground truth comes from cam.original_image, so resolution always matches.

  STANDALONE MODE (env B -- just torch + pillow, no rasterizer)
      python eval_repair.py --gt_dir <D>/gt --dirs <d1> <d2> ... [--lpips]
      Ground truth comes from PNGs dumped by repair_renders.py --save_gt.
      Nothing from the gaussian-splatting repo is imported at all.

Each scored directory may contain renders/ and/or repaired/. Directories
produced by repair_images.py only have repaired/, so the baseline for the
delta table is taken from the first renders/ found (override with --baseline).

SSIM is implemented here with the same 11x11 / sigma=1.5 Gaussian window the
3DGS repo uses, so numbers are comparable across both modes.
"""
import os
import sys
import csv
import json
import math
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

STANDALONE = "--gt_dir" in sys.argv


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _gauss_window(ws, sigma, channel, device, dtype):
    g = torch.tensor([math.exp(-((x - ws // 2) ** 2) / (2.0 * sigma ** 2))
                      for x in range(ws)], device=device, dtype=dtype)
    g = (g / g.sum()).unsqueeze(1)
    w2 = g.mm(g.t()).unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, ws, ws).contiguous()


def ssim(a, b, ws=11, sigma=1.5):
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


def load_png(path, device="cuda"):
    from PIL import Image
    import numpy as np
    a = np.array(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).to(device).clamp(0, 1)


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def names_in(d, sub, gt_by_name):
    """Stems present in <d>/<sub> that also have ground truth."""
    path = os.path.join(d, sub)
    if not os.path.isdir(path):
        return None
    out = {os.path.splitext(f)[0] for f in os.listdir(path)
           if f.lower().endswith(".png")}
    out &= set(gt_by_name)
    return out or None


def score_dir(d, sub, gt_by_name, lpips_fn, device, allowed=None):
    path = os.path.join(d, sub)
    if not os.path.isdir(path):
        return None, []
    rows = []
    for fn in sorted(os.listdir(path)):
        if not fn.lower().endswith(".png"):
            continue
        name = os.path.splitext(fn)[0]
        if allowed is not None and name not in allowed:
            continue
        gt = gt_by_name.get(name)
        if gt is None:
            continue
        if callable(gt):          # lazy PNG loader
            gt = gt()
        pred = load_png(os.path.join(path, fn), device)
        if pred.shape != gt.shape:
            print(f"  [skip] {sub}/{fn}: {tuple(pred.shape)} != gt {tuple(gt.shape)}")
            continue
        row = {"name": name, "psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
        if lpips_fn is not None:
            with torch.no_grad():
                row["lpips"] = float(
                    lpips_fn(pred.unsqueeze(0) * 2 - 1,
                             gt.unsqueeze(0) * 2 - 1).flatten()[0])
        rows.append(row)
        del pred
    if not rows:
        return None, []
    agg = {k: mean_or_none([r.get(k) for r in rows])
           for k in ("psnr", "ssim", "lpips")}
    agg["n"] = len(rows)
    return agg, rows


# --------------------------------------------------------------------------
# ground truth sources
# --------------------------------------------------------------------------
def gt_from_pngs(gt_dir, device):
    gt_dir = os.path.expanduser(gt_dir)
    if not os.path.isdir(gt_dir):
        sys.exit(f"--gt_dir {gt_dir} is not a directory.")
    out = {}
    for fn in sorted(os.listdir(gt_dir)):
        if fn.lower().endswith(".png"):
            p = os.path.join(gt_dir, fn)
            # lazy: 189 full-res images would not all fit comfortably in VRAM
            out[os.path.splitext(fn)[0]] = (lambda p=p: load_png(p, device))
    if not out:
        sys.exit(f"No PNGs in {gt_dir}")
    return out


def gt_from_scene(args, device):
    from scene import Scene, GaussianModel
    from utils.general_utils import safe_state
    safe_state(args.quiet)
    dataset = args._lp.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.load_iteration,
                      shuffle=False)
    cams = scene.getTestCameras() if args.view_set == "test" \
        else scene.getTrainCameras()
    if len(cams) == 0:
        sys.exit(f"No {args.view_set} cameras in this scene.")
    out = {}
    for c in cams:
        g = getattr(c, "original_image", None)
        if g is not None:
            out[c.image_name] = g.to(device).clamp(0, 1)
    if not out:
        sys.exit("Cameras carry no original_image; cannot evaluate.")
    return out


# --------------------------------------------------------------------------
def main():
    ap = ArgumentParser("Evaluate step-3 repair outputs against ground truth")
    lp_ = None
    if not STANDALONE:
        from arguments import ModelParams
        lp_ = ModelParams(ap, sentinel=True)

    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--gt_dir", default="",
                    help="score against dumped GT PNGs; skips the Scene "
                         "entirely so this runs without the 3DGS stack")
    ap.add_argument("--baseline", default="",
                    help="dir whose renders/ is the comparison baseline; "
                         "defaults to the first renders/ found")
    ap.add_argument("--view_set", default="test", choices=["test", "train"])
    ap.add_argument("--load_iteration", type=int, default=-1)
    ap.add_argument("--lpips", action="store_true", default=False)
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--no_intersect", action="store_true", default=False,
                    help="score every directory over whatever views it has, "
                         "instead of restricting all of them to the views "
                         "they share. Almost always wrong for ablations.")
    ap.add_argument("--csv", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quiet", action="store_true", default=False)

    if STANDALONE:
        args = ap.parse_args(sys.argv[1:])
    else:
        from arguments import get_combined_args
        args = get_combined_args(ap)
        args._lp = lp_

    device = args.device
    if args.gt_dir:
        gt_by_name = gt_from_pngs(args.gt_dir, device)
        print(f"ground truth: {len(gt_by_name)} PNGs from {args.gt_dir}\n")
    else:
        gt_by_name = gt_from_scene(args, device)
        print(f"ground truth: {len(gt_by_name)} {args.view_set} views\n")

    lpips_fn = None
    if args.lpips:
        try:
            import lpips as _lpips
            lpips_fn = _lpips.LPIPS(net=args.lpips_net).to(device).eval()
        except Exception as e:
            print(f"[warn] LPIPS unavailable ({e}); continuing without it.")
            if "onnx" in str(e).lower() or "DiagnosticOptions" in str(e):
                print("       ^ lpips imports torchvision, and your torchvision "
                      "does not match the installed torch. This is an env "
                      "problem, not a metric problem.\n")
            else:
                print()

    # ---- pass 1: work out which views EVERY scored set has in common ----
    # Without this, a 189-view baseline gets compared against a 4-view
    # --limit run and the deltas are meaningless: the subsets differ in
    # difficulty, so the numbers are not measuring the repair at all.
    targets = []
    for d in args.dirs:
        d = os.path.expanduser(d)
        for sub in ("renders", "repaired"):
            ns = names_in(d, sub, gt_by_name)
            if ns:
                targets.append((d, sub, ns))
    if not targets:
        sys.exit("Nothing scorable. Check --dirs.")

    allowed = None
    if not args.no_intersect:
        allowed = set.intersection(*[ns for _, _, ns in targets])
        if not allowed:
            sys.exit("The scored directories share no common view names. "
                     "Re-run with --no_intersect if that is intentional.")
        sizes = {len(ns) for _, _, ns in targets}
        if len(sizes) > 1:
            print(f"[info] view counts differ across directories "
                  f"({sorted(sizes)}); restricting all of them to the "
                  f"{len(allowed)} views they share so the comparison is "
                  f"like-for-like. Use --no_intersect to disable.\n")

    results, per_view = [], []
    for d in args.dirs:
        d = os.path.expanduser(d)
        label = os.path.basename(os.path.normpath(d))
        print(f"scoring {d}")
        found = False
        for sub in ("renders", "repaired"):
            agg, rows = score_dir(d, sub, gt_by_name, lpips_fn, device,
                                  allowed=allowed)
            if agg is None:
                continue
            found = True
            agg.update({"dir": label, "set": sub})
            results.append(agg)
            for r in rows:
                r.update({"dir": label, "set": sub})
                per_view.append(r)
        if not found:
            print("  [skip] no scorable PNGs (renders/ or repaired/)")

    if not results:
        sys.exit("Nothing scored. Check --dirs.")

    w = max(len(r["dir"]) for r in results) + 2
    header = f"{'dir':<{w}}{'set':<11}{'n':>5}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['dir']:<{w}}{r['set']:<11}{r['n']:>5}"
              f"{fmt(r['psnr'], 3):>10}{fmt(r['ssim'], 4):>10}"
              f"{fmt(r.get('lpips'), 4):>10}")

    # ---- deltas vs a single baseline (the un-repaired render) ----
    base = None
    if args.baseline:
        want = os.path.basename(os.path.normpath(os.path.expanduser(args.baseline)))
        base = next((r for r in results
                     if r["dir"] == want and r["set"] == "renders"), None)
    if base is None:
        base = next((r for r in results if r["set"] == "renders"), None)

    if base is None:
        print("\n(no renders/ found among --dirs, so no baseline to compare to)")
    else:
        print(f"\ndelta vs baseline [{base['dir']}/renders]:")
        for r in results:
            if r is base:
                continue
            parts = []
            for k, better in (("psnr", "up"), ("ssim", "up"), ("lpips", "down")):
                a, b = base.get(k), r.get(k)
                if a is None or b is None:
                    continue
                delta = b - a
                if abs(delta) < 1e-6:
                    tag = "SAME"
                else:
                    good = (delta > 0) if better == "up" else (delta < 0)
                    tag = "OK" if good else "WORSE"
                parts.append(f"{k} {delta:+.4f} {tag}")
            print(f"  {r['dir']}/{r['set']:<10} {'   '.join(parts)}")

    if args.csv:
        out = os.path.expanduser(args.csv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        cols = ["dir", "set", "name", "psnr", "ssim", "lpips"]
        with open(out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wtr.writeheader()
            wtr.writerows(per_view)
        with open(os.path.splitext(out)[0] + "_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out} and {os.path.splitext(out)[0]}_summary.json")


if __name__ == "__main__":
    main()