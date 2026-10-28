import argparse
import csv
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless backend for Slurm / Rangpur
import matplotlib.pyplot as plt
import numpy as np
import torch

import modules
from dataset import HipMRIDataset, DEFAULT_ROOT, NUM_CLASSES, PROSTATE_LABEL

BLADDER_LABEL, RECTUM_LABEL = 3, 4
SMOOTH = 1e-5
END_MARGIN = 2
AREA_EDGES = (0, 50, 100, 200, 400, math.inf)
_NAME_RE = re.compile(r"(\d+)_week_(\d+)_slice_(\d+)")

RUN_NAME = "improved"
ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
CKPT = None
OUT_DIR = None
N_FAILURES = 5
N_EXAMPLES = 4
SEED = 42

BLADDER_LABEL, RECTUM_LABEL = 3, 4
SMOOTH = 1e-5
END_MARGIN = 2

AREA_EDGES = (0, 50, 100, 200, 400, math.inf)
_NAME_RE = re.compile(r"(\d+)_week_(\d+)_slice_(\d+)")

def parse_name(path: str, fallback_idx: int):
    """Return (case, week, slice_idx) from a filename; fall back to the order index."""
    m = _NAME_RE.search(os.path.basename(path))
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return "unknown", "0", fallback_idx


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    """Dice and IoU between two boolean masks."""
    inter = np.logical_and(pred, gt).sum()
    p, g = pred.sum(), gt.sum()
    return (2 * inter + SMOOTH) / (p + g + SMOOTH), (inter + SMOOTH) / (p + g - inter + SMOOTH)


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=32):
    """Argmax label map (N, H, W) uint8 for every slice in the dataset."""
    model.eval()
    preds = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.images[start:start + batch_size].to(device)
        preds.append(model(batch).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.concatenate(preds)


def overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """RGB overlay on the MR slice: green = true positive, red = false positive
    (over-contouring), blue = false negative (under-contouring)."""
    lo, hi = image.min(), image.max()
    gray = (image - lo) / (hi - lo + 1e-8)
    rgb = np.stack([gray] * 3, axis=-1)
    colours = {
        "tp": (np.logical_and(pred_mask, gt_mask), (0.0, 1.0, 0.0)),
        "fp": (np.logical_and(pred_mask, ~gt_mask), (1.0, 0.0, 0.0)),
        "fn": (np.logical_and(~pred_mask, gt_mask), (0.0, 0.3, 1.0)),
    }
    for mask, colour in colours.values():
        rgb[mask] = 0.4 * rgb[mask] + 0.6 * np.array(colour)
    return rgb


def _clean(obj):
    """Make nested dict/list JSON-safe (NaN -> None, numpy scalars -> Python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")

def per_slice_table(dataset, preds):
    """One row per test slice with prostate Dice, error pixels and boundary leakage."""
    labels = dataset.labels.numpy()
    rows = []
    for i in range(len(dataset)):
        case, week, sl = parse_name(dataset.image_paths[i], i)
        gt = labels[i] == PROSTATE_LABEL
        pr = preds[i] == PROSTATE_LABEL
        d, iou = dice_iou(pr, gt)
        rows.append({
            "idx": i, "case": case, "week": week, "slice": sl,
            "gt_px": int(gt.sum()), "pred_px": int(pr.sum()),
            "dice": float(d) if gt.any() else float("nan"),   # undefined if no prostate
            "iou": float(iou) if gt.any() else float("nan"),
            "fp_px": int(np.logical_and(pr, ~gt).sum()),
            "fn_px": int(np.logical_and(~pr, gt).sum()),
            # predicted prostate pixels that sit on ground-truth bladder / rectum
            "leak_bladder_px": int(np.logical_and(pr, labels[i] == BLADDER_LABEL).sum()),
            "leak_rectum_px": int(np.logical_and(pr, labels[i] == RECTUM_LABEL).sum()),
        })
    # relative position of each slice within its own case/week volume (0..1)
    groups = {}
    for r in rows:
        groups.setdefault((r["case"], r["week"]), []).append(r["slice"])
    for r in rows:
        s = groups[(r["case"], r["week"])]
        lo, hi = min(s), max(s)
        r["rel_pos"] = (r["slice"] - lo) / (hi - lo) if hi > lo else 0.5
    add_extent_info(rows)
    return rows


def add_extent_info(rows):
    """Add position relative to each scan's ground-truth prostate extent.

    ext_pos      0 = lowest slice index containing prostate, 1 = highest (NaN if no prostate)
    dist_to_end  number of slices to the nearest end of that extent (NaN if no prostate)
    """
    groups = {}
    for r in rows:
        if r["gt_px"] > 0:
            groups.setdefault((r["case"], r["week"]), []).append(r["slice"])
    for r in rows:
        r["ext_pos"] = float("nan")
        r["dist_to_end"] = float("nan")
        key = (r["case"], r["week"])
        if r["gt_px"] > 0 and key in groups:
            lo, hi = min(groups[key]), max(groups[key])
            r["ext_pos"] = (r["slice"] - lo) / (hi - lo) if hi > lo else 0.5
            r["dist_to_end"] = float(min(r["slice"] - lo, hi - r["slice"]))
    return rows


def pooled_dice(rows):
    """Prostate Dice pooled over all pixels of the given rows (TP = gt - FN)."""
    tp = sum(r["gt_px"] - r["fn_px"] for r in rows)
    denom = sum(r["gt_px"] for r in rows) + sum(r["pred_px"] for r in rows)
    return (2 * tp + SMOOTH) / (denom + SMOOTH)


def area_table(rows):
    """Mean prostate Dice grouped by ground-truth prostate area (slices with prostate only)."""
    table = []
    for lo, hi in zip(AREA_EDGES[:-1], AREA_EDGES[1:]):
        vals = [r["dice"] for r in rows if r["gt_px"] > 0 and lo < r["gt_px"] <= hi]
        label = f">{lo}" if math.isinf(hi) else f"{lo + 1}-{int(hi)}"
        table.append({"gt_area_px": label, "n_slices": len(vals),
                      "mean_dice": _mean(vals),
                      "median_dice": float(np.median(vals)) if vals else float("nan")})
    return table


def patient_table(rows):
    """Per-patient summary (the test split contains only a handful of patients)."""
    table = []
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        has = [r for r in sub if r["gt_px"] > 0]
        vals = [r["dice"] for r in has]
        table.append({
            "case": case,
            "n_scans": len({r["week"] for r in sub}),
            "n_slices_with_prostate": len(has),
            "mean_slice_dice": _mean(vals),
            "frac_slices_below_0.75": (sum(v < 0.75 for v in vals) / len(vals)) if vals else float("nan"),
            "pooled_dice": float(pooled_dice(sub)),
        })
    return table


def summary_dict(rows, class_stats):
    """Headline numbers for comparing runs."""
    has = [r for r in rows if r["gt_px"] > 0]
    vals = [r["dice"] for r in has]
    empty = [r for r in rows if r["gt_px"] == 0]
    ends = [r["dice"] for r in has if r["dist_to_end"] <= END_MARGIN]
    interior = [r["dice"] for r in has if r["dist_to_end"] > END_MARGIN]
    total_pred = sum(r["pred_px"] for r in rows)
    return {
        "n_slices": len(rows),
        "n_slices_with_prostate": len(has),
        "n_patients": len({r["case"] for r in rows}),
        "pooled_prostate_dice": float(pooled_dice(rows)),
        "mean_slice_dice": _mean(vals),
        "median_slice_dice": float(np.median(vals)) if vals else float("nan"),
        "frac_slices_below_0.75": (sum(v < 0.75 for v in vals) / len(vals)) if vals else float("nan"),
        "n_slices_dice_below_0.01": int(sum(v < 0.01 for v in vals)),
        "mean_dice_end_slices": _mean(ends),
        "n_end_slices": len(ends),
        "mean_dice_interior_slices": _mean(interior),
        "n_interior_slices": len(interior),
        "end_margin_slices": END_MARGIN,
        "fp_px_total": int(sum(r["fp_px"] for r in rows)),
        "fn_px_total": int(sum(r["fn_px"] for r in rows)),
        "pred_prostate_px_total": int(total_pred),
        "leak_bladder_px_total": int(sum(r["leak_bladder_px"] for r in rows)),
        "leak_rectum_px_total": int(sum(r["leak_rectum_px"] for r in rows)),
        "leak_bladder_frac_of_pred": (sum(r["leak_bladder_px"] for r in rows) / total_pred) if total_pred else float("nan"),
        "empty_slices": len(empty),
        "empty_slices_with_prostate_predicted": int(sum(r["pred_px"] > 0 for r in empty)),
        "classes": class_stats,
    }


def class_statistics(dataset, preds):
    """Pooled per-class Dice / IoU over the whole test split."""
    labels = dataset.labels.numpy()
    stats = {}
    for c in range(NUM_CLASSES):
        d, iou = dice_iou(preds == c, labels == c)
        stats[str(c)] = {"dice": float(d), "iou": float(iou)}
    return stats


def print_overall(class_stats):
    print("\nOverall test results (pooled over all slices):")
    for c in range(NUM_CLASSES):
        tag = " (prostate)" if c == PROSTATE_LABEL else ""
        s = class_stats[str(c)]
        print(f"  class {c}{tag}: Dice {s['dice']:.4f} | IoU {s['iou']:.4f}")


def print_summary(s):
    print(f"\nProstate, per-slice view ({s['n_slices_with_prostate']} slices with prostate, "
          f"{s['n_patients']} patients):")
    print(f"  pooled Dice {s['pooled_prostate_dice']:.3f} | mean per-slice Dice "
          f"{s['mean_slice_dice']:.3f} | median {s['median_slice_dice']:.3f}")
    print(f"  slices with Dice < 0.75: {s['frac_slices_below_0.75']:.1%} | "
          f"near-zero Dice (<0.01): {s['n_slices_dice_below_0.01']}")
    print(f"  mean Dice within {s['end_margin_slices']} slices of either end of the prostate: "
          f"{s['mean_dice_end_slices']:.3f} (n={s['n_end_slices']}) | interior: "
          f"{s['mean_dice_interior_slices']:.3f} (n={s['n_interior_slices']})")
    print(f"  FP pixels {s['fp_px_total']} | FN pixels {s['fn_px_total']}")
    print(f"  predicted prostate on GT bladder: {s['leak_bladder_px_total']} px "
          f"({s['leak_bladder_frac_of_pred']:.2%} of predicted prostate), on GT rectum: "
          f"{s['leak_rectum_px_total']} px")
    print(f"  slices without prostate: {s['empty_slices']}; model predicted prostate on "
          f"{s['empty_slices_with_prostate_predicted']} of them")


def print_tables(area, patients):
    print("\nProstate Dice by ground-truth prostate area (pixels):")
    for a in area:
        print(f"  {a['gt_area_px']:>8s} px: mean Dice {a['mean_dice']:.3f}  (n={a['n_slices']})")
    print("\nPer-patient summary:")
    for p in patients:
        print(f"  case {p['case']}: {p['n_scans']} scan(s), {p['n_slices_with_prostate']} slices with prostate | "
              f"mean slice Dice {p['mean_slice_dice']:.3f} | pooled Dice {p['pooled_dice']:.3f}")


def write_csv(path, table):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

def plot_per_slice(rows, path, title):
    has = [r for r in rows if r["gt_px"] > 0]
    pos = np.array([r["ext_pos"] for r in has])
    dsc = np.array([r["dice"] for r in has])
    bin_idx = np.clip(np.floor(pos * 10).astype(int), 0, 9)

    centres, means, counts = [], [], []
    for b in range(10):
        sel = dsc[bin_idx == b]
        if len(sel):
            centres.append((b + 0.5) / 10)
            means.append(sel.mean())
            counts.append(len(sel))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pos, dsc, s=8, alpha=0.35, label="individual slice")
    ax.plot(centres, means, "o-", color="C1", label="mean per position bin")
    for x, y, n in zip(centres, means, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    ax.axhline(0.75, color="grey", linestyle="--", label="target 0.75")
    ax.set_xlabel("Position within ground-truth prostate extent\n"
                  "(0 = lowest slice index containing prostate, 1 = highest)")
    ax.set_ylabel("Prostate Dice")
    ax.set_title(title)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _row_panels(axes, dataset, preds, idx):
    img = dataset.images[idx, 0].numpy()
    gt = dataset.labels[idx].numpy() == PROSTATE_LABEL
    pr = preds[idx] == PROSTATE_LABEL
    axes[0].imshow(img, cmap="gray")
    axes[1].imshow(gt, cmap="gray")
    axes[2].imshow(pr, cmap="gray")
    axes[3].imshow(overlay(img, pr, gt))
    for a in axes:
        a.axis("off")


def plot_cases(dataset, preds, rows, indices, path, title):
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.8 * n), squeeze=False)
    for k, idx in enumerate(indices):
        _row_panels(axes[k], dataset, preds, idx)
        r = rows[idx]
        head = f"{r['case']} wk{r['week']} slice {r['slice']}"
        if r["gt_px"] > 0:
            axes[k, 0].set_title(f"{head}\nDice {r['dice']:.2f} (GT {r['gt_px']} px)", fontsize=8)
        else:
            axes[k, 0].set_title(f"{head}\n(no prostate in GT)", fontsize=8)
        if k == 0:
            axes[k, 1].set_title("Ground truth\n(prostate)", fontsize=8)
            axes[k, 2].set_title("Prediction\n(prostate)", fontsize=8)
            axes[k, 3].set_title("Green=TP  Red=FP  Blue=FN", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def main():
    run = RUN_NAME
    ckpt = CKPT or f"{run}_best.pth"
    out_dir = OUT_DIR or f"{run}_predict_outputs"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | run: {run} | checkpoint: {ckpt}")

    dataset = HipMRIDataset(ROOT, "test")
    model = modules.UNet(n_channels=1, n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)

    preds = predict_all(model, dataset, device)
    rows = per_slice_table(dataset, preds)
    class_stats = class_statistics(dataset, preds)
    summary = summary_dict(rows, class_stats)
    area = area_table(rows)
    patients = patient_table(rows)

    print_overall(class_stats)
    print_summary(summary)
    print_tables(area, patients)

    # tables
    write_csv(os.path.join(out_dir, "per_slice_results.csv"), rows)
    write_csv(os.path.join(out_dir, "dice_by_area.csv"), area)
    write_csv(os.path.join(out_dir, "per_patient_summary.csv"), patients)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(_clean(summary), f, indent=2)

    plot_per_slice(rows, os.path.join(out_dir, "per_slice_dice.png"),
                   f"Prostate Dice vs. position in prostate extent ({run}, test set)")

    # random examples among slices that contain prostate (fixed seed, not cherry-picked)
    rng = np.random.default_rng(SEED)
    candidates = [r["idx"] for r in rows if r["gt_px"] > 0]
    ex = rng.choice(candidates, size=min(N_EXAMPLES, len(candidates)), replace=False)
    plot_cases(dataset, preds, rows, [int(i) for i in ex],
               os.path.join(out_dir, "example_predictions.png"),
               f"Random test slices containing prostate ({run}, uncurated)")

    # worst failures among slices that contain prostate
    has = [r for r in rows if r["gt_px"] > 0]
    worst = sorted(has, key=lambda r: r["dice"])[:N_FAILURES]
    plot_cases(dataset, preds, rows, [r["idx"] for r in worst],
               os.path.join(out_dir, "failure_cases.png"),
               f"{len(worst)} lowest-Dice test slices ({run})")

    print(f"\n{len(worst)} worst slices (prostate present):")
    for r in worst:
        print(f"  {r['case']} wk{r['week']} slice {r['slice']:>3d} | Dice {r['dice']:.3f} | "
              f"GT {r['gt_px']:>5d}px  pred {r['pred_px']:>5d}px | "
              f"FP {r['fp_px']:>5d}  FN {r['fn_px']:>5d}")
    print(f"\nSaved figures, tables and summary.json to {out_dir}/")


if __name__ == "__main__":
    main()
