#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate confusion-matrix plots:
  - counts (optionally logscale)
  - row-normalized (percent), with precise control of low-end coloring:
      * 0.00 shown as pure white
      * > percent_vmin shown with color
      * optional PowerNorm to emphasize tiny values

You can process:
  - a whole tree under --root (default behavior), or
  - specific files via --files, or
  - one directory via --dir (non-recursive).

Outputs next to each confusion_matrix.npy:
  - confusion_matrix_counts.png
  - confusion_matrix_row_norm.png
  - (optional) confusion_matrix_row_norm.npy
  - metrics.csv
"""

import argparse
import csv
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def try_load_class_names(n: int, start_dir: Path) -> Optional[List[str]]:
    candidates = []
    cur = start_dir
    for _ in range(3):
        candidates += [cur / "class_names.txt", cur / "class_names.csv"]
        cur = cur.parent
    for p in candidates:
        if p.suffix.lower() == ".txt" and p.exists():
            labels = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(labels) == n:
                return labels
        if p.suffix.lower() == ".csv" and p.exists():
            rows = p.read_text(encoding="utf-8").splitlines()
            labels = [row.split(",")[0].strip() for row in rows if row.strip()]
            if len(labels) == n:
                return labels
    return None


def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return cm / row_sums


def overall_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else 0.0


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0.0] = 1.0
    return np.diag(cm) / row_sums


def autosize(n: int) -> Tuple[Tuple[int, int], bool]:
    if n <= 20:
        return (12, 10), True
    if n <= 60:
        return (14, 14), True
    if n <= 100:
        return (18, 18), False
    return (22, 22), False


def plot_matrix(cm: np.ndarray,
                out_path: Path,
                title: str,
                class_names: Optional[List[str]],
                *,
                fmt: str,                  # "counts" or "percent"
                logscale: bool = False,    # for counts
                vmin: float = 0.0,
                vmax: Optional[float] = None,
                annot_thresh: Optional[float] = None,
                percent_vmin: float = 1e-6,
                percent_gamma: Optional[float] = None):
    """
    Plot confusion matrix.
    - For fmt="percent": values are expected in [0,1].
      * 0.00 is rendered as pure white (cmap.set_under('white'), vmin=percent_vmin).
      * Any value > percent_vmin gets color.
      * If percent_gamma is set (e.g., 0.4), PowerNorm emphasizes small values.
    """
    n = cm.shape[0]
    (w, h), show_text = autosize(n)
    fig = plt.figure(figsize=(w, h), dpi=160)
    ax = plt.gca()

    if fmt == "counts" and logscale:
        if np.any(cm > 0):
            smallest_pos = float(cm[cm > 0].min())
            vmin_log = max(smallest_pos, 1.0)
            vmax_log = float(cm.max())
        else:
            vmin_log = 1.0
            vmax_log = 1.0
        norm = LogNorm(vmin=vmin_log, vmax=vmax_log)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", norm=norm)
    elif fmt == "percent":
        cmap = plt.cm.Blues.copy()
        cmap.set_under("white")  # strictly below vmin -> pure white (0.00)
        if percent_gamma is not None:
            norm = PowerNorm(gamma=percent_gamma, vmin=percent_vmin, vmax=1.0)
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=percent_vmin, vmax=1.0)
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)

    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    if class_names and len(class_names) == n:
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticklabels(np.arange(n))
        ax.set_yticklabels(np.arange(n))

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.2)
    ax.tick_params(axis="both", which="both", length=0)

    if show_text:
        is_percent = (fmt == "percent")
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if (annot_thresh is not None) and (val < annot_thresh):
                    continue
                text = f"{val:.2f}" if is_percent else f"{int(val)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(cm_counts: np.ndarray, out_csv: Path, class_names=None):
    recalls = per_class_recall(cm_counts)
    acc = overall_accuracy(cm_counts)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["overall_accuracy", f"{acc:.6f}"])
        w.writerow([])
        w.writerow(["class", "recall"])
        for i, r in enumerate(recalls):
            name = class_names[i] if (class_names and i < len(class_names)) else str(i)
            w.writerow([name, f"{r:.6f}"])


def find_all_cm_npy(root: Path):
    pats = [str(root / "json_*" / "confusion_matrices" / "**" / "confusion_matrix.npy")]
    hits = []
    for p in pats:
        hits.extend(glob(p, recursive=True))
    return sorted([Path(p) for p in hits], key=lambda x: str(x).lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/e/MasterThesisLogsAll",
                   help="Root folder to scan recursively (default for batch mode).")
    p.add_argument("--dir", default=None,
                   help="Process a single directory (non-recursive) that contains confusion_matrix.npy.")
    p.add_argument("--files", nargs="+", default=None,
                   help="Explicit confusion_matrix.npy file(s) to process.")
    p.add_argument("--save-norm-npy", action="store_true",
                   help="Also save confusion_matrix_row_norm.npy.")
    p.add_argument("--force", action="store_true",
                   help="Recreate plots even if they exist.")
    p.add_argument("--quiet", action="store_true",
                   help="Less console output.")
    p.add_argument("--logscale-counts", action="store_true",
                   help="Use log scale for counts heatmap.")
    p.add_argument("--annot-thresh", type=float, default=None,
                   help="Only annotate values >= this threshold.")

    # Low-end coloring for percent plots
    p.add_argument("--percent-vmin", type=float, default=1e-6,
                   help="Values < this are pure white on percent plot (default 1e-6).")
    p.add_argument("--percent-gamma", type=float, default=None,
                   help="Gamma for PowerNorm on percent plot (e.g., 0.4 to boost small values).")

    args = p.parse_args()

    # Determine worklist
    work: List[Path] = []
    if args.files:
        work = [Path(f) for f in args.files]
    elif args.dir:
        cand = Path(args.dir) / "confusion_matrix.npy"
        if cand.exists():
            work = [cand]
    else:
        work = find_all_cm_npy(Path(args.root))

    if not work:
        print("No confusion_matrix.npy files to process.")
        return

    total = len(work)
    if not args.quiet:
        print(f"Processing {total} file(s).")

    pb = tqdm(total=total, desc="Processing", unit="file") if (tqdm and not args.quiet) else None

    made_png = made_csv = made_npy = 0
    for idx, npy_path in enumerate(work, 1):
        try:
            cm = np.load(npy_path)
            n = cm.shape[0]
            out_dir = npy_path.parent
            labels = try_load_class_names(n, out_dir) or [str(i) for i in range(n)]

            counts_png = out_dir / "confusion_matrix_counts.png"
            norm_png   = out_dir / "confusion_matrix_row_norm.png"
            norm_npy   = out_dir / "confusion_matrix_row_norm.npy"
            metrics_csv= out_dir / "metrics.csv"

            # COUNTS
            if args.force or (not counts_png.exists()):
                plot_matrix(cm, counts_png,
                            title=f"Confusion Matrix (counts) — N={n}",
                            class_names=labels,
                            fmt="counts",
                            logscale=args.logscale_counts,
                            vmin=0.0, vmax=None,
                            annot_thresh=args.annot_thresh)
                made_png += 1

            # PERCENT (row-normalized)
            cm_norm = row_normalize(cm)
            if args.force or (not norm_png.exists()):
                plot_matrix(cm_norm, norm_png,
                            title=f"Confusion Matrix (row-normalized) — N={n}",
                            class_names=labels,
                            fmt="percent",
                            percent_vmin=args.percent_vmin,
                            percent_gamma=args.percent_gamma,
                            annot_thresh=args.annot_thresh)
                made_png += 1

            if args.save_norm_npy and (args.force or (not norm_npy.exists())):
                np.save(norm_npy, cm_norm)
                made_npy += 1

            if args.force or (not metrics_csv.exists()):
                save_metrics_csv(cm, metrics_csv, class_names=labels)
                made_csv += 1

        except Exception as e:
            if not args.quiet:
                print(f"[{idx}/{total}] ERROR {npy_path}: {e}")

        finally:
            if pb is not None:
                pb.update(1)

    if pb is not None:
        pb.close()

    if not args.quiet:
        print(f"\nDone. PNGs: {made_png}  CSVs: {made_csv}  NPYs: {made_npy}")
        print("Tip: for sharp low-end contrast use e.g. --percent-vmin 0.001 --percent-gamma 0.4")

if __name__ == "__main__":
    main()
