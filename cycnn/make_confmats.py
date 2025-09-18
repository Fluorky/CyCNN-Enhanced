#!/usr/bin/env python3
"""
Walk through MasterThesisLogsAll/json_*/confusion_matrices/**/confusion_matrix.npy
and produce:
  - confusion_matrix_counts.png
  - confusion_matrix_row_norm.png
  - (optional) confusion_matrix_row_norm.npy
  - metrics.csv (overall accuracy + per-class recall)

Class count is detected from the matrix shape, so it works for MNIST(10), GTSRB(43), LEGO(50), etc.
Optionally reads class names from class_names.txt or class_names.csv if present.

Adds:
  - progress + detailed logs
  - annotate ALL cells by default (even zeros)
  - optional log-scale for counts heatmap
"""

import argparse
import csv
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# headless backend (safe for WSL/servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
                fmt: str,                 # "counts" or "percent"
                logscale: bool = False,   # for counts
                vmin: float = 0.0,
                vmax: Optional[float] = None,
                annot_thresh: Optional[float] = None):
    n = cm.shape[0]
    (w, h), show_text = autosize(n)
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = plt.gca()

    norm = None
    if fmt == "counts" and logscale:
        smallest_pos = cm[cm > 0].min() if np.any(cm > 0) else 1.0
        norm = LogNorm(vmin=max(smallest_pos, 1.0), vmax=cm.max())
        vmax = None

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax, norm=norm)
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

    # annotate all cells (even zeros)
    if show_text:
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if annot_thresh is not None and val < annot_thresh:
                    continue
                text = f"{val:.2f}" if fmt == "percent" else f"{int(val)}"
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
        writer = csv.writer(f)
        writer.writerow(["overall_accuracy", f"{acc:.6f}"])
        writer.writerow([])
        writer.writerow(["class", "recall"])
        for i, r in enumerate(recalls):
            name = class_names[i] if (class_names and i < len(class_names)) else str(i)
            writer.writerow([name, f"{r:.6f}"])


def find_all_cm_npy(root: Path):
    pats = [str(root / "json_*" / "confusion_matrices" / "**" / "confusion_matrix.npy")]
    hits = []
    for p in pats:
        hits.extend(glob(p, recursive=True))
    return sorted([Path(p) for p in hits], key=lambda x: str(x).lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/e/MasterThesisLogsAll",
                        help="Root folder (WSL path). Default: /mnt/e/MasterThesisLogsAll")
    parser.add_argument("--save-norm-npy", action="store_true",
                        help="Also save confusion_matrix_row_norm.npy next to the plots.")
    parser.add_argument("--force", action="store_true",
                        help="Recreate plots even if output files already exist.")
    parser.add_argument("--quiet", action="store_true",
                        help="Less console output (only summary).")
    parser.add_argument("--logscale-counts", action="store_true",
                        help="Use log scale for counts heatmap.")
    parser.add_argument("--annot-thresh", type=float, default=None,
                        help="Only annotate values >= this threshold.")
    args = parser.parse_args()

    root = Path(args.root)
    paths = find_all_cm_npy(root)
    total = len(paths)
    if total == 0:
        print("No confusion_matrix.npy files found.")
        return

    print(f"Found {total} matrices under {root} (only json_*).")

    processed = created_png = created_csv = created_npy = 0
    errors = []
    pb = tqdm(total=total, desc="Processing", unit="file") if (tqdm and not args.quiet) else None

    def log(msg: str):
        if args.quiet:
            return
        if pb is not None:
            tqdm.write(msg)
        else:
            print(msg)

    for idx, npy_path in enumerate(paths, 1):
        try:
            log(f"[{idx}/{total}] {npy_path.parent}")
            cm = np.load(npy_path)
            n = cm.shape[0]
            labels = try_load_class_names(n, npy_path.parent) or [str(i) for i in range(n)]

            out_dir = npy_path.parent
            counts_png = out_dir / "confusion_matrix_counts.png"
            norm_png = out_dir / "confusion_matrix_row_norm.png"
            norm_npy = out_dir / "confusion_matrix_row_norm.npy"
            metrics_csv = out_dir / "metrics.csv"

            if (not counts_png.exists()) or args.force:
                plot_matrix(cm, counts_png,
                            title=f"Confusion Matrix (counts) — N={n}",
                            class_names=labels,
                            fmt="counts",
                            logscale=args.logscale_counts,
                            vmin=0.0, vmax=None,
                            annot_thresh=args.annot_thresh)
                created_png += 1
                log(f"   [✓] {counts_png}")

            cm_norm = row_normalize(cm)
            if (not norm_png.exists()) or args.force:
                plot_matrix(cm_norm, norm_png,
                            title=f"Confusion Matrix (row-normalized) — N={n}",
                            class_names=labels,
                            fmt="percent",
                            logscale=False,
                            vmin=0.0, vmax=1.0,
                            annot_thresh=args.annot_thresh)
                created_png += 1
                log(f"   [✓] {norm_png}")

            if args.save_norm_npy and ((not norm_npy.exists()) or args.force):
                np.save(norm_npy, cm_norm)
                created_npy += 1
                log(f"   [✓] {norm_npy}")

            if (not metrics_csv.exists()) or args.force:
                save_metrics_csv(cm, metrics_csv, class_names=labels)
                created_csv += 1
                log(f"   [✓] {metrics_csv}")

            processed += 1

        except Exception as e:
            errors.append((str(npy_path), repr(e)))
            log(f"   [!] Error on {npy_path}: {e}")

        finally:
            if pb is not None:
                pb.update(1)

    if pb is not None:
        pb.close()

    print("\n=== Summary ===")
    print(f"Processed: {processed}/{total} files")
    print(f"Created/updated PNGs: {created_png}")
    print(f"Created/updated CSVs: {created_csv}")
    if args.save_norm_npy:
        print(f"Created/updated row-norm NPys: {created_npy}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for p, msg in errors[:10]:
            print(f" - {p}: {msg}")
        if len(errors) > 10:
            print(f" ... and {len(errors)-10} more.")


if __name__ == "__main__":
    main()
