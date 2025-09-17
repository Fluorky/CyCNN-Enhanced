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

Adds progress tracking with tqdm (falls back to simple counter if tqdm not available).
"""

import argparse
import csv
import os
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# optional: tqdm for pretty progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fallback to simple prints


def try_load_class_names(n: int, start_dir: Path) -> Optional[List[str]]:
    """
    Look for class_names.txt (one label per line) or class_names.csv (first column)
    in start_dir and its parents up to the confusion_matrices root.
    Return list of length n if found & valid, else None.
    """
    candidates = []
    cur = start_dir
    # search current dir and two parents (dataset/model level)
    for _ in range(3):
        candidates.append(cur / "class_names.txt")
        candidates.append(cur / "class_names.csv")
        cur = cur.parent

    for p in candidates:
        if p.suffix.lower() == ".txt" and p.exists():
            labels = [ln.rstrip("\n\r") for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() != ""]
            if len(labels) == n:
                return labels
        if p.suffix.lower() == ".csv" and p.exists():
            rows = p.read_text(encoding="utf-8").splitlines()
            labels = [row.split(",")[0].strip() for row in rows if row.strip() != ""]
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


def autosize(n: int) -> Tuple[Tuple[int, int], bool, bool]:
    """Return (figsize, annotate_diag, show_text) depending on class count."""
    if n <= 20:
        return (12, 10), True, True
    if n <= 60:
        return (14, 14), True, True  # still annotate diagonal
    if n <= 100:
        return (18, 18), True, False  # diagonal only but no text to keep clean
    return (22, 22), False, False


def plot_matrix(cm: np.ndarray,
                out_path: Path,
                title: str,
                class_names: Optional[List[str]],
                vmax=None,
                fmt="counts"):
    n = cm.shape[0]
    (w, h), annotate_diag, show_text = autosize(n)
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    if class_names and len(class_names) == n:
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticklabels(np.arange(n))
        ax.set_yticklabels(np.arange(n))

    # thin grid
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.2)
    ax.tick_params(axis="both", which="both", length=0)

    # annotate (only diagonal; suppress text for very large N)
    if annotate_diag:
        for i in range(n):
            if not show_text:
                continue
            val = cm[i, i]
            text = f"{val:.2f}" if fmt == "percent" else f"{int(round(val))}"
            ax.text(i, i, text, ha="center", va="center", fontsize=6, color="black")

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
    patterns = [
        str(root / "json_*" / "confusion_matrices" / "**" / "confusion_matrix.npy")
    ]
    results = []
    for pat in patterns:
        results.extend(glob(pat, recursive=True))
    return [Path(p) for p in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/mnt/e/MasterThesisLogsAll",
        help="Root folder (WSL path). Default: /mnt/e/MasterThesisLogsAll",
    )
    parser.add_argument(
        "--save-norm-npy",
        action="store_true",
        help="Also save confusion_matrix_row_norm.npy next to the plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate plots even if output files already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output (only progress + summary).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    npy_paths = find_all_cm_npy(root)
    total = len(npy_paths)
    if total == 0:
        print("No confusion_matrix.npy files found.")
        return

    print(f"Found {total} matrices under {root} (only json_*).")

    processed = 0
    created_png = 0
    created_csv = 0
    created_npy = 0
    errors = []

    iterator = enumerate(npy_paths, start=1)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="Processing confusion matrices", unit="file")

    for idx, npy in iterator:
        try:
            # show current path in tqdm postfix or print
            current_info = f"{npy.parent}"
            if tqdm is not None:
                if hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str(str(npy.parent.name))
            else:
                print(f"[{idx}/{total}] {current_info}")

            cm = np.load(npy)
            n = cm.shape[0]
            labels = try_load_class_names(n, npy.parent) or [str(i) for i in range(n)]

            out_dir = npy.parent
            counts_png = out_dir / "confusion_matrix_counts.png"
            norm_png = out_dir / "confusion_matrix_row_norm.png"
            norm_npy = out_dir / "confusion_matrix_row_norm.npy"
            metrics_csv = out_dir / "metrics.csv"

            # counts plot
            if (not counts_png.exists()) or args.force:
                plot_matrix(
                    cm,
                    counts_png,
                    title=f"Confusion Matrix (counts) — N={n}",
                    class_names=labels,
                    fmt="counts",
                )
                created_png += 1
                if not args.quiet and tqdm is None:
                    print(f"  [✓] {counts_png}")

            # normalized
            cm_norm = row_normalize(cm)
            if (not norm_png.exists()) or args.force:
                plot_matrix(
                    cm_norm,
                    norm_png,
                    title=f"Confusion Matrix (row-normalized) — N={n}",
                    class_names=labels,
                    fmt="percent",
                    vmax=1.0,
                )
                created_png += 1
                if not args.quiet and tqdm is None:
                    print(f"  [✓] {norm_png}")

            if args.save_norm_npy and ((not norm_npy.exists()) or args.force):
                np.save(norm_npy, cm_norm)
                created_npy += 1
                if not args.quiet and tqdm is None:
                    print(f"  [✓] {norm_npy}")

            if (not metrics_csv.exists()) or args.force:
                save_metrics_csv(cm, metrics_csv, class_names=labels)
                created_csv += 1
                if not args.quiet and tqdm is None:
                    print(f"  [✓] {metrics_csv}")

            processed += 1

        except Exception as e:
            errors.append((str(npy), repr(e)))
            if not args.quiet:
                print(f"  [!] Error on {npy}: {e}")

    # summary
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
