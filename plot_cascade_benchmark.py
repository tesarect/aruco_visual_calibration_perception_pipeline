#!/usr/bin/env python3
# Reads findings/cascade_benchmark.csv (written by benchmark_cascade.py,
# possibly across multiple runs/devices) and produces comparison graphs as
# PNG files under findings/. Safe to re-run any time after new benchmark
# data is appended — always regenerates all graphs from the full CSV.
#
# Run inside YOLO-pipeline/venv only.
#
# Usage:
#   python3 plot_cascade_benchmark.py

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

FINDINGS_DIR = Path("findings")
CSV_PATH = FINDINGS_DIR / "cascade_benchmark.csv"


def load_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found — run benchmark_cascade.py at least once first "
            "(e.g. once on GPU here, once on the rosject's CPU)."
        )
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def device_label(rows):
    """Group rows by (device, label) so e.g. 'cuda' (this machine) and
    'cpu' + label 'rosject run 1' plot as distinct series."""
    labels = {}
    for r in rows:
        key = (r["device"], r["label"])
        if key not in labels:
            display = r["device"]
            if r["label"]:
                display += f" ({r['label']})"
            labels[key] = display
    return labels


def plot_time_to_first_success(rows, labels):
    """Per device/label: average time (ms) to find the first successful
    cascade variant, across all images where a marker was eventually found."""
    per_group = defaultdict(list)
    for r in rows:
        if r["is_first_success"] == "True" and r["time_to_first_success_ms"]:
            per_group[(r["device"], r["label"])].append(float(r["time_to_first_success_ms"]))

    if not per_group:
        print("No successful detections found in CSV — skipping time-to-first-success plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    group_keys = sorted(per_group.keys())
    names = [labels[k] for k in group_keys]
    avgs = [sum(v) / len(v) for v in (per_group[k] for k in group_keys)]

    bars = ax.bar(names, avgs, color="#5b9bd5")
    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{avg:.2f} ms", ha="center", va="bottom")

    ax.set_ylabel("avg. time to first successful detection (ms)")
    ax.set_title("Cascade: time to find a working preprocessing variant\n(GPU vs CPU comparison)")
    fig.tight_layout()
    out = FINDINGS_DIR / "time_to_first_success.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_total_time_per_image(rows, labels):
    """Per device/label: average total per-image time (all variants tried
    for that image, whether or not one succeeded)."""
    per_group = defaultdict(list)
    seen_images = defaultdict(set)
    for r in rows:
        if r["image_total_time_ms"]:
            key = (r["device"], r["label"])
            if r["image"] not in seen_images[key]:
                seen_images[key].add(r["image"])
                per_group[key].append(float(r["image_total_time_ms"]))

    if not per_group:
        print("No image-total timing data found — skipping total-time-per-image plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    group_keys = sorted(per_group.keys())
    names = [labels[k] for k in group_keys]
    avgs = [sum(v) / len(v) for v in (per_group[k] for k in group_keys)]

    bars = ax.bar(names, avgs, color="#ed7d31")
    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{avg:.2f} ms", ha="center", va="bottom")

    ax.set_ylabel("avg. total detection time per image (ms)")
    ax.set_title("Overall per-image detection time\n(GPU vs CPU comparison)")
    fig.tight_layout()
    out = FINDINGS_DIR / "total_time_per_image.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_per_variant_breakdown(rows, labels):
    """Per device/label: average time per individual cascade variant (only
    rows where that variant was actually attempted) — shows where time goes
    inside the cascade, not just the aggregate."""
    per_group_variant = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["variant"] and r["variant_time_ms"]:
            key = (r["device"], r["label"])
            per_group_variant[key][r["variant"]].append(float(r["variant_time_ms"]))

    if not per_group_variant:
        print("No per-variant timing data found — skipping per-variant breakdown plot.")
        return

    variant_names = sorted({v for g in per_group_variant.values() for v in g})
    group_keys = sorted(per_group_variant.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(variant_names))
    bar_width = 0.8 / max(len(group_keys), 1)

    for i, key in enumerate(group_keys):
        avgs = [
            (sum(per_group_variant[key][v]) / len(per_group_variant[key][v]))
            if per_group_variant[key][v] else 0
            for v in variant_names
        ]
        offsets = [xi + i * bar_width for xi in x]
        ax.bar(offsets, avgs, width=bar_width, label=labels[key])

    ax.set_xticks([xi + bar_width * (len(group_keys) - 1) / 2 for xi in x])
    ax.set_xticklabels(variant_names, rotation=30, ha="right")
    ax.set_ylabel("avg. time (ms)")
    ax.set_title("Per-variant preprocessing time, by device")
    ax.legend()
    fig.tight_layout()
    out = FINDINGS_DIR / "per_variant_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    rows = load_rows()
    labels = device_label(rows)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}, groups: {list(labels.values())}")

    plot_time_to_first_success(rows, labels)
    plot_total_time_per_image(rows, labels)
    plot_per_variant_breakdown(rows, labels)

    print(f"\nAll graphs saved under {FINDINGS_DIR}/")