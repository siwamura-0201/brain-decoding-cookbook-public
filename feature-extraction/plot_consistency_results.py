"""Visualize the consistency-check results in `compare_true_feature.ipynb`'s form.

Reads the `perimage_<dataset>_<a>_vs_<b>.npz` files produced by
`check_true_feature_consistency.py` and draws, per stimulus set, the four
checks of the notebook as horizontal swarm+box distributions over layers,
via the same `bdpy.fig.makeplots` call the notebook uses.

The notebook plots one Caffe/PyTorch pair, so each layer row holds a single
distribution. Here every pair among the three stores (caffe / original /
ours) is drawn, passed to `makeplots` as `group`, so each layer row holds
three dodged distributions - the three-way comparison in the notebook's own
figure form.

Two departures from the notebook, both forced by the three-way data:

- the rMSE figures use a log x-axis, because `original vs ours` sits two
  orders of magnitude below the Caffe pairs and vanishes on a linear axis
  (`--linear` restores the notebook's scale); correlation stays linear
- above `--swarm-max` images the swarm is replaced by `box`, whose points are
  the outliers only; they are drawn as the same small round dots the swarm
  uses, not matplotlib's default diamonds (`--plot-type` forces either form)

Example
-------
    python plot_consistency_results.py consistency_check_results
"""


from typing import Dict, List, Tuple

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")

from bdpy.fig import makeplots
import numpy as np
import pandas as pd


# Notebook layer order (deepest first); makeplots draws them top-down.
LAYERS = [
    "fc8", "fc7", "fc6",
    "conv5_4", "conv5_3", "conv5_2", "conv5_1",
    "conv4_4", "conv4_3", "conv4_2", "conv4_1",
    "conv3_4", "conv3_3", "conv3_2", "conv3_1",
    "conv2_2", "conv2_1", "conv1_2", "conv1_1",
]

# Point size shared by the swarm dots and the box plots' outlier markers.
DOT_SIZE = 2

# Pair order in the legend; also the group order makeplots dodges by.
PAIR_ORDER = ["caffe vs original", "caffe vs ours", "original vs ours"]

# The four notebook figures: metric key in the npz -> (file stem, title, x label).
# metric key in the npz -> (file stem, title, axis label, log axis?)
FIGURES = [
    ("raw_image_rmse", "true_feature_image_rmse",
     "True features rMSE", "image rmse", True),
    ("ref_norm_image_rmse", "true_feature_ref_normalized_image_rmse",
     "True features scaled rMSE (first store's SD)", "image rmse", True),
    ("tgt_norm_image_rmse", "true_feature_tgt_normalized_image_rmse",
     "True features scaled rMSE (second store's SD)", "image rmse", True),
    ("image_corr", "true_feature_image_corr",
     "True features correlation", "correlation", False),
]

FILENAME_RE = re.compile(r"^perimage_(.+?)_([a-z]+)_vs_([a-z]+)$")


def _load_per_image(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def _build_frame(
        pairs: Dict[str, Dict[str, np.ndarray]], metric: str
) -> Tuple[pd.DataFrame, List[str]]:
    """One row per (layer, pair), holding the per-image vector - the shape
    `makeplots` expects (the notebook stores whole arrays in a cell)."""

    rows = []
    for pair in PAIR_ORDER:
        if pair not in pairs:
            continue
        for layer in LAYERS:
            key = "%s/%s" % (layer, metric)
            if key not in pairs[pair]:
                continue
            rows.append({"layer": layer, "pair": pair, "value": pairs[pair][key]})

    present = [p for p in PAIR_ORDER if p in pairs]
    return pd.DataFrame(rows, columns=["layer", "pair", "value"]), present


def _round_fliers(fig) -> None:
    """Redraw box-plot outliers as the swarm's round dots.

    `makeplots` takes a `flierprops` argument but never forwards it to its box
    plot, so the outliers keep matplotlib's default diamond and the `box`
    figures do not match the `swarm+box` ones. Fix them on the drawn figure
    instead: seaborn leaves each flier as a marker-only Line2D.
    """
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_marker() in ("d", "D"):
                line.set_marker("o")
                line.set_markersize(DOT_SIZE)
                line.set_markeredgewidth(0)
                line.set_alpha(0.7)


def _plot(
        dataset: str, pairs: Dict[str, Dict[str, np.ndarray]], out_dir: str,
        plot_type: str, log_scale: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for metric, stem, title, x_label, metric_is_log in FIGURES:
        df, present = _build_frame(pairs, metric)
        if df.empty:
            print("No data for %s / %s - skipped" % (dataset, metric))
            continue

        fig = makeplots(
            df,
            x="layer", x_list=LAYERS,
            y="value",
            group="pair", group_list=present,
            subplot=None, subplot_list=None,
            figure=None, figure_list=None,
            plot_type=plot_type,
            horizontal=True,
            x_label="layer", y_label=x_label,
            title="%s (%s)" % (title, dataset),
            style="seaborn-bright",
            plot_size_auto=True, plot_size=(6, 0.3 * len(present)), max_col=2,
            swarm_dot_size=DOT_SIZE,
        )

        _round_fliers(fig)

        # A log x-axis keeps `original vs ours` readable beside the Caffe
        # pairs; the categorical dodge lives on the y-axis, so rescaling x
        # after plotting leaves the grouping untouched.
        if log_scale and metric_is_log:
            for ax in fig.axes:
                if ax.get_xlabel() or ax.lines or ax.collections:
                    ax.set_xscale("log")

        for ext in ("png", "pdf"):
            out_path = os.path.join(out_dir, "%s_%s.%s" % (stem, dataset, ext))
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            print("Saved " + out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz_dir", nargs="?", default="consistency_check_results",
        help="directory holding perimage_<dataset>_<a>_vs_<b>.npz "
             "(default: consistency_check_results)")
    parser.add_argument(
        "--out-dir", default=None,
        help="output directory for figures (default: <npz_dir>/figures)")
    parser.add_argument(
        "--plot-type", choices=["swarm+box", "box", "violin"], default=None,
        help="makeplots plot type (default: swarm+box up to --swarm-max "
             "images, box above it)")
    parser.add_argument(
        "--swarm-max", type=int, default=200,
        help="largest image count still drawn as a swarm (default: 200)")
    parser.add_argument(
        "--linear", action="store_true",
        help="use the notebook's linear axis for the rMSE figures instead of "
             "a log axis")
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(args.npz_dir, "figures")

    paths = sorted(glob.glob(os.path.join(args.npz_dir, "perimage_*.npz")))
    if not paths:
        raise RuntimeError("No perimage_*.npz found in " + args.npz_dir)

    datasets: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = FILENAME_RE.match(stem)
        if not m:
            raise RuntimeError(
                "Cannot parse dataset/pair from filename: %s "
                "(expected perimage_<dataset>_<a>_vs_<b>.npz)" % path)
        dataset, a, b = m.group(1), m.group(2), m.group(3)
        datasets.setdefault(dataset, {})["%s vs %s" % (a, b)] = _load_per_image(path)

    for dataset, pairs in datasets.items():
        n_images = max(len(v) for pair in pairs.values() for v in pair.values())
        plot_type = args.plot_type or (
            "swarm+box" if n_images <= args.swarm_max else "box")
        print("%s: %d images per layer -> plot_type=%s"
              % (dataset, n_images, plot_type))
        _plot(dataset, pairs, os.path.join(out_root, dataset),
              plot_type=plot_type, log_scale=not args.linear)


if __name__ == "__main__":
    main()
