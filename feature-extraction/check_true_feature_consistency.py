"""Consistency check among the true-feature stores (Caffe and PyTorch).

This is the "real data" numerical check that `README.md` / `HANDOFF.md` mark as
outstanding for `extract_features.py`: whether the PyTorch-extracted features
agree with the Caffe features the feature-decoding decoders were trained on.

Three stores are compared for each stimulus set, all pairs against each other:

- ``caffe``   : the distributed Caffe true features the decoders were trained on
- ``original``: the pre-existing PyTorch true-feature store named in the
                notebook's "## ture features paths" cell (extracted by someone
                else's script, not this repository's `extract_features.py`)
- ``ours``    : features produced by *this repository's* `extract_features.py`,
                run on the same stimulus images

Unlike `compare_features.py` (correlation / relative RMSE / max |diff|, no
notion of "how many SDs apart"), this script reproduces the four checks used in
`compare_true_feature.ipynb` for each layer and each pair (a, b):

1. raw error     : a - b, both image-wise and unit-wise RMSE
2. ref-normalized: (a - b) / SD_a(unit), i.e. "how many SDs of a apart"
3. tgt-normalized: (a - b) / SD_b(unit)
4. correlation   : per-image Pearson correlation between the two feature vectors

Per-layer aggregates go to `consistency_<dataset>_<a>_vs_<b>.csv`; the
underlying *per-image* values (what `compare_true_feature.ipynb` plots as
swarm+box distributions) go to `perimage_<dataset>_<a>_vs_<b>.npz`, with one
array per layer and metric (`<layer>/<metric>`).

Example
-------
    python check_true_feature_consistency.py
    python check_true_feature_consistency.py --dataset ImageNetTest
    python check_true_feature_consistency.py --pair original ours
    python check_true_feature_consistency.py \
        --reference /path/to/caffe/VGG19 --target /path/to/pytorch/VGG19
"""


from typing import Dict, List, Sequence, Tuple

import argparse
import csv
import os

from bdpy.dataform import Features
import numpy as np


# Layers common to the Caffe and PyTorch feature stores (pre-ReLU, as saved by
# `extract_features.py`); order follows `compare_true_feature.ipynb`.
LAYERS = [
    "fc8", "fc7", "fc6",
    "conv5_4", "conv5_3", "conv5_2", "conv5_1",
    "conv4_4", "conv4_3", "conv4_2", "conv4_1",
    "conv3_4", "conv3_3", "conv3_2", "conv3_1",
    "conv2_2", "conv2_1", "conv1_2", "conv1_1",
]

# The three true-feature stores per stimulus set. The caffe/original paths are
# the ones named under "## ture features paths" in compare_true_feature.ipynb.
STORES: Dict[str, Dict[str, str]] = {
    "ImageNetTest": {
        "caffe": "/home/share/data/contents_shared/ImageNetTest/derivatives/features/caffe/VGG_ILSVRC_19_layers",
        "original": "/home/share/data/contents_shared/ImageNetTest/derivatives/features/pytorch/VGG_ILSVRC_19_layers/",
        "ours": "/home/siwamura/dev/brain-decoding-cookbook-public/feature-extraction/data/features/ImageNetTest/pytorch/VGG19",
    },
    "ImageNetTraining": {
        "caffe": "/home/share/data/contents_shared/ImageNetTraining/derivatives/features/caffe/VGG_ILSVRC_19_layers",
        "original": "/home/umehara/project/extract_feature_pytorch/data/ImageNetTraining/derivatives/features/pytorch/VGG_ILSVRC_19_layers/",
        "ours": "/var/tmp/siwamura_features/ImageNetTraining/pytorch/VGG19",
    },
}

# Every pair among the three stores ("three-way comparison").
PAIRS: List[Tuple[str, str]] = [
    ("caffe", "original"),
    ("caffe", "ours"),
    ("original", "ours"),
]

FIELDS = [
    "layer", "n", "n_units",
    "raw_image_rmse", "raw_unit_rmse",
    "ref_norm_image_rmse", "ref_norm_unit_rmse",
    "tgt_norm_image_rmse", "tgt_norm_unit_rmse",
    "image_corr_mean", "image_corr_min",
]


def _compare_arrays(
        ref2: np.ndarray, tgt2: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compute the four `compare_true_feature.ipynb` checks for one layer,
    given the two stores' arrays already reshaped to (n_samples, -1) and
    aligned to the same label order.

    Returns the per-layer aggregates and, separately, the per-image vectors
    behind them (the notebook plots those as swarm+box distributions)."""

    diff = ref2 - tgt2
    row: Dict[str, float] = {}
    per_image: Dict[str, np.ndarray] = {}

    # 1. Raw error --------------------------------------------------------
    per_image["raw_image_rmse"] = np.sqrt(np.mean(diff ** 2, axis=1))
    row["raw_image_rmse"] = float(np.mean(per_image["raw_image_rmse"]))
    row["raw_unit_rmse"] = float(np.mean(np.sqrt(np.mean(diff ** 2, axis=0))))

    with np.errstate(invalid="ignore", divide="ignore"):
        # 2. Error in units of the reference store's SD ---------------------
        ref_sd = np.std(ref2, axis=0)
        err_ref = diff / ref_sd
        err_ref = err_ref[:, ~np.isnan(err_ref).any(axis=0)]
        per_image["ref_norm_image_rmse"] = np.sqrt(np.mean(err_ref ** 2, axis=1))
        row["ref_norm_image_rmse"] = float(np.mean(per_image["ref_norm_image_rmse"]))
        row["ref_norm_unit_rmse"] = float(np.mean(np.sqrt(np.mean(err_ref ** 2, axis=0))))

        # 3. Error in units of the target store's SD ------------------------
        tgt_sd = np.std(tgt2, axis=0)
        err_tgt = diff / tgt_sd
        err_tgt = err_tgt[:, ~np.isnan(err_tgt).any(axis=0)]
        per_image["tgt_norm_image_rmse"] = np.sqrt(np.mean(err_tgt ** 2, axis=1))
        row["tgt_norm_image_rmse"] = float(np.mean(per_image["tgt_norm_image_rmse"]))
        row["tgt_norm_unit_rmse"] = float(np.mean(np.sqrt(np.mean(err_tgt ** 2, axis=0))))

    # 4. Per-image correlation --------------------------------------------
    # Vectorized Pearson correlation per row (equivalent to
    # np.corrcoef(ref2[i], tgt2[i])[0, 1] for each i, without the per-image
    # Python loop).
    mean_a = ref2.mean(axis=1)
    mean_b = tgt2.mean(axis=1)
    mean_ab = np.mean(ref2 * tgt2, axis=1)
    var_a = np.mean(ref2 ** 2, axis=1) - mean_a ** 2
    var_b = np.mean(tgt2 ** 2, axis=1) - mean_b ** 2
    corr = (mean_ab - mean_a * mean_b) / np.sqrt(var_a * var_b)
    per_image["image_corr"] = corr
    row["image_corr_mean"] = float(np.mean(corr))
    row["image_corr_min"] = float(np.min(corr))

    return row, per_image


def _align_to(ref_labels, tgt_labels, tgt: np.ndarray) -> np.ndarray:
    if np.array_equal(ref_labels, tgt_labels):
        return tgt
    index = [np.where(np.array(tgt_labels) == x)[0][0] for x in ref_labels]
    return tgt[index]


def check_consistency(
        reference_dir: str,
        target_dir: str,
        layers: Sequence[str] = LAYERS,
) -> List[Dict[str, float]]:
    """Compute per-layer agreement between two true-feature stores, following
    `compare_true_feature.ipynb`."""

    rows, _ = check_consistency_pairs(
        {"reference": reference_dir, "target": target_dir},
        [("reference", "target")], layers=layers)
    return rows[("reference", "target")]


def check_consistency_pairs(
        stores: Dict[str, str],
        pairs: Sequence[Tuple[str, str]],
        layers: Sequence[str] = LAYERS,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, float]]],
           Dict[Tuple[str, str], Dict[str, np.ndarray]]]:
    """Compare several stores pairwise, loading each store's layer only once
    however many pairs it takes part in.

    Returns (per-layer aggregate rows, per-image arrays keyed
    "<layer>/<metric>") for each pair."""

    needed = sorted({name for pair in pairs for name in pair})
    handles = {name: Features(stores[name]) for name in needed}

    anchor = handles[needed[0]]
    for name, store in handles.items():
        if set(anchor.labels) != set(store.labels):
            only_anchor = set(anchor.labels) - set(store.labels)
            only_store = set(store.labels) - set(anchor.labels)
            raise RuntimeError(
                "Stores '%s' and '%s' do not share the same stimulus labels "
                "(%d only in the first, %d only in the second)."
                % (needed[0], name, len(only_anchor), len(only_store)))

    all_rows: Dict[Tuple[str, str], List[Dict[str, float]]] = {p: [] for p in pairs}
    all_per_image: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {p: {} for p in pairs}

    for layer in layers:
        # Load every store this layer is needed from, aligned to one label order.
        arrays = {}
        labels = None
        for name in needed:
            data = handles[name].get(layer=layer)
            if labels is None:
                labels = handles[name].labels
            else:
                data = _align_to(labels, handles[name].labels, data)
            arrays[name] = data.reshape(data.shape[0], -1)

        n, n_units = arrays[needed[0]].shape

        for pair in pairs:
            a, b = pair
            row, per_image = _compare_arrays(arrays[a], arrays[b])
            row["layer"] = layer
            row["n"] = n
            row["n_units"] = n_units
            all_rows[pair].append(row)
            for metric, values in per_image.items():
                all_per_image[pair]["%s/%s" % (layer, metric)] = values.astype(np.float32)

            print(
                "%-10s %-18s n=%-6d units=%-10d raw rmse(img/unit)=%9.4g/%9.4g "
                "SD(%s) rmse(img/unit)=%6.4f/%6.4f SD(%s) rmse(img/unit)=%6.4f/%6.4f "
                "corr(mean/min)=%.4f/%.4f"
                % (
                    layer, "%s_vs_%s" % pair, row["n"], row["n_units"],
                    row["raw_image_rmse"], row["raw_unit_rmse"],
                    a, row["ref_norm_image_rmse"], row["ref_norm_unit_rmse"],
                    b, row["tgt_norm_image_rmse"], row["tgt_norm_unit_rmse"],
                    row["image_corr_mean"], row["image_corr_min"],
                )
            )

        del arrays

    return all_rows, all_per_image


def _write_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved " + out_path)


def _write_npz(per_image: Dict[str, np.ndarray], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    np.savez_compressed(out_path, **per_image)
    print("Saved " + out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=list(STORES) + ["all"], default="all",
        help="which stimulus set to check (default: all)")
    parser.add_argument(
        "--pair", type=str, nargs=2, action="append", metavar=("A", "B"),
        default=None,
        help="check only this pair of stores (repeatable); "
             "default: every pair among caffe/original/ours")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="reference feature directory; overrides --dataset")
    parser.add_argument(
        "--target", type=str, default=None,
        help="target feature directory; overrides --dataset")
    parser.add_argument(
        "-l", "--layers", type=str, nargs="+", default=None,
        help="layers to check (default: the 19 layers shared by the stores)")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="if given, write one CSV per dataset and pair under this directory")
    args = parser.parse_args()

    layers = args.layers if args.layers is not None else LAYERS

    if args.reference is not None or args.target is not None:
        if args.reference is None or args.target is None:
            parser.error("--reference and --target must be given together")
        jobs = {"custom": {"reference": args.reference, "target": args.target}}
        pairs = [("reference", "target")]
    else:
        jobs = STORES if args.dataset == "all" else {args.dataset: STORES[args.dataset]}
        pairs = [tuple(p) for p in args.pair] if args.pair else list(PAIRS)

    for name, stores in jobs.items():
        unknown = {s for pair in pairs for s in pair} - set(stores)
        if unknown:
            parser.error("unknown store(s) for %s: %s" % (name, ", ".join(sorted(unknown))))

        print("=" * 88)
        print("Dataset: " + name)
        for store_name in sorted({s for pair in pairs for s in pair}):
            print("  %-9s %s" % (store_name + ":", stores[store_name]))
        print("Pairs:   " + ", ".join("%s vs %s" % p for p in pairs))
        print("-" * 88)

        all_rows, all_per_image = check_consistency_pairs(stores, pairs, layers=layers)

        if args.out_dir is not None:
            for pair, rows in all_rows.items():
                _write_csv(
                    rows,
                    os.path.join(args.out_dir,
                                 "consistency_%s_%s_vs_%s.csv" % (name, pair[0], pair[1])))
                _write_npz(
                    all_per_image[pair],
                    os.path.join(args.out_dir,
                                 "perimage_%s_%s_vs_%s.npz" % (name, pair[0], pair[1])))

        print("")


if __name__ == "__main__":
    main()
