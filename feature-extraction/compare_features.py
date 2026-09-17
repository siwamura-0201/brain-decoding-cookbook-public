"""Compare two sets of DNN features layer by layer.

Intended to check features extracted by `extract_features.py` against the
features distributed via figshare, which were extracted with the original
Caffe implementation. The two are not guaranteed to be identical.

Example
-------
    uv run compare_features.py \
        ./data/features/ImageNetTest/caffe/VGG19 \
        ./data/features/ImageNetTest/pytorch/VGG19
"""


from typing import List, Optional

import argparse
import os

import hdf5storage
import numpy as np
import scipy.io as sio


def compare_features(
        reference_dir: str,
        target_dir: str,
        layers: Optional[List[str]] = None
) -> None:
    """Report per-layer agreement between two feature directories."""

    if layers is None:
        layers = _common_subdirs(reference_dir, target_dir)
    if len(layers) == 0:
        raise RuntimeError("No layer shared by the two directories.")

    print("Reference: " + reference_dir)
    print("Target:    " + target_dir)
    print("")
    print("%-10s %6s %10s %10s %12s %12s"
          % ("layer", "n", "mean r", "min r", "mean relRMSE", "max |diff|"))
    print("-" * 64)

    for layer in layers:
        labels = _common_labels(
            os.path.join(reference_dir, layer), os.path.join(target_dir, layer))

        correlations, relative_rmses, max_diffs = [], [], []
        for label in labels:
            a = _load_feature(os.path.join(reference_dir, layer, label)).flatten()
            b = _load_feature(os.path.join(target_dir, layer, label)).flatten()
            if a.shape != b.shape:
                raise RuntimeError(
                    "Shape mismatch in %s/%s: %s vs %s"
                    % (layer, label, a.shape, b.shape))

            correlations.append(np.corrcoef(a, b)[0, 1])
            relative_rmses.append(
                np.linalg.norm(a - b) / np.linalg.norm(a)
                if np.linalg.norm(a) > 0 else np.nan)
            max_diffs.append(np.max(np.abs(a - b)))

        print("%-10s %6d %10.6f %10.6f %12.3e %12.3e"
              % (layer, len(labels), np.mean(correlations),
                 np.min(correlations), np.nanmean(relative_rmses),
                 np.max(max_diffs)))


def _common_subdirs(a_dir: str, b_dir: str) -> List[str]:
    a = set([d for d in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, d))])
    b = set([d for d in os.listdir(b_dir) if os.path.isdir(os.path.join(b_dir, d))])
    return sorted(a & b)


def _common_labels(a_dir: str, b_dir: str) -> List[str]:
    a = set([f for f in os.listdir(a_dir) if f.endswith(".mat")])
    b = set([f for f in os.listdir(b_dir) if f.endswith(".mat")])
    return sorted(a & b)


def _load_feature(path: str) -> np.ndarray:
    try:
        return sio.loadmat(path)["feat"]
    except (NotImplementedError, ValueError):
        return hdf5storage.loadmat(path)["feat"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=str, help="reference feature directory")
    parser.add_argument("target", type=str, help="target feature directory")
    parser.add_argument("-l", "--layers", type=str, nargs="+", default=None,
                        help="layers to compare (default: all shared layers)")
    args = parser.parse_args()

    compare_features(args.reference, args.target, args.layers)
