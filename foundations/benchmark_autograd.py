"""
Speed + correctness comparison of the three regression walkthroughs. 🏁

We run the SAME two learning tasks with each of our three gradient engines and
time how long training takes as the dataset grows:

    numpy      -> gradient hand-derived         (regression_walkthrough_numpy)
    torch      -> PyTorch autograd (C backend)  (regression_walkthrough_torch)
    escargrad  -> our own Snail autograd 🐌      (regression_walkthrough_escargrad)

Two tasks:
    - Regression       -> quality measured by MSE      (lower is better)
    - Classification   -> quality measured by accuracy (higher is better)

All three engines run identical math, so their QUALITY should match; only the
SPEED should differ. Any quality gap would flag a bug.

Output: a 2x2 PNG chart (speed + quality for each task) plus printed tables.
Run:  python foundations/benchmark_autograd.py
"""

import time

import numpy as np
import torch

# The three implementations of each model, imported under distinct names.
from regression_walkthrough_numpy import (
    LinearRegression as NumpyLR,
    LogisticClassifier as NumpyClf,
)
from regression_walkthrough_torch import (
    LinearRegression as TorchLR,
    LogisticClassifier as TorchClf,
)
from regression_walkthrough_escargrad import (
    LinearRegression as EscargradLR,
    LogisticClassifier as EscargradClf,
)

import matplotlib

matplotlib.use("Agg")  # headless: render straight to a file, no window
import matplotlib.pyplot as plt


# ---- knobs -----------------------------------------------------------------
SAMPLE_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
NUM_FEATURES = 8            # weights to learn (plus a bias column)
NUM_ITERATIONS = 300        # gradient-descent steps per training run
REPEATS = 3                 # take the best of N timings (reduces noise)
MARKERS = {"numpy": "o", "torch": "s", "escargrad": "^"}


def make_regression_data(n_samples: int, n_features: int, seed: int = 0):
    """Fake a linear dataset: y = X @ true_w + small noise."""
    rng = np.random.default_rng(seed)
    # np.random uniform features in [-5, 5], with a trailing bias column of 1s
    X = rng.uniform(-5, 5, size=(n_samples, n_features))
    bias = np.ones((n_samples, 1))
    X = np.hstack([X, bias])                       # (n, n_features + 1)
    true_w = rng.uniform(-3, 3, size=n_features + 1)
    y = X @ true_w + rng.normal(0, 0.1, size=n_samples)
    return X, y, np.zeros(n_features + 1)


def make_classification_data(n_samples: int, n_features: int, seed: int = 1):
    """Fake a linearly-separable-ish dataset: label = 1 if score > 0."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n_samples, n_features))
    bias = np.ones((n_samples, 1))
    X = np.hstack([X, bias])
    true_w = rng.uniform(-3, 3, size=n_features + 1)
    logits = X @ true_w + rng.normal(0, 0.5, size=n_samples)  # a little noise
    # .astype: turn the True/False labels into 1.0/0.0
    y = (logits > 0).astype(np.float64)
    return X, y, np.zeros(n_features + 1)


def train_once(model, X, y, w0, use_torch: bool, learning_rate: float):
    """Run one training, timed. Returns (seconds, learned_weights_as_numpy)."""
    model.learning_rate = learning_rate
    start = time.perf_counter()
    if use_torch:
        weights = model.train_model(
            torch.as_tensor(X), torch.as_tensor(y),
            num_iterations=NUM_ITERATIONS, initial_weights=torch.as_tensor(w0),
        )
    else:
        weights = model.train_model(
            X, y, num_iterations=NUM_ITERATIONS, initial_weights=w0
        )
    elapsed = time.perf_counter() - start
    # np.asarray: pull weights back to numpy (torch returns a Tensor)
    return elapsed, np.asarray(weights, dtype=np.float64)


def regression_quality(X, y, w) -> float:
    # np.mean: final mean-squared error (lower is better)
    return float(np.mean((X @ w - y) ** 2))


def classification_quality(X, y, w) -> float:
    # sigmoid then threshold at 0.5; np.mean of matches = accuracy (higher better)
    probs = 1.0 / (1.0 + np.exp(-(X @ w)))
    preds = (probs >= 0.5).astype(np.float64)
    return float(np.mean(preds == y))


def run_task(task_name, make_data, engines, learning_rate, quality_fn, quality_label):
    """Benchmark all engines across SAMPLE_SIZES for one task."""
    times = {key: [] for key, *_ in engines}
    quality = {key: [] for key, *_ in engines}

    print(f"\n### {task_name} "
          f"(lr={learning_rate}, {NUM_ITERATIONS} iters, best of {REPEATS})")
    header = f"{'n':>8} | " + " | ".join(f"{key:>10} t(ms)/{quality_label:<4}" for key, *_ in engines)
    print(header)
    print("-" * len(header))

    for n in SAMPLE_SIZES:
        X, y, w0 = make_data(n, NUM_FEATURES)
        cells = []
        for key, model, use_torch in engines:
            best = float("inf")
            learned = None
            # Warmup: burn one untimed run so caches / lazy init are hot before
            # we start the clock — equalizes cold-start noise across engines.
            train_once(model, X, y, w0, use_torch, learning_rate)
            for _ in range(REPEATS):
                secs, learned = train_once(model, X, y, w0, use_torch, learning_rate)
                best = min(best, secs)
            times[key].append(best)
            q = quality_fn(X, y, learned)
            quality[key].append(q)
            cells.append(f"{best*1000:>8.1f}/{q:<7.4f}")
        print(f"{n:>8} | " + " | ".join(cells))

    return times, quality


def main():
    reg_engines = [
        ("numpy", NumpyLR(), False),
        ("torch", TorchLR(), True),
        ("escargrad", EscargradLR(), False),
    ]
    clf_engines = [
        ("numpy", NumpyClf(), False),
        ("torch", TorchClf(), True),
        ("escargrad", EscargradClf(), False),
    ]

    reg_times, reg_quality = run_task(
        "REGRESSION", make_regression_data, reg_engines,
        learning_rate=0.01, quality_fn=regression_quality, quality_label="MSE",
    )
    clf_times, clf_quality = run_task(
        "CLASSIFICATION", make_classification_data, clf_engines,
        learning_rate=0.1, quality_fn=classification_quality, quality_label="acc",
    )

    # ---- plot: 2x2 (speed | quality) x (regression | classification) -------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def line_plot(ax, data, title, ylabel, logy):
        for key in ("numpy", "torch", "escargrad"):
            ax.plot(SAMPLE_SIZES, data[key], marker=MARKERS[key], linewidth=2, label=key)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("number of samples (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    line_plot(axes[0, 0], reg_times, "Regression — training speed", "seconds (log)", True)
    line_plot(axes[0, 1], reg_quality, "Regression — quality (should match)", "final MSE", False)
    line_plot(axes[1, 0], clf_times, "Classification — training speed", "seconds (log)", True)
    line_plot(axes[1, 1], clf_quality, "Classification — quality (should match)", "accuracy", False)

    fig.suptitle(
        f"Three gradient engines: numpy (hand-derived) vs torch (autograd/C) vs escargrad (ours)\n"
        f"{NUM_FEATURES + 1} weights, {NUM_ITERATIONS} iterations",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = "foundations/benchmark_autograd.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved chart -> {out_path}")


if __name__ == "__main__":
    main()
