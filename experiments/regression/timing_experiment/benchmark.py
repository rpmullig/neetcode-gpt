"""
Timing experiment: eager vs. frozen (static graph) gradient engines. 🏁🐌

We train the SAME linear regression with five configurations and time each as
the dataset grows:

    numpy              -> gradient hand-derived (no autograd machinery)
    torch (eager)      -> PyTorch autograd, graph rebuilt each step
    torch (traced)     -> PyTorch forward FROZEN with torch.jit.trace
    escargrad (eager)  -> our Snail autograd, trail rebuilt each step
    escargrad (frozen) -> our Snail graph FROZEN once and replayed

The point: freezing removes the per-iteration graph-BUILDING overhead. All five
run identical math, so their final MSE must match — only speed should differ.

Run from anywhere:  python benchmark.py
Output: benchmark.png + a printed table.
"""

import os
import sys
import time

import numpy as np
import torch

# Import the three walkthroughs from the parent regression/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regression_walkthrough_numpy import LinearRegression as NumpyLR
from regression_walkthrough_torch import LinearRegression as TorchLR
from regression_walkthrough_escargrad import LinearRegression as EscargradLR

import matplotlib

matplotlib.use("Agg")  # headless: render straight to a file, no window
import matplotlib.pyplot as plt


# ---- knobs -----------------------------------------------------------------
SAMPLE_SIZES = [100, 1_000, 10_000, 100_000, 300_000, 1_000_000]
NUM_FEATURES = 8            # weights to learn (plus a bias column)
NUM_ITERATIONS = 300        # gradient-descent steps per training run
REPEATS = 3                 # take the best of N timings (reduces noise)


def make_regression_data(n_samples: int, n_features: int, seed: int = 0):
    """Fake a linear dataset: y = X @ true_w + small noise."""
    rng = np.random.default_rng(seed)
    # np.random uniform features in [-5, 5], with a trailing bias column of 1s
    X = rng.uniform(-5, 5, size=(n_samples, n_features))
    bias = np.ones((n_samples, 1))
    # np.hstack: glue the bias column onto the features -> (n, n_features + 1)
    X = np.hstack([X, bias])
    true_w = rng.uniform(-3, 3, size=n_features + 1)
    y = X @ true_w + rng.normal(0, 0.1, size=n_samples)
    # np.zeros: every engine starts from the same all-zero weights
    return X, y, np.zeros(n_features + 1)


# Each engine is a function (X, y, w0) -> (learned_weights_as_numpy).
def run_numpy(X, y, w0):
    return NumpyLR().train_model(X, y, NUM_ITERATIONS, w0)


def run_torch_eager(X, y, w0):
    w = TorchLR().train_model(
        torch.as_tensor(X), torch.as_tensor(y), NUM_ITERATIONS, torch.as_tensor(w0)
    )
    return np.asarray(w)


def run_torch_traced(X, y, w0):
    w = TorchLR().train_model_traced(
        torch.as_tensor(X), torch.as_tensor(y), NUM_ITERATIONS, torch.as_tensor(w0)
    )
    return np.asarray(w)


def run_escargrad_eager(X, y, w0):
    return EscargradLR().train_model(X, y, NUM_ITERATIONS, w0)


def run_escargrad_frozen(X, y, w0):
    return EscargradLR().train_model_frozen(X, y, NUM_ITERATIONS, w0)


ENGINES = [
    ("numpy", run_numpy),
    ("torch (eager)", run_torch_eager),
    ("torch (traced)", run_torch_traced),
    ("escargrad (eager)", run_escargrad_eager),
    ("escargrad (frozen)", run_escargrad_frozen),
]
STYLE = {  # (color-agnostic) marker + linestyle so eager/frozen pairs are readable
    "numpy": ("o", "-"),
    "torch (eager)": ("s", "--"),
    "torch (traced)": ("s", "-"),
    "escargrad (eager)": ("^", "--"),
    "escargrad (frozen)": ("^", "-"),
}


def time_engine(fn, X, y, w0):
    """Best-of-REPEATS wall-clock time (seconds) + learned weights.

    A warmup run is burned first so caches / lazy init (torch especially) are
    hot before we start the clock — an even playing field across engines.
    """
    fn(X, y, w0)                       # warmup, discarded
    best = float("inf")
    learned = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        learned = fn(X, y, w0)
        best = min(best, time.perf_counter() - start)
    return best, np.asarray(learned, dtype=np.float64)


def main():
    times = {name: [] for name, _ in ENGINES}
    mses = {name: [] for name, _ in ENGINES}

    header = f"{'n':>8} | " + " | ".join(f"{name:>18}" for name, _ in ENGINES)
    print(header)
    print("-" * len(header))

    for n in SAMPLE_SIZES:
        X, y, w0 = make_regression_data(n, NUM_FEATURES)
        cells = []
        for name, fn in ENGINES:
            secs, w = time_engine(fn, X, y, w0)
            times[name].append(secs)
            # np.mean: final mean-squared error — the correctness check
            mses[name].append(float(np.mean((X @ w - y) ** 2)))
            cells.append(f"{secs * 1000:>13.1f}ms")
        print(f"{n:>8} | " + " | ".join(cells))

    # Correctness: every engine should reach essentially the same MSE.
    print("\nFinal MSE at n =", SAMPLE_SIZES[-1], ":")
    for name, _ in ENGINES:
        print(f"  {name:>18}: {mses[name][-1]:.6f}")

    # ---- plot: speed (left) + quality (right) -----------------------------
    fig, (ax_speed, ax_mse) = plt.subplots(1, 2, figsize=(14, 6))

    for name, _ in ENGINES:
        marker, ls = STYLE[name]
        ax_speed.plot(SAMPLE_SIZES, times[name], marker=marker, linestyle=ls,
                      linewidth=2, label=name)
        ax_mse.plot(SAMPLE_SIZES, mses[name], marker=marker, linestyle=ls,
                    linewidth=2, label=name)

    ax_speed.set_xscale("log"); ax_speed.set_yscale("log")
    ax_speed.set_xlabel("number of samples (log scale)")
    ax_speed.set_ylabel("training time, seconds (log scale)")
    ax_speed.set_title("Training speed: eager vs. frozen (static graph)")
    ax_speed.grid(True, which="both", alpha=0.3)
    ax_speed.legend()

    ax_mse.set_xscale("log")
    ax_mse.set_xlabel("number of samples (log scale)")
    ax_mse.set_ylabel("final MSE")
    ax_mse.set_title("Quality (should all match — correctness check)")
    ax_mse.grid(True, which="both", alpha=0.3)
    ax_mse.legend()

    fig.suptitle(
        f"Precomputed-gradient (static graph) speedup — linear regression\n"
        f"{NUM_FEATURES + 1} weights, {NUM_ITERATIONS} iterations, best of {REPEATS}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved chart -> {out_path}")


if __name__ == "__main__":
    main()
