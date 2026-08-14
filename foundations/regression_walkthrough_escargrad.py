"""
Regression walkthrough (Escargrad) — our own tiny autograd engine. 🐌

"Escargrad" = escargot (a snail) + grad:
  - It's SLOW (pure Python + NumPy, no C under the hood like real PyTorch).
  - A snail leaves a TRAIL behind it — and that trail (which operations we
    ran, in what order) is exactly what we record so we can walk it backward
    and compute gradients. That backward walk IS autograd.

This is the third sibling of:
  - regression_walkthrough_torch.py   (gradient via PyTorch autograd)
  - regression_walkthrough_numpy.py   (gradient hand-derived)
Here we REBUILD autograd ourselves so there's no magic left.

How it works, in three ideas:
  1. Every number lives inside a `Snail`. A Snail holds its value (`.data`),
     a slot for its gradient (`.grad`), and a note of which Snails it came
     from and how (its "trail").
  2. Doing math (`.multiply`, `.matmul`, `.sigmoid`, ...) makes a NEW Snail
     and, crucially, records a tiny `_backward` function that knows the LOCAL
     derivative of just that one operation.
  3. Calling `.backward()` on the final loss walks the whole trail in reverse,
     multiplying local derivatives together (the chain rule) until every
     weight's `.grad` is filled in.

We use explicit method calls (foo.multiply(x), not foo * x) on purpose, so the
trail-building is visible and there's no operator-overloading cleverness to
squint at.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Global switch: are we currently recording a trail? (This is our no_grad.)
# ---------------------------------------------------------------------------
_RECORDING = True


def is_recording() -> bool:
    return _RECORDING


class no_grad:
    """Context manager: pause trail-recording inside a `with no_grad():` block.

    Handy when we just want a prediction and don't care about gradients — e.g.
    while reporting accuracy, or while applying the weight update itself.
    """

    def __enter__(self):
        global _RECORDING
        self._previous = _RECORDING
        _RECORDING = False
        return self

    def __exit__(self, *exc):
        global _RECORDING
        _RECORDING = self._previous
        return False


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reverse of NumPy broadcasting: sum `grad` back down to `shape`.

    When a small array got stretched to match a bigger one during the forward
    pass, its gradient must be summed back over the stretched dimensions.
    """
    # np.sum over leading axes: collapse any extra dimensions grad grew
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # sum over any axis that was size-1 (and thus broadcast) in the original
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Snail:
    """A NumPy array that remembers how it was computed, so it can be
    differentiated. The whole engine is this one class."""

    def __init__(self, data, requires_grad: bool = False, _parents=()):
        # np.asarray: store the raw values as a float64 array
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        # np.zeros_like: start every gradient at zero; backward() adds into it
        self.grad: np.ndarray = np.zeros_like(self.data)
        # the "trail": which Snails fed into this one...
        self._parents = set(_parents)
        # ...and the local-derivative rule for the op that made this Snail.
        self._backward = lambda: None

    # -- helpers ---------------------------------------------------------
    def _wrap(self, other) -> "Snail":
        """Turn a raw number/array into a constant Snail if it isn't one."""
        return other if isinstance(other, Snail) else Snail(other)

    def _make(self, data, parents, backward_rule) -> "Snail":
        """Build the result Snail and attach its backward rule — but only
        record the trail if (a) we're recording and (b) a parent needs grad."""
        track = is_recording() and any(p.requires_grad for p in parents)
        out = Snail(data, requires_grad=track, _parents=parents if track else ())
        if track:
            out._backward = backward_rule
        return out

    def zero_grad(self) -> None:
        # np.zeros_like: wipe the stored gradient before the next backward pass
        self.grad = np.zeros_like(self.data)

    # -- operations (each records its own local derivative) --------------
    def add(self, other) -> "Snail":
        other = self._wrap(other)
        out_data = self.data + other.data

        def backward_rule():
            # derivative of a sum passes the gradient straight through to both
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out = self._make(out_data, (self, other), backward_rule)
        return out

    def subtract(self, other) -> "Snail":
        other = self._wrap(other)
        out_data = self.data - other.data

        def backward_rule():
            # d(a-b): +grad flows to a, -grad flows to b
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad, other.data.shape)

        out = self._make(out_data, (self, other), backward_rule)
        return out

    def multiply(self, other) -> "Snail":
        other = self._wrap(other)
        # element-wise multiply of the two arrays
        out_data = self.data * other.data

        def backward_rule():
            # d(a*b)/da = b, and d(a*b)/db = a  (each scaled by the incoming grad)
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out = self._make(out_data, (self, other), backward_rule)
        return out

    def matmul(self, other) -> "Snail":
        other = self._wrap(other)
        # the actual matrix multiply (@) for the forward pass
        out_data = self.data @ other.data

        def backward_rule():
            g = out.grad
            # np.atleast_2d / reshape: treat 1D vectors as skinny 2D matrices so
            # one formula covers (n,m)@(m,) and (n,m)@(m,k) alike.
            A = np.atleast_2d(self.data)              # (n, m)
            B = other.data.reshape(A.shape[1], -1)    # (m, k)
            G = g.reshape(A.shape[0], B.shape[1])     # (n, k)
            if self.requires_grad:
                # dL/dA = G @ B^T
                self.grad += (G @ B.T).reshape(self.data.shape)
            if other.requires_grad:
                # dL/dB = A^T @ G
                other.grad += (A.T @ G).reshape(other.data.shape)

        out = self._make(out_data, (self, other), backward_rule)
        return out

    def pow(self, k: float) -> "Snail":
        # raise every element to the power k
        out_data = self.data ** k

        def backward_rule():
            # power rule: d(x^k)/dx = k * x^(k-1)
            if self.requires_grad:
                self.grad += (k * self.data ** (k - 1)) * out.grad

        out = self._make(out_data, (self,), backward_rule)
        return out

    def neg(self) -> "Snail":
        out_data = -self.data

        def backward_rule():
            # d(-x)/dx = -1
            if self.requires_grad:
                self.grad += -out.grad

        out = self._make(out_data, (self,), backward_rule)
        return out

    def sum(self) -> "Snail":
        # np.sum: collapse the whole array to a single number
        out_data = self.data.sum()

        def backward_rule():
            # every element contributed equally, so each gets the same grad back
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out = self._make(out_data, (self,), backward_rule)
        return out

    def mean(self) -> "Snail":
        # np.mean: average of all elements -> a single number
        out_data = self.data.mean()
        n = self.data.size

        def backward_rule():
            # mean = sum / n, so each element's share of the grad is grad / n
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * (out.grad / n)

        out = self._make(out_data, (self,), backward_rule)
        return out

    def sigmoid(self) -> "Snail":
        # np.exp: 1/(1+e^-x) squashes each value into (0, 1)
        s = 1.0 / (1.0 + np.exp(-self.data))

        def backward_rule():
            # neat identity: d sigmoid/dx = sigmoid * (1 - sigmoid)
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad

        out = self._make(s, (self,), backward_rule)
        return out

    def log(self) -> "Snail":
        # np.log: natural log of each element
        out_data = np.log(self.data)

        def backward_rule():
            # d(log x)/dx = 1/x
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out = self._make(out_data, (self,), backward_rule)
        return out

    def clip(self, lo: float, hi: float) -> "Snail":
        # np.clip: pin values into [lo, hi] (keeps log() away from 0)
        out_data = np.clip(self.data, lo, hi)

        def backward_rule():
            # gradient flows only where we did NOT clip; clipped spots are flat
            if self.requires_grad:
                mask = (self.data >= lo) & (self.data <= hi)
                self.grad += mask * out.grad

        out = self._make(out_data, (self,), backward_rule)
        return out

    # -- the backward pass ----------------------------------------------
    def backward(self) -> None:
        """Walk the whole trail in reverse and fill every `.grad` (chain rule).

        Must be called on a single-number Snail (our scalar loss).
        """
        # 1. Put the parents-before-children order into a list (topological sort).
        ordered, visited = [], set()

        def visit(node: "Snail"):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    visit(parent)
                ordered.append(node)

        visit(self)

        # 2. Seed: d(loss)/d(loss) = 1.
        self.grad = np.ones_like(self.data)

        # 3. Replay operations from last to first, each handing grad to its parents.
        for node in reversed(ordered):
            node._backward()

    def __repr__(self) -> str:
        return f"Snail(data={self.data}, requires_grad={self.requires_grad})"


# ===========================================================================
# Same decomposed model structure as the other two files — now powered by our
# very own Snail autograd. Subclasses only write forward() and loss(); the
# gradient is discovered by walking the trail, exactly like real autograd.
# ===========================================================================
class Model(ABC):
    learning_rate: float = 0.01

    # --- Component: the model -------------------------------------------
    @abstractmethod
    def forward(self, X: Snail, weights: Snail) -> Snail:
        """Map inputs X and weights -> predictions (as Snails)."""
        raise NotImplementedError

    # --- Component: the loss --------------------------------------------
    @abstractmethod
    def loss(self, prediction: Snail, ground_truth: Snail) -> Snail:
        """Scalar loss Snail between predictions and ground truth."""
        raise NotImplementedError

    # --- Component: convenience predictor -------------------------------
    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # no_grad: we only want the numbers, so don't bother recording a trail
        with no_grad():
            return self.forward(Snail(X), Snail(weights)).data

    # --- Component: the training loop (shared) --------------------------
    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
        verbose: bool = False,
    ) -> NDArray[np.float64]:
        # X and Y are constants (no gradient needed); weights are what we tune.
        X_snail = Snail(X)
        Y_snail = Snail(Y)
        weights = Snail(initial_weights, requires_grad=True)

        for iteration in range(num_iterations):
            # 1. Forward pass: build a fresh trail with the current weights.
            prediction = self.forward(X_snail, weights)

            # 2. Loss: how wrong we are, as a single-number Snail.
            loss = self.loss(prediction, Y_snail)

            # 3. Backward pass: our own autograd fills weights.grad.
            weights.zero_grad()          # clear last step's gradient first
            loss.backward()

            if verbose:
                print(f"iter {iteration:>4}  loss={float(loss.data):.6f}")

            # 4. Update rule: w = w - lr * grad  (gradient_descent.py).
            #    Poke .data directly (no need to record this bookkeeping step).
            with no_grad():
                weights.data -= self.learning_rate * weights.grad

        # np.round: rounds the final learned weights to 5 decimal places
        return np.round(weights.data, 5)


class LinearRegression(Model):
    """Ordinary least squares: predict a continuous value."""

    # forward = X @ weights
    def forward(self, X: Snail, weights: Snail) -> Snail:
        return X.matmul(weights)

    # loss = mean squared error = mean( (pred - y)^2 )
    def loss(self, prediction: Snail, ground_truth: Snail) -> Snail:
        error = prediction.subtract(ground_truth)
        return error.pow(2).mean()


class LogisticClassifier(Model):
    """Binary logistic regression: predict a probability / class label."""

    EPSILON: float = 1e-8

    # forward = sigmoid(X @ weights)
    def forward(self, X: Snail, weights: Snail) -> Snail:
        return X.matmul(weights).sigmoid()

    # loss = binary cross-entropy
    #   -mean( y*log(p) + (1-y)*log(1-p) )
    def loss(self, prediction: Snail, ground_truth: Snail) -> Snail:
        p = prediction.clip(self.EPSILON, 1 - self.EPSILON)
        y = ground_truth
        # term1 = y * log(p)
        term1 = y.multiply(p.log())
        # term2 = (1 - y) * log(1 - p)   built with neg().add(1.0) since 1 is a constant
        term2 = y.neg().add(1.0).multiply(p.neg().add(1.0).log())
        # loss = -mean(term1 + term2)
        return term1.add(term2).mean().neg()

    # classifier-specific helper: turn probabilities into 0/1 labels.
    def predict_labels(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64],
        threshold: float = 0.5,
    ) -> NDArray[np.float64]:
        probabilities = self.get_model_prediction(X, weights)
        # compare to threshold, then .astype: cast True/False to 1.0/0.0
        return (probabilities >= threshold).astype(np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---------- Demo 1: Linear regression -------------------------------
    #   true model:  y = 2*x0 + 3*x1 + 1(bias)
    n = 200
    x0 = rng.uniform(-5, 5, size=n)
    x1 = rng.uniform(-5, 5, size=n)
    bias = np.ones(n)
    # np.stack: glue the three columns into an (n, 3) design matrix
    X_reg = np.stack([x0, x1, bias], axis=1)
    true_w = np.array([2.0, 3.0, 1.0])
    Y_reg = X_reg @ true_w + rng.normal(0, 0.1, size=n)

    reg = LinearRegression()
    learned = reg.train_model(
        X_reg, Y_reg, num_iterations=1000, initial_weights=np.zeros(3)
    )
    pred = reg.get_model_prediction(X_reg, learned)
    # np.mean: average squared error, computed with plain numpy for reporting
    reg_mse = float(np.mean((pred - Y_reg) ** 2))
    print("=== LinearRegression (Escargrad) ===")
    print(f"true weights   : {true_w.tolist()}")
    print(f"learned weights: {learned.tolist()}")
    print(f"final MSE      : {reg_mse:.6f}\n")

    # ---------- Demo 2: Logistic classifier -----------------------------
    #   label = 1 if (x0 + x1 > 0) else 0.
    X_cls = np.stack([x0, x1, bias], axis=1)
    # .astype: turn the True/False labels into 1.0/0.0
    Y_cls = ((x0 + x1) > 0).astype(np.float64)

    clf = LogisticClassifier()
    clf.learning_rate = 0.1
    clf_w = clf.train_model(
        X_cls, Y_cls, num_iterations=2000, initial_weights=np.zeros(3)
    )
    preds = clf.predict_labels(X_cls, clf_w)
    # np.mean: fraction of predictions that match the true labels
    accuracy = float(np.mean(preds == Y_cls))
    print("=== LogisticClassifier (Escargrad) ===")
    print(f"learned weights: {clf_w.tolist()}")
    print(f"train accuracy : {accuracy:.4f}")
