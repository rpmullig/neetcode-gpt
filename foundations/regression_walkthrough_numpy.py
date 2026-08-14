"""
Regression walkthrough (NumPy) — a single-file, decomposed walkthrough.

Same structure as regression_walkthrough_torch.py, but with NO PyTorch: here
the gradient is hand-derived (the calculus that autograd would otherwise do
for us). This is the "show your work" version.

Structure:

    Model (abstract base)          -> shared training loop + update step.
                                       Subclasses supply forward(), loss(),
                                       and gradient() (the hand-derived slope).
      |
      +-- LinearRegression         -> forward = X @ w,       loss = MSE
      +-- LogisticClassifier       -> forward = sigmoid(Xw), loss = BCE

Component map (what came from where):
    - linear_regression.py            -> forward() / MSE loss()
    - loss.py                         -> binary cross-entropy loss()
    - activations.py                  -> sigmoid in the classifier
    - gradient_descent.py             -> the w = w - lr * grad update rule
    - linear_regression_training.py   -> get_derivative() + the train_model loop
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Model(ABC):
    """Abstract linear model. Subclasses define forward(), loss(), gradient().

    Everything shared — the training loop and the gradient-descent step —
    lives here.
    """

    learning_rate: float = 0.01

    # --- Component: the model -------------------------------------------
    @abstractmethod
    def forward(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map inputs X and weights -> predictions."""
        raise NotImplementedError

    # --- Component: the loss --------------------------------------------
    @abstractmethod
    def loss(
        self,
        prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> float:
        """Scalar loss between predictions and ground truth."""
        raise NotImplementedError

    # --- Component: the gradient (hand-derived) -------------------------
    @abstractmethod
    def gradient(
        self,
        X: NDArray[np.float64],
        prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Slope of the loss w.r.t. every weight, as one array."""
        raise NotImplementedError

    # --- Component: convenience predictor -------------------------------
    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.forward(X, weights)

    # --- Component: the training loop (shared) --------------------------
    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
        verbose: bool = False,
    ) -> NDArray[np.float64]:
        # np.asarray: make sure X and Y are float64 arrays (no copy if already)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        # np.array: copy the starting weights so we don't overwrite the caller's array
        weights = np.array(initial_weights, dtype=np.float64)

        for iteration in range(num_iterations):
            # 1. Forward pass: predict with the current weights.
            prediction = self.forward(X, weights)

            # 2. Measure how wrong we are.
            if verbose:
                print(f"iter {iteration:>4}  loss={self.loss(prediction, Y):.6f}")

            # 3. Gradient: hand-derived slope of the loss w.r.t. each weight.
            gradients = self.gradient(X, prediction, Y)

            # 4. Update rule: w = w - lr * grad  (gradient_descent.py).
            weights = weights - self.learning_rate * gradients

        # np.round: rounds the final learned weights to 5 decimal places
        return np.round(weights, 5)


class LinearRegression(Model):
    """Ordinary least squares: predict a continuous value."""

    # forward = X @ weights  (linear_regression.py::get_model_prediction)
    def forward(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # np.matmul: matrix multiply X by weights; np.squeeze: flatten to 1D
        return np.squeeze(np.matmul(X, weights))

    # loss = mean squared error  (linear_regression.py::get_error)
    def loss(
        self,
        prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> float:
        # np.mean: average of the squared errors into one number
        return float(np.mean((prediction - ground_truth) ** 2))

    # gradient of MSE:  -2/N * X^T (y - y_hat)   (linear_regression_training.py)
    def gradient(
        self,
        X: NDArray[np.float64],
        prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        N = len(ground_truth)
        # np.dot: X^T times the errors — sums each feature's contribution per weight
        return -2 * np.dot(X.T, (ground_truth - prediction)) / N


class LogisticClassifier(Model):
    """Binary logistic regression: predict a probability / class label."""

    EPSILON: float = 1e-8

    # forward = sigmoid(X @ weights)  (activations.py::sigmoid)
    def forward(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # np.matmul: linear score per row; np.squeeze: flatten to 1D
        logits = np.squeeze(np.matmul(X, weights))
        # np.exp: e^(-logits) for the sigmoid; 1/(1+e^-z) maps scores to (0, 1)
        return 1.0 / (1.0 + np.exp(-logits))

    # loss = binary cross-entropy  (loss.py::binary_cross_entropy)
    def loss(
        self,
        prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> float:
        # np.clip: keep probabilities away from exactly 0 or 1 so log() is safe
        p = np.clip(prediction, self.EPSILON, 1 - self.EPSILON)
        # np.log: natural log of each probability; np.mean: average the terms
        return float(
            -np.mean(ground_truth * np.log(p) + (1 - ground_truth) * np.log(1 - p))
        )

    # gradient of BCE (with sigmoid) is the SAME clean form as MSE's:
    #   -1/N * X^T (y - y_hat)   — one of the reasons this pairing is so common.
    def gradient(
        self,
        X: NDArray[np.float64],
        prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        N = len(ground_truth)
        # np.dot: X^T times the errors — the per-weight slope of the loss
        return -np.dot(X.T, (ground_truth - prediction)) / N

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
        X_reg, Y_reg, num_iterations=1000,
        initial_weights=np.zeros(3),
    )
    reg_mse = reg.loss(reg.get_model_prediction(X_reg, learned), Y_reg)
    print("=== LinearRegression ===")
    print(f"true weights   : {true_w.tolist()}")
    print(f"learned weights: {learned.tolist()}")
    print(f"final MSE      : {reg_mse:.6f}\n")

    # ---------- Demo 2: Logistic classifier -----------------------------
    #   label = 1 if (x0 + x1 > 0) else 0, learned via a decision boundary.
    X_cls = np.stack([x0, x1, bias], axis=1)
    # .astype: turn the True/False labels into 1.0/0.0
    Y_cls = ((x0 + x1) > 0).astype(np.float64)

    clf = LogisticClassifier()
    clf.learning_rate = 0.1
    clf_w = clf.train_model(
        X_cls, Y_cls, num_iterations=2000,
        initial_weights=np.zeros(3),
    )
    preds = clf.predict_labels(X_cls, clf_w)
    # np.mean: fraction of predictions that match the true labels
    accuracy = float(np.mean(preds == Y_cls))
    print("=== LogisticClassifier ===")
    print(f"learned weights: {clf_w.tolist()}")
    print(f"train accuracy : {accuracy:.4f}")
