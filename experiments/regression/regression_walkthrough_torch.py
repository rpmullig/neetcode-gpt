"""
Regression walkthrough (Torch) — a single-file, decomposed walkthrough with
PyTorch autograd. A NumPy twin with the same structure lives in
regression_walkthrough_numpy.py (there the gradient is hand-derived instead).

This file stitches together, in one place, the pieces that live decomposed
across the other foundations modules — but instead of hand-deriving the
gradient (see linear_regression_training.py::get_derivative), we let
PyTorch's autograd compute it for us.

Structure:

    Model (abstract base)          -> shared training loop + autograd step.
                                       Subclasses supply forward() and loss().
      |
      +-- LinearRegression         -> forward = X @ w,       loss = MSE
      +-- LogisticClassifier       -> forward = sigmoid(Xw), loss = BCE

Component map (what came from where):
    - linear_regression.py            -> forward() / MSE loss()
    - loss.py                         -> binary cross-entropy loss()
    - activations.py                  -> sigmoid in the classifier
    - gradient_descent.py             -> the w = w - lr * grad update rule
    - linear_regression_training.py   -> the train_model loop
      (get_derivative is now replaced by autograd: loss.backward())
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Model(ABC):
    """Abstract linear model. Subclasses define the forward pass and loss.

    Everything shared — the training loop and the autograd-powered
    gradient-descent step — lives here.
    """

    learning_rate: float = 0.01

    # --- Component: the model -------------------------------------------
    @abstractmethod
    def forward(self, X: Tensor, weights: Tensor) -> Tensor:
        """Map inputs X and weights -> predictions."""
        raise NotImplementedError

    # --- Component: the loss --------------------------------------------
    @abstractmethod
    def loss(self, prediction: Tensor, ground_truth: Tensor) -> Tensor:
        """Scalar loss between predictions and ground truth."""
        raise NotImplementedError

    # --- Component: convenience predictor -------------------------------
    def get_model_prediction(self, X: Tensor, weights: Tensor) -> Tensor:
        """Detached predictions (no autograd graph), for inspection/reporting."""
        # torch.no_grad: run without recording gradients (we're just reporting)
        with torch.no_grad():
            return self.forward(X, weights)

    # --- Component: the training loop (shared) --------------------------
    # Replaces the manual get_derivative + update with autograd.
    def train_model(
        self,
        X: Tensor,
        Y: Tensor,
        num_iterations: int,
        initial_weights: Tensor,
        verbose: bool = False,
    ) -> Tensor:
        # torch.as_tensor: make sure X and Y are float64 tensors (no copy if already)
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64)

        # Weights are the leaf tensor we differentiate w.r.t.
        # .clone: make our own copy so we don't overwrite the caller's array
        # .requires_grad_(True): tell autograd to track gradients for these weights
        weights = torch.as_tensor(
            initial_weights, dtype=torch.float64
        ).clone().requires_grad_(True)

        for iteration in range(num_iterations):
            # 1. Forward pass: predict with the current weights.
            prediction = self.forward(X, weights)

            # 2. Measure how wrong we are.
            loss = self.loss(prediction, Y)

            # 3. Backward pass: autograd fills weights.grad = dLoss/dw.
            #    (This is what get_derivative did by hand, now automatic.)
            # loss.backward: autograd walks the math backward and fills weights.grad
            loss.backward()

            if verbose:
                # loss.item: pull the single scalar value out of the tensor
                print(f"iter {iteration:>4}  loss={loss.item():.6f}")

            # 4. Update rule: w = w - lr * grad  (gradient_descent.py).
            # torch.no_grad: change the weights without autograd recording the edit
            with torch.no_grad():
                weights -= self.learning_rate * weights.grad
                weights.grad.zero_()  # reset the stored gradient for the next iteration

        # .detach: drop the autograd tracking; torch.round: round to 5 decimals
        return torch.round(weights.detach(), decimals=5)

    # --- Component: the training loop, FROZEN (static graph) ------------
    # Same math, but we freeze the forward pass into a static graph with
    # torch.jit.trace (a built-in feature — no torch.compile, which needs a C++
    # compiler this machine lacks). The traced graph runs the forward without
    # Python op-dispatch overhead; autograd still flows through it for backward.
    def train_model_traced(
        self,
        X: Tensor,
        Y: Tensor,
        num_iterations: int,
        initial_weights: Tensor,
        verbose: bool = False,
    ) -> Tensor:
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64)
        weights = torch.as_tensor(
            initial_weights, dtype=torch.float64
        ).clone().requires_grad_(True)

        # Trace forward+loss into a frozen graph, ONCE, using example inputs.
        def loss_fn(X, w, Y):
            return self.loss(self.forward(X, w), Y)

        # torch.jit.trace: record the ops into a reusable ScriptFunction
        traced = torch.jit.trace(loss_fn, (X, weights, Y))

        for iteration in range(num_iterations):
            # Run the frozen forward; autograd still builds the backward graph.
            loss = traced(X, weights, Y)
            loss.backward()

            if verbose:
                print(f"iter {iteration:>4}  loss={loss.item():.6f}")

            with torch.no_grad():
                weights -= self.learning_rate * weights.grad
                weights.grad.zero_()

        return torch.round(weights.detach(), decimals=5)


class LinearRegression(Model):
    """Ordinary least squares: predict a continuous value."""

    # forward = X @ weights  (linear_regression.py::get_model_prediction)
    def forward(self, X: Tensor, weights: Tensor) -> Tensor:
        # X @ weights: matrix multiply; torch.squeeze: flatten to a 1D prediction
        return torch.squeeze(X @ weights)

    # loss = mean squared error  (linear_regression.py::get_error)
    def loss(self, prediction: Tensor, ground_truth: Tensor) -> Tensor:
        # torch.mean: average of the squared errors into one scalar
        return torch.mean((prediction - ground_truth) ** 2)


class LogisticClassifier(Model):
    """Binary logistic regression: predict a probability / class label."""

    EPSILON: float = 1e-8

    # forward = sigmoid(X @ weights)  (activations.py::sigmoid)
    def forward(self, X: Tensor, weights: Tensor) -> Tensor:
        # X @ weights: linear score; torch.squeeze: flatten to 1D
        logits = torch.squeeze(X @ weights)
        # torch.sigmoid: squash each score into a probability between 0 and 1
        return torch.sigmoid(logits)

    # loss = binary cross-entropy  (loss.py::binary_cross_entropy)
    def loss(self, prediction: Tensor, ground_truth: Tensor) -> Tensor:
        # torch.clamp: keep probabilities away from exactly 0 or 1 so log() is safe
        p = torch.clamp(prediction, self.EPSILON, 1 - self.EPSILON)
        # torch.log: natural log of each probability; torch.mean: average the terms
        return -torch.mean(
            ground_truth * torch.log(p) + (1 - ground_truth) * torch.log(1 - p)
        )

    # classifier-specific helper: turn probabilities into 0/1 labels.
    def predict_labels(
        self, X: Tensor, weights: Tensor, threshold: float = 0.5
    ) -> Tensor:
        probabilities = self.get_model_prediction(X, weights)
        # compare each probability to the threshold, then .to: cast True/False to 1.0/0.0
        return (probabilities >= threshold).to(torch.float64)


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---------- Demo 1: Linear regression -------------------------------
    #   true model:  y = 2*x0 + 3*x1 + 1(bias)
    n = 200
    x0 = torch.empty(n).uniform_(-5, 5)
    x1 = torch.empty(n).uniform_(-5, 5)
    bias = torch.ones(n)
    X_reg = torch.stack([x0, x1, bias], dim=1).to(torch.float64)
    true_w = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64)
    Y_reg = X_reg @ true_w + torch.randn(n, dtype=torch.float64) * 0.1

    reg = LinearRegression()
    learned = reg.train_model(
        X_reg, Y_reg, num_iterations=1000,
        initial_weights=torch.zeros(3, dtype=torch.float64),
    )
    reg_mse = reg.loss(reg.get_model_prediction(X_reg, learned), Y_reg).item()
    print("=== LinearRegression ===")
    print(f"true weights   : {true_w.tolist()}")
    print(f"learned weights: {learned.tolist()}")
    print(f"final MSE      : {reg_mse:.6f}\n")

    # ---------- Demo 2: Logistic classifier -----------------------------
    #   label = 1 if (x0 + x1 > 0) else 0, learned via a decision boundary.
    X_cls = torch.stack([x0, x1, bias], dim=1).to(torch.float64)
    Y_cls = ((x0 + x1) > 0).to(torch.float64)

    clf = LogisticClassifier()
    clf.learning_rate = 0.1
    clf_w = clf.train_model(
        X_cls, Y_cls, num_iterations=2000,
        initial_weights=torch.zeros(3, dtype=torch.float64),
    )
    preds = clf.predict_labels(X_cls, clf_w)
    accuracy = (preds == Y_cls).to(torch.float64).mean().item()
    print("=== LogisticClassifier ===")
    print(f"learned weights: {clf_w.tolist()}")
    print(f"train accuracy : {accuracy:.4f}")
