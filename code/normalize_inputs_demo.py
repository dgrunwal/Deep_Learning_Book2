"""
Normalizing Inputs                                       normalize_inputs_demo.py
==============================================================================
A small, self-contained experiment for Chapter 4 of
"Improving Deep Neural Networks."

It shows, on the SAME data and the SAME model, what input normalization does
for gradient descent:

    1. Train on RAW (unnormalized) features    -> slow, many steps to converge
    2. Train on NORMALIZED features            -> fast, far fewer steps

It also demonstrates the two checks every practitioner should run after
normalizing:

    CHECK 1: each normalized feature has mean ~0 and standard deviation ~1
    CHECK 2: the dev/test data is scaled with the TRAINING statistics,
             never with its own -- the test set must stay a true unknown

Only NumPy is needed. We use a tiny linear-regression-style model so the
effect of feature scale on gradient descent is easy to see without any
deep-learning machinery getting in the way.
"""

import numpy as np

np.random.seed(1)


# -------------------------------------------------------------------
# STEP 1: Make data whose two features live on very different scales
# -------------------------------------------------------------------
# Feature 1 is small (roughly 0-3); feature 2 is large (roughly 0-3000).
# This mismatch is exactly what distorts the cost surface and slows
# gradient descent down. The target y is a fixed linear combination plus
# a little noise, so a good model is definitely learnable.
def make_data(n=200):
    f1 = np.random.rand(n) * 3.0            # small-scale feature
    f2 = np.random.rand(n) * 3000.0         # large-scale feature
    X = np.column_stack([f1, f2])           # shape (n, 2)
    true_w = np.array([2.0, 0.005])         # the relationship to recover
    y = X @ true_w + 1.0 + np.random.randn(n) * 0.5
    return X, y


# -------------------------------------------------------------------
# STEP 2: Normalization -- subtract the mean, divide by the std
# -------------------------------------------------------------------
# We compute the mean and standard deviation from the TRAINING data only,
# and return them so the SAME numbers can be reused on dev/test data.
def fit_normalizer(X_train):
    mean = X_train.mean(axis=0)             # one mean per feature
    std = X_train.std(axis=0)               # one std per feature
    return mean, std


def apply_normalizer(X, mean, std):
    # Subtract the training mean and divide by the training spread.
    return (X - mean) / std


# -------------------------------------------------------------------
# STEP 3: Plain batch gradient descent for a linear model
# -------------------------------------------------------------------
# Model: y_hat = X @ w + b. We minimize mean squared error. The number of
# iterations needed to reach a low cost is our stand-in for "how hard
# gradient descent had to work."
def train(X, y, lr=0.01, iterations=2000, target=0.30):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    steps_to_converge = None
    for i in range(iterations):
        y_hat = X @ w + b
        error = y_hat - y
        cost = np.mean(error ** 2)
        # gradients of mean squared error
        grad_w = (2.0 / n) * (X.T @ error)
        grad_b = (2.0 / n) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b
        # record the first iteration where the cost reaches the target.
        # (target is set just above the noise floor, since the data has
        # random noise that no model can fit away.)
        if steps_to_converge is None and cost < target:
            steps_to_converge = i + 1
    final_cost = np.mean((X @ w + b - y) ** 2)
    return final_cost, steps_to_converge


# -------------------------------------------------------------------
# STEP 4: Run raw vs. normalized and report
# -------------------------------------------------------------------
def main():
    X, y = make_data()

    # Split into train and test the simple way.
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # ---- Fit the normalizer on TRAINING data only ----
    mean, std = fit_normalizer(X_train)
    X_train_norm = apply_normalizer(X_train, mean, std)
    X_test_norm = apply_normalizer(X_test, mean, std)   # reuse TRAIN stats

    # Use the same iteration budget for both runs. The raw features span
    # such a wide range that gradient descent is only stable at a tiny
    # learning rate; the normalized features tolerate a much larger one.
    # That difference IS the lesson.
    ITERS = 2000

    raw_cost, raw_steps = train(X_train, y_train, lr=1e-8, iterations=ITERS)
    norm_cost, norm_steps = train(X_train_norm, y_train, lr=0.1, iterations=ITERS)

    print("Effect of normalization on gradient descent")
    print("-" * 52)
    print(f"{'Inputs':<14}{'Final cost':>14}{'Steps to converge':>22}")
    raw_disp = raw_steps if raw_steps is not None else f">{ITERS}"
    norm_disp = norm_steps if norm_steps is not None else f">{ITERS}"
    print(f"{'raw':<14}{raw_cost:>14.4f}{str(raw_disp):>22}")
    print(f"{'normalized':<14}{norm_cost:>14.4f}{str(norm_disp):>22}")

    # ---------------------------------------------------------------
    # STEP 5: Verify the normalization -- the two checks
    # ---------------------------------------------------------------
    print("\nCHECK 1: each normalized TRAIN feature has mean ~0, std ~1")
    print("-" * 52)
    m = X_train_norm.mean(axis=0)
    s = X_train_norm.std(axis=0)
    for j in range(X_train_norm.shape[1]):
        print(f"  feature {j + 1}:  mean = {m[j]:+.6f}   std = {s[j]:.6f}")

    print("\nCHECK 2: dev/test uses TRAIN stats (its mean is NOT forced to 0)")
    print("-" * 52)
    tm = X_test_norm.mean(axis=0)
    ts = X_test_norm.std(axis=0)
    for j in range(X_test_norm.shape[1]):
        print(f"  feature {j + 1}:  mean = {tm[j]:+.6f}   std = {ts[j]:.6f}")
    print("  (slightly off 0/1 is correct -- test data is scaled by TRAIN"
          " numbers,")
    print("   not its own, so it stays a true unknown.)")


if __name__ == "__main__":
    main()
