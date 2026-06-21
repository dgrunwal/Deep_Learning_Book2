"""
Regularization: L2 and L1                              regularization_demo.py
==============================================================================
A small, self-contained experiment for Chapter 5 of
"Improving Deep Neural Networks."

It trains the SAME three-layer network on the SAME noisy dataset three times,
changing only the regularization:

    1. none   -- the baseline; free to overfit
    2. L2      -- penalize the sum of SQUARED weights  (weight decay)
    3. L1      -- penalize the sum of ABSOLUTE weights (drives weights to 0)

For each run it prints the training and test accuracy, so you can watch the
train/test gap -- the fingerprint of overfitting -- shrink as regularization
is added. It then runs a simple CHECK that confirms regularization did what it
promises: it made the weights smaller.

Only NumPy is needed; the network is written out by hand so every term is
visible.
"""

import numpy as np

np.random.seed(1)


# -------------------------------------------------------------------
# STEP 1: A small, noisy two-class dataset
# -------------------------------------------------------------------
# Two clusters with deliberate overlap (noise). A powerful network is
# tempted to wiggle around the noisy points -- exactly the overfitting
# that regularization is meant to curb.
def make_data(n=300):
    half = n // 2
    c0 = np.random.randn(half, 2) * 1.3 + np.array([-1.2, -1.2])
    c1 = np.random.randn(half, 2) * 1.3 + np.array([1.2, 1.2])
    X = np.vstack([c0, c1]).T            # (2, n): features in rows
    Y = np.array([[0] * half + [1] * half])
    # shuffle so train/test split is not all-one-class
    idx = np.random.permutation(n)
    return X[:, idx], Y[:, idx]


# -------------------------------------------------------------------
# STEP 2: A 3-layer network (2 -> 20 -> 10 -> 1), by hand
# -------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -500, 500)            # keep exp() from overflowing
    return 1.0 / (1.0 + np.exp(-z))

def initialize(layers):
    p = {}
    for l in range(1, len(layers)):
        # He initialization (Chapter 3): scale by sqrt(2 / fan_in)
        p['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2.0 / layers[l - 1])
        p['b' + str(l)] = np.zeros((layers[l], 1))
    return p

def forward(X, p):
    Z1 = p['W1'] @ X + p['b1']; A1 = relu(Z1)
    Z2 = p['W2'] @ A1 + p['b2']; A2 = relu(Z2)
    Z3 = p['W3'] @ A2 + p['b3']; A3 = sigmoid(Z3)
    return A3, (X, Z1, A1, Z2, A2, Z3, A3)


# -------------------------------------------------------------------
# STEP 3: Cost -- cross-entropy plus an optional penalty term
# -------------------------------------------------------------------
# The penalty is what makes large weights "expensive." L2 adds the sum of
# squared weights; L1 adds the sum of absolute weights. lambd controls the
# strength. Biases are NOT penalized -- almost all parameters live in W.
def compute_cost(A3, Y, p, lambd=0.0, mode="none"):
    m = Y.shape[1]
    eps = 1e-8
    cross_entropy = -np.mean(Y * np.log(A3 + eps) + (1 - Y) * np.log(1 - A3 + eps))
    penalty = 0.0
    if mode == "L2":
        penalty = (lambd / (2 * m)) * sum(np.sum(p['W' + str(l)] ** 2) for l in (1, 2, 3))
    elif mode == "L1":
        penalty = (lambd / m) * sum(np.sum(np.abs(p['W' + str(l)])) for l in (1, 2, 3))
    return cross_entropy + penalty


# -------------------------------------------------------------------
# STEP 4: Backprop -- the gradient gains an extra term to match the cost
# -------------------------------------------------------------------
# Because the penalty was added to the cost, its derivative must be added to
# each weight gradient. For L2 the extra piece is (lambd/m) * W. For L1 it is
# (lambd/m) * sign(W). The bias gradients are untouched, mirroring the cost.
def backward(cache, Y, p, lambd=0.0, mode="none"):
    X, Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[1]

    def reg_term(W):
        if mode == "L2":
            return (lambd / m) * W
        if mode == "L1":
            return (lambd / m) * np.sign(W)
        return 0.0

    dZ3 = A3 - Y
    g = {}
    g['dW3'] = (dZ3 @ A2.T) / m + reg_term(p['W3'])
    g['db3'] = np.sum(dZ3, axis=1, keepdims=True) / m
    dZ2 = (p['W3'].T @ dZ3) * (Z2 > 0)
    g['dW2'] = (dZ2 @ A1.T) / m + reg_term(p['W2'])
    g['db2'] = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (p['W2'].T @ dZ2) * (Z1 > 0)
    g['dW1'] = (dZ1 @ X.T) / m + reg_term(p['W1'])
    g['db1'] = np.sum(dZ1, axis=1, keepdims=True) / m
    return g

def update(p, g, lr):
    for l in (1, 2, 3):
        p['W' + str(l)] -= lr * g['dW' + str(l)]
        p['b' + str(l)] -= lr * g['db' + str(l)]
    return p


# -------------------------------------------------------------------
# STEP 5: Train once with a given regularization setting
# -------------------------------------------------------------------
def accuracy(X, Y, p):
    A3, _ = forward(X, p)
    return np.mean((A3 > 0.5).astype(int) == Y) * 100

def weight_magnitude(p):
    """Average absolute weight across all three weight matrices -- our
    'how big are the weights?' yardstick for the check."""
    total = sum(np.sum(np.abs(p['W' + str(l)])) for l in (1, 2, 3))
    count = sum(p['W' + str(l)].size for l in (1, 2, 3))
    return total / count

def train(Xtr, Ytr, mode="none", lambd=0.0, iterations=20000, lr=0.3):
    p = initialize([Xtr.shape[0], 20, 10, 1])
    for _ in range(iterations):
        A3, cache = forward(Xtr, p)
        g = backward(cache, Ytr, p, lambd=lambd, mode=mode)
        p = update(p, g, lr)
    return p


# -------------------------------------------------------------------
# STEP 6: Run baseline, L2, and L1; compare and check
# -------------------------------------------------------------------
def main():
    X, Y = make_data()
    Xtr, Ytr = X[:, :200], Y[:, :200]
    Xte, Yte = X[:, 200:], Y[:, 200:]

    runs = [
        ("none (baseline)", "none", 0.0),
        ("L2",              "L2",   0.5),
        ("L1",              "L1",   0.3),
    ]

    print("Effect of regularization (same network, same data)")
    print("-" * 64)
    print(f"{'Regularization':<18}{'Train acc':>11}{'Test acc':>10}{'Avg |weight|':>16}")
    results = {}
    for label, mode, lambd in runs:
        p = train(Xtr, Ytr, mode=mode, lambd=lambd)
        tr = accuracy(Xtr, Ytr, p)
        te = accuracy(Xte, Yte, p)
        mag = weight_magnitude(p)
        results[label] = (tr, te, mag)
        print(f"{label:<18}{tr:>10.1f}%{te:>9.1f}%{mag:>16.4f}")

    # ---------------------------------------------------------------
    # THE CHECK: did regularization actually shrink the weights, and
    # did it narrow the train/test gap?
    # ---------------------------------------------------------------
    print("\nCHECK: regularization should shrink weights and close the gap")
    print("-" * 64)
    base_tr, base_te, base_mag = results["none (baseline)"]
    base_gap = base_tr - base_te
    for label in ("L2", "L1"):
        tr, te, mag = results[label]
        gap = tr - te
        smaller = "yes" if mag < base_mag else "NO"
        closed = "yes" if gap < base_gap else "NO"
        print(f"  {label}: weights smaller than baseline? {smaller}   "
              f"train/test gap narrowed? {closed}")
    print(f"\n  baseline gap = {base_gap:.1f} points,  avg |weight| = {base_mag:.4f}")


if __name__ == "__main__":
    main()
