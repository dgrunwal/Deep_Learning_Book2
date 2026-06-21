"""
Dropout: With and Without                                       dropout_demo.py
==============================================================================
A small, self-contained experiment for Chapter 6 of
"Improving Deep Neural Networks."

We train the SAME three-layer network on the SAME overfit-prone dataset twice,
changing exactly ONE thing:

    1. no dropout   -- keep_prob = 1.0 in every layer (plain network)
    2. with dropout -- keep_prob = 0.7 in the two hidden layers

After each run we record the training accuracy and the dev accuracy at every
epoch, then print the two curves side by side so you can SEE the train/dev gap
shrink when dropout is switched on. The whole point is the gap, not the peak
training number: a network that scores high on train and far lower on dev is
overfitting, and a healthy regularizer pulls those two numbers closer.

Only NumPy is needed. The network -- including the inverted-dropout mask and
its matching backward pass -- is written out by hand so nothing is hidden.
"""

import numpy as np

np.random.seed(1)


# -------------------------------------------------------------------
# STEP 1: A small, overfit-prone dataset
# -------------------------------------------------------------------
# Two interleaving clusters with a little noise. Few examples + a network
# with more than enough capacity = a setting where overfitting is easy to
# provoke, which is exactly when dropout earns its place.
def make_data(m=300, seed=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(2, m)
    # a curvy true boundary, then flip a few labels as label noise
    Y = (X[0, :] ** 2 + 0.5 * X[1, :] > 1.0).astype(int).reshape(1, m)
    flip = rng.rand(m) < 0.10
    Y[0, flip] = 1 - Y[0, flip]
    return X, Y


def split(X, Y, train_frac=0.5):
    m = X.shape[1]
    n_train = int(m * train_frac)
    return (X[:, :n_train], Y[:, :n_train],
            X[:, n_train:], Y[:, n_train:])


# -------------------------------------------------------------------
# STEP 2: Initialize a 2 -> 40 -> 40 -> 1 network (He scaling)
# -------------------------------------------------------------------
def initialize(layer_dims, seed=3):
    rng = np.random.RandomState(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        params["W" + str(l)] = (rng.randn(layer_dims[l], layer_dims[l - 1])
                                * np.sqrt(2.0 / layer_dims[l - 1]))
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return params


def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


# -------------------------------------------------------------------
# STEP 3: Forward pass WITH inverted dropout
# -------------------------------------------------------------------
# keep_prob is the probability a unit is KEPT. The drop probability is
# therefore 1 - keep_prob. After masking we divide by keep_prob -- the
# "inverted" step -- so the expected value of each activation is unchanged
# and test time needs no special scaling.
def forward(X, params, keep_prob=1.0, train=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cache = {"A0": X}
    A = X
    L = len(params) // 2
    for l in range(1, L):                      # hidden layers (ReLU + dropout)
        Z = params["W" + str(l)] @ A + params["b" + str(l)]
        A = relu(Z)
        if train and keep_prob < 1.0:
            D = (np.random.rand(*A.shape) < keep_prob).astype(float)
            A = A * D                          # zero out 1 - keep_prob of units
            A = A / keep_prob                  # inverted: rescale to keep E[A]
            cache["D" + str(l)] = D
        cache["A" + str(l)] = A
        cache["Z" + str(l)] = Z
    # output layer (sigmoid, never dropped)
    ZL = params["W" + str(L)] @ A + params["b" + str(L)]
    AL = sigmoid(ZL)
    cache["A" + str(L)] = AL
    return AL, cache


def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-12, 1 - 1e-12)
    return -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m


# -------------------------------------------------------------------
# STEP 4: Backward pass -- the SAME mask must be reused
# -------------------------------------------------------------------
# Whatever units were zeroed in forward prop must be zeroed in backprop, and
# the same divide-by-keep_prob applied, so the gradient matches the function
# that was actually evaluated.
def backward(Y, params, cache, keep_prob=1.0):
    grads = {}
    m = Y.shape[1]
    L = len(params) // 2
    AL = cache["A" + str(L)]
    dZ = AL - Y
    A_prev = cache["A" + str(L - 1)]
    grads["dW" + str(L)] = (dZ @ A_prev.T) / m
    grads["db" + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = params["W" + str(L)].T @ dZ
    for l in reversed(range(1, L)):
        if keep_prob < 1.0:
            dA_prev = dA_prev * cache["D" + str(l)]   # same mask
            dA_prev = dA_prev / keep_prob             # same rescale
        dZ = dA_prev * (cache["Z" + str(l)] > 0)      # ReLU derivative
        A_prev = cache["A" + str(l - 1)]
        grads["dW" + str(l)] = (dZ @ A_prev.T) / m
        grads["db" + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = params["W" + str(l)].T @ dZ
    return grads


def update(params, grads, lr):
    L = len(params) // 2
    for l in range(1, L + 1):
        params["W" + str(l)] -= lr * grads["dW" + str(l)]
        params["b" + str(l)] -= lr * grads["db" + str(l)]
    return params


def accuracy(X, Y, params):
    # IMPORTANT: dropout is OFF at evaluation time (train=False)
    AL, _ = forward(X, params, keep_prob=1.0, train=False)
    pred = (AL > 0.5).astype(int)
    return float(np.mean(pred == Y)) * 100.0


# -------------------------------------------------------------------
# STEP 5: Train once, recording train and dev accuracy each epoch
# -------------------------------------------------------------------
def train_model(Xtr, Ytr, Xdev, Ydev, keep_prob, epochs=3000, lr=0.05):
    params = initialize([2, 40, 40, 1])
    history = []
    for i in range(epochs):
        AL, cache = forward(Xtr, params, keep_prob=keep_prob,
                            train=True, seed=i)
        grads = backward(Ytr, params, cache, keep_prob=keep_prob)
        params = update(params, grads, lr)
        if i % 300 == 0 or i == epochs - 1:
            tr = accuracy(Xtr, Ytr, params)
            dv = accuracy(Xdev, Ydev, params)
            history.append((i, tr, dv))
    return params, history


def main():
    X, Y = make_data()
    Xtr, Ytr, Xdev, Ydev = split(X, Y)

    print("=" * 66)
    print(" Same network, same data -- only dropout changes.")
    print(" keep_prob = 1.0  ->  no dropout      (plain network)")
    print(" keep_prob = 0.7  ->  drop 30% of hidden units each step")
    print("=" * 66)

    print("\n--- NO DROPOUT (keep_prob = 1.0) ---")
    _, h_plain = train_model(Xtr, Ytr, Xdev, Ydev, keep_prob=1.0)
    print(f"{'epoch':>7} {'train%':>8} {'dev%':>8} {'gap':>7}")
    for i, tr, dv in h_plain:
        print(f"{i:>7} {tr:>8.1f} {dv:>8.1f} {tr - dv:>7.1f}")

    print("\n--- WITH DROPOUT (keep_prob = 0.7) ---")
    _, h_drop = train_model(Xtr, Ytr, Xdev, Ydev, keep_prob=0.7)
    print(f"{'epoch':>7} {'train%':>8} {'dev%':>8} {'gap':>7}")
    for i, tr, dv in h_drop:
        print(f"{i:>7} {tr:>8.1f} {dv:>8.1f} {tr - dv:>7.1f}")

    # final side-by-side summary
    _, p_tr, p_dv = h_plain[-1]
    _, d_tr, d_dv = h_drop[-1]
    print("\n" + "=" * 66)
    print(" FINAL COMPARISON")
    print("=" * 66)
    print(f" {'':<14}{'train%':>9}{'dev%':>9}{'gap':>9}")
    print(f" no dropout    {p_tr:>9.1f}{p_dv:>9.1f}{p_tr - p_dv:>9.1f}")
    print(f" with dropout  {d_tr:>9.1f}{d_dv:>9.1f}{d_tr - d_dv:>9.1f}")
    print("\n Read the GAP column: dropout trades a little training accuracy")
    print(" for a smaller train/dev gap -- the signature of less overfitting.")


if __name__ == "__main__":
    main()
