"""
Comparing Three Initializations                       compare_initializations.py
==============================================================================
A small, self-contained experiment for Chapter 3 of
"Improving Deep Neural Networks."

We train the SAME deep network, on the SAME data, for the SAME number of
steps, three times -- changing only how the weights are first filled in:

    1. zeros        -- every weight starts at 0
    2. large random -- random weights, scaled up by 10 (too big on purpose)
    3. He           -- random weights scaled by sqrt(2 / fan_in)

The only thing that differs between the three runs is one line of code (the
weight scale). Everything else is identical. The accuracies you see at the
end show, concretely, that the starting point alone can decide whether a
network learns at all.

Only NumPy is needed -- we build the tiny network by hand so every step is
visible.
"""

import numpy as np

# A fixed seed makes the "random" numbers repeat run to run, so your results
# match the book exactly.
np.random.seed(3)


# -------------------------------------------------------------------
# STEP 1: A small two-class dataset (two interlocking clusters)
# -------------------------------------------------------------------
# We make 200 points in 2-D. Class 0 sits to the lower-left, class 1 to the
# upper-right, with enough overlap that the network must actually learn a
# boundary rather than memorize.
def make_data(n=200):
    half = n // 2
    class0 = np.random.randn(half, 2) * 0.7 + np.array([-1.0, -1.0])
    class1 = np.random.randn(half, 2) * 0.7 + np.array([1.0, 1.0])
    X = np.vstack([class0, class1]).T          # shape (2, n): features in rows
    Y = np.array([[0] * half + [1] * half])    # shape (1, n): labels in a row
    return X, Y


# -------------------------------------------------------------------
# STEP 2: The three initialization methods
# -------------------------------------------------------------------
# Each returns a dictionary of weight matrices W1, W2, ... and bias vectors
# b1, b2, .... The SHAPES are identical across all three methods; only the
# starting VALUES of the weights change.

def initialize_zeros(layers_dims):
    """Every weight and bias starts at zero -- the tempting mistake."""
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_large_random(layers_dims):
    """Random weights scaled up by 10 -- symmetry broken, but far too large."""
    np.random.seed(3)
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_he(layers_dims):
    """Random weights scaled by sqrt(2 / fan_in) -- the right size for ReLU."""
    np.random.seed(3)
    parameters = {}
    for l in range(1, len(layers_dims)):
        fan_in = layers_dims[l - 1]                       # inputs feeding the layer
        scale = np.sqrt(2.0 / fan_in)                     # the He scale
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * scale
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# -------------------------------------------------------------------
# STEP 3: The network -- forward pass, cost, backward pass, update
# -------------------------------------------------------------------
# A 3-layer network: 2 inputs -> 10 -> 5 -> 1, with ReLU in the hidden
# layers and a sigmoid at the output. Written out by hand so nothing hides.

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    # Clip z to a safe range so the exp() of a very large negative number
    # (which can happen with the oversized "large random" weights) does not
    # overflow and clutter the output with warnings.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, p):
    """Run data forward and cache the pieces backprop needs."""
    Z1 = p['W1'] @ X + p['b1']; A1 = relu(Z1)
    Z2 = p['W2'] @ A1 + p['b2']; A2 = relu(Z2)
    Z3 = p['W3'] @ A2 + p['b3']; A3 = sigmoid(Z3)
    cache = (X, Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def compute_cost(A3, Y):
    """Binary cross-entropy. A tiny epsilon keeps log() from blowing up."""
    m = Y.shape[1]
    eps = 1e-8
    cost = -np.mean(Y * np.log(A3 + eps) + (1 - Y) * np.log(1 - A3 + eps))
    return cost

def backward(cache, Y, p):
    """Compute the gradients layer by layer."""
    X, Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[1]
    dZ3 = A3 - Y
    grads = {}
    grads['dW3'] = (dZ3 @ A2.T) / m; grads['db3'] = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = p['W3'].T @ dZ3; dZ2 = dA2 * (Z2 > 0)
    grads['dW2'] = (dZ2 @ A1.T) / m; grads['db2'] = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = p['W2'].T @ dZ2; dZ1 = dA1 * (Z1 > 0)
    grads['dW1'] = (dZ1 @ X.T) / m; grads['db1'] = np.sum(dZ1, axis=1, keepdims=True) / m
    return grads

def update(p, grads, lr):
    for l in range(1, 4):
        p['W' + str(l)] -= lr * grads['dW' + str(l)]
        p['b' + str(l)] -= lr * grads['db' + str(l)]
    return p


# -------------------------------------------------------------------
# STEP 4: Train once with a given initializer and report accuracy
# -------------------------------------------------------------------
def train(X, Y, init_fn, iterations=8000, lr=0.05):
    layers_dims = [X.shape[0], 10, 5, 1]   # 2 -> 10 -> 5 -> 1
    p = init_fn(layers_dims)
    first_cost = None
    for i in range(iterations):
        A3, cache = forward(X, p)
        cost = compute_cost(A3, Y)
        if first_cost is None:
            first_cost = cost
        grads = backward(cache, Y, p)
        p = update(p, grads, lr)
    predictions = (A3 > 0.5).astype(int)
    accuracy = np.mean(predictions == Y) * 100
    return first_cost, cost, accuracy


# -------------------------------------------------------------------
# STEP 5: Run all three and print a comparison table
# -------------------------------------------------------------------
def main():
    X, Y = make_data()
    methods = [
        ("zeros",        initialize_zeros),
        ("large random", initialize_large_random),
        ("He",           initialize_he),
    ]

    print(f"{'Initialization':<16}{'First cost':>12}{'Final cost':>12}{'Train acc':>11}")
    print("-" * 51)
    for name, fn in methods:
        first_cost, final_cost, acc = train(X, Y, fn)
        print(f"{name:<16}{first_cost:>12.4f}{final_cost:>12.4f}{acc:>10.0f}%")


if __name__ == "__main__":
    main()
