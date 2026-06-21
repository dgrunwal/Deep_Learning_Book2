"""
Gradient Checking: validate any backward pass                gradient_check.py
==============================================================================
A small, self-contained routine for Chapter 7 of
"Improving Deep Neural Networks."

Backprop bugs are SILENT. A network with a subtly wrong gradient still trains
-- it just trains badly, and you waste days blaming the data, the learning
rate, or your luck. Gradient checking ends that guessing. It compares the
gradients your backward pass produced against a slow-but-trustworthy NUMERICAL
estimate built only from the cost function, and reports a single number that
tells you whether to trust your backprop.

This file does three things:

    1. defines a tiny 3-layer network -- the SAME architecture the Part IV
       capstone trains (LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID)
    2. provides gradient_check(), a routine you can drop into ANY model
    3. runs it twice: once on a CORRECT backward pass (it passes) and once on
       a backward pass with a deliberate bug planted in it (it fails, and we
       hunt the bug down by component)

Only NumPy is needed. Everything is written out by hand so nothing is hidden.
"""

import numpy as np

np.random.seed(1)


# ===========================================================================
# SECTION 1 -- The network (same shape as the capstone)
# ===========================================================================
def initialize_parameters(layer_dims, seed=3):
    """He initialization, identical in spirit to the capstone's."""
    np.random.seed(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        params["W" + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l - 1])
                                * np.sqrt(2.0 / layer_dims[l - 1]))
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return params


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward_propagation(X, parameters):
    """Run the network forward; cache everything backprop will need."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache


def compute_cost(a3, Y):
    """Cross-entropy cost averaged over the examples."""
    m = Y.shape[1]
    a3 = np.clip(a3, 1e-12, 1 - 1e-12)
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    return (1.0 / m) * np.sum(logprobs)


def backward_propagation(X, Y, cache):
    """CORRECT gradients for every parameter, via backprop."""
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = (1.0 / m) * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,
            "dW3": dW3, "db3": db3}


def backward_propagation_with_bug(X, Y, cache):
    """The SAME backward pass with ONE planted bug: db2 is summed over the
    wrong axis. This is exactly the kind of silent error grad check exists to
    catch -- the shapes still broadcast, the network still trains, but the
    gradient for b2 is wrong."""
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = (1.0 / m) * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True) * 2.0   # <-- BUG: off by 2x

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,
            "dW3": dW3, "db3": db3}


# ===========================================================================
# SECTION 2 -- Roll parameters into one vector and back
# ===========================================================================
# Gradient checking treats the whole network as a single function J(theta) of
# one long parameter vector. To do that we must flatten every W and b into one
# column vector -- and be able to put a changed vector back into dictionary
# form so forward_propagation can use it. We also flatten the gradient
# dictionary the SAME way, so the two vectors line up element for element.

def dictionary_to_vector(params, keys):
    """Flatten the listed keys of a dict into one (n, 1) column vector."""
    vec = np.concatenate([params[k].reshape(-1, 1) for k in keys], axis=0)
    return vec


def vector_to_dictionary(vec, template, keys):
    """Inverse of the above: rebuild a parameter dict shaped like `template`."""
    out, start = {}, 0
    for k in keys:
        size = template[k].size
        out[k] = vec[start:start + size].reshape(template[k].shape)
        start += size
    return out


def gradients_to_vector(grads, keys):
    """Flatten the gradient dict in the SAME key order as the parameters."""
    grad_keys = ["d" + k for k in keys]
    return np.concatenate([grads[gk].reshape(-1, 1) for gk in grad_keys], axis=0)


# ===========================================================================
# SECTION 3 -- The gradient check itself
# ===========================================================================
def gradient_check(parameters, gradients, X, Y, epsilon=1e-7, verbose=True):
    """Compare analytic `gradients` against a two-sided numerical estimate.

    Returns the relative difference. Small (<= 1e-7) means the backward pass
    is almost certainly correct; large (>= 1e-3) means hunt for a bug.
    """
    keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
    theta = dictionary_to_vector(parameters, keys)        # all params, flat
    grad = gradients_to_vector(gradients, keys)           # analytic grads, flat
    num_params = theta.shape[0]

    grad_approx = np.zeros((num_params, 1))

    # For each scalar parameter: nudge it up by epsilon, then down by epsilon,
    # recompute the cost each time, and estimate the slope with a TWO-SIDED
    # difference. This needs only forward_propagation + compute_cost -- it
    # never looks at the backward pass, which is exactly why it can judge it.
    for i in range(num_params):
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        a3_plus, _ = forward_propagation(
            X, vector_to_dictionary(theta_plus, parameters, keys))
        J_plus = compute_cost(a3_plus, Y)

        theta_minus = np.copy(theta)
        theta_minus[i] -= epsilon
        a3_minus, _ = forward_propagation(
            X, vector_to_dictionary(theta_minus, parameters, keys))
        J_minus = compute_cost(a3_minus, Y)

        grad_approx[i] = (J_plus - J_minus) / (2.0 * epsilon)

    # Relative difference: Euclidean distance between the two gradient vectors,
    # normalized by their combined length so the verdict does not depend on
    # how big the gradients happen to be.
    numerator = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
    difference = numerator / denominator

    if verbose:
        if difference <= 2e-7:
            print(f"  PASS  relative difference = {difference:.2e}  "
                  "(backward pass looks correct)")
        elif difference <= 1e-5:
            print(f"  CHECK relative difference = {difference:.2e}  "
                  "(borderline -- inspect the components)")
        else:
            print(f"  FAIL  relative difference = {difference:.2e}  "
                  "(there is almost certainly a bug)")
    return difference, grad, grad_approx


def locate_bug(grad, grad_approx, parameters, keys=("W1", "b1", "W2", "b2",
                                                    "W3", "b3")):
    """When a check fails, report which parameter block holds the mismatch.
    Different slices of the flat vector belong to different W's and b's, so a
    large error confined to one slice points straight at the buggy gradient."""
    print("  Per-parameter error (where the trouble lives):")
    start = 0
    for k in keys:
        size = parameters[k].size
        g = grad[start:start + size]
        ga = grad_approx[start:start + size]
        denom = np.linalg.norm(g) + np.linalg.norm(ga) + 1e-12
        block_diff = np.linalg.norm(g - ga) / denom
        flag = "   <-- look here" if block_diff > 1e-5 else ""
        print(f"    d{k:<3} relative error = {block_diff:.2e}{flag}")
        start += size


# ===========================================================================
# SECTION 4 -- Run it: a correct pass (passes) and a buggy one (fails)
# ===========================================================================
def main():
    # a small batch is plenty for a gradient check -- speed matters here
    X = np.random.randn(4, 5)
    Y = (np.random.randn(1, 5) > 0).astype(float)
    params = initialize_parameters([4, 5, 3, 1])

    print("=" * 70)
    print(" GRADIENT CHECK 1: a CORRECT backward pass")
    print("=" * 70)
    _, cache = forward_propagation(X, params)
    grads_good = backward_propagation(X, Y, cache)
    diff_good, _, _ = gradient_check(params, grads_good, X, Y)

    print()
    print("=" * 70)
    print(" GRADIENT CHECK 2: a backward pass with ONE planted bug (db2 x2)")
    print("=" * 70)
    _, cache = forward_propagation(X, params)
    grads_bug = backward_propagation_with_bug(X, Y, cache)
    diff_bug, g, ga = gradient_check(params, grads_bug, X, Y)
    locate_bug(g, ga, params)

    print()
    print("=" * 70)
    print(" VERDICT")
    print("=" * 70)
    print(f" correct pass: {diff_good:.2e}  -> trust it")
    print(f" buggy pass:   {diff_bug:.2e}  -> the db2 block is wrong, as planted")


if __name__ == "__main__":
    main()
