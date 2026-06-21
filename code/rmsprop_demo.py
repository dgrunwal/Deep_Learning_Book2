"""
RMSprop on the stretched ravine                                 rmsprop_demo.py
==============================================================================
A small, self-contained experiment for Chapter 11 of
"Improving Deep Neural Networks."

This is the SAME stretched-ravine problem Chapter 10 used for Momentum -- a
bowl that is 25x steeper in one direction than the other. Plain gradient
descent oscillates wildly across the steep direction while crawling along the
gentle one. Chapter 10 fixed that by AVERAGING the gradients (Momentum). This
chapter fixes it a different way: by DIVIDING each parameter's step by the
recent size of its own gradients (RMSprop).

The two functions below -- initialize_rmsprop() and
update_parameters_with_rmsprop() -- keep a running average of the SQUARED
gradient in a dictionary called s, using the hyperparameter beta2. That s
dictionary, and the divide-by-sqrt(s) step, are EXACTLY what Adam reuses in
Chapter 12 as its "second moment." Nothing here is thrown away; it is the
half of Adam you are building now.

We run plain gradient descent and RMSprop on the identical surface, from the
identical start, and print the path each one takes to the minimum. Only NumPy
is needed. Save it as rmsprop_demo.py.
"""

import numpy as np


# ===========================================================================
# SECTION 1 -- The two functions you are really building (used again in Adam)
# ===========================================================================
# These follow the exact same dictionary conventions as the rest of the book
# and the Part IV capstone: parameters live in params["W1"], params["b1"], ...
# and their squared-gradient averages live in s["dW1"], s["db1"], ...

def initialize_rmsprop(parameters):
    """Create the s dictionary: one ZERO array per parameter, the same shape
    as the gradient whose square it will track. Before the first step there
    are no past gradients to average, so zero is the honest start. This is
    identical to the s half of initialize_adam()."""
    L = len(parameters) // 2
    s = {}
    for l in range(1, L + 1):
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    return s


def update_parameters_with_rmsprop(parameters, grads, s, beta2=0.999,
                                   learning_rate=0.01, epsilon=1e-8):
    """One RMSprop step for every parameter.

    For each parameter:
        s = beta2 * s + (1 - beta2) * (gradient ** 2)     # avg of squares
        param -= learning_rate * gradient / (sqrt(s) + epsilon)

    The squaring is ELEMENT-WISE. Dividing by sqrt(s) shrinks the step in
    directions whose gradients have been large (the oscillating ones) and
    barely touches directions whose gradients have been small (the steady
    ones). epsilon just keeps the denominator away from zero.
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        # running average of the SQUARED gradient (the "mean square")
        s["dW" + str(l)] = (beta2 * s["dW" + str(l)]
                            + (1 - beta2) * (grads["dW" + str(l)] ** 2))
        s["db" + str(l)] = (beta2 * s["db" + str(l)]
                            + (1 - beta2) * (grads["db" + str(l)] ** 2))
        # step divided by the ROOT of that mean square -> "root mean square"
        parameters["W" + str(l)] -= (learning_rate * grads["dW" + str(l)]
                                     / (np.sqrt(s["dW" + str(l)]) + epsilon))
        parameters["b" + str(l)] -= (learning_rate * grads["db" + str(l)]
                                     / (np.sqrt(s["db" + str(l)]) + epsilon))
    return parameters, s


def update_parameters_with_gd(parameters, grads, learning_rate):
    """Plain gradient descent, for comparison: step straight along the
    gradient, no scaling, no memory."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


# ===========================================================================
# SECTION 2 -- The SAME stretched ravine from Chapter 10 (b vs w)
# ===========================================================================
# We minimize a bowl stretched 25x in one direction:
#
#       J(W) = 0.5 * ( W1^2  +  25 * W2^2 )
#
# Following the chapter's b-vs-w picture: W1 is the gentle "w" direction we
# want to keep moving along, and W2 is the steep "b" direction that
# oscillates. The gradient is [W1, 25*W2]: tiny in w, huge in b. Plain GD
# must keep its learning rate small or the b-wobble explodes. RMSprop divides
# the b-step by the root-mean-square of b's (large) gradients, shrinking it,
# while leaving w's (small) gradients almost untouched.
STRETCH = 25.0


def grad_J(parameters):
    W = parameters["W1"]
    dW = np.array([[W[0, 0]], [STRETCH * W[1, 0]]])
    return {"dW1": dW, "db1": np.zeros((1, 1))}


def cost_J(parameters):
    W = parameters["W1"]
    return 0.5 * (W[0, 0] ** 2 + STRETCH * W[1, 0] ** 2)


def start_point():
    # one weight vector of shape (2,1); a dummy b1 keeps the dict shape
    return {"W1": np.array([[8.0], [1.5]]), "b1": np.zeros((1, 1))}


# ===========================================================================
# SECTION 3 -- Run both optimizers on the identical surface
# ===========================================================================
def run(optimizer, steps=40, learning_rate=0.05, beta2=0.999):
    parameters = start_point()
    s = initialize_rmsprop(parameters)
    history = []
    for i in range(steps):
        grads = grad_J(parameters)
        if optimizer == "gd":
            parameters = update_parameters_with_gd(parameters, grads,
                                                   learning_rate)
        elif optimizer == "rmsprop":
            parameters, s = update_parameters_with_rmsprop(
                parameters, grads, s, beta2=beta2,
                learning_rate=learning_rate)
        w = parameters["W1"]
        history.append((i + 1, w[0, 0], w[1, 0], cost_J(parameters)))
    return history


def show(title, history, every=5):
    print(f"\n--- {title} ---")
    print(f"{'step':>5} {'w (gentle)':>12} {'b (steep)':>12} {'cost':>12}")
    for i, w1, w2, c in history:
        if i % every == 0 or i == 1:
            print(f"{i:>5} {w1:>12.4f} {w2:>12.4f} {c:>12.5f}")


def main():
    print("=" * 60)
    print(" Same ravine, same start (w=8.0, b=1.5), same learning rate.")
    print(" J(W) = 0.5 * (w^2 + 25 * b^2).  The b direction is 25x steeper.")
    print("=" * 60)

    # Plain GD at a rate the steep direction can just barely tolerate.
    gd = run("gd", learning_rate=0.07)
    show("PLAIN GRADIENT DESCENT (lr=0.07)", gd)

    # RMSprop at the SAME rate -- watch the steep 'b' wobble get damped.
    rms = run("rmsprop", learning_rate=0.07, beta2=0.999)
    show("RMSPROP (lr=0.07, beta2=0.999)", rms)

    print("\n" + "=" * 60)
    print(" FINAL POSITION  (target is w=0, b=0, cost=0)")
    print("=" * 60)
    _, gw, gb, gc = gd[-1]
    _, rw, rb, rc = rms[-1]
    print(f" {'':<22}{'w':>10}{'b':>10}{'cost':>12}")
    print(f" plain GD              {gw:>10.4f}{gb:>10.4f}{gc:>12.5f}")
    print(f" RMSprop               {rw:>10.4f}{rb:>10.4f}{rc:>12.5f}")
    print("\n Read the b column: plain GD's steep direction keeps flipping")
    print(" sign and barely settles; RMSprop scales that step down by the")
    print(" root-mean-square of b's own gradients, so it settles smoothly.")


if __name__ == "__main__":
    main()
