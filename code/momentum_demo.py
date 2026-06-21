"""
Gradient Descent with Momentum                              momentum_demo.py
==============================================================================
A small, self-contained experiment for Chapter 10 of
"Improving Deep Neural Networks."

Momentum's one idea: instead of stepping along the raw gradient, keep a
running (exponentially weighted) average of recent gradients -- the VELOCITY
-- and step along that. On a stretched, ravine-shaped cost surface this
cancels the back-and-forth wobble while preserving the steady push toward the
minimum.

This file does two things:

    1. defines initialize_velocity() and update_parameters_with_momentum()
       EXACTLY as you implement them in the assignment and as the Part IV
       capstone uses them -- nothing renamed, nothing simplified away
    2. runs two experiments on the SAME stretched bowl:
         A) a safe learning rate both methods survive -- momentum takes the
            straighter path
         B) an aggressive learning rate where PLAIN GD DIVERGES but momentum
            stays stable -- the headline reason momentum lets you train faster

Only NumPy is needed. The parameter/gradient dictionaries use the same
"W1"/"b1" keys the rest of the book uses, so these two functions drop
straight into any model.
"""

import numpy as np

np.random.seed(1)


# ===========================================================================
# SECTION 1 -- The two functions you build (assignment + capstone, verbatim)
# ===========================================================================
def initialize_velocity(parameters):
    """Create the velocity dictionary v: one zero array per parameter, the
    same shape as the gradient it will smooth. Velocity starts at zero
    because, before any step, there is no history of gradients to average."""
    L = len(parameters) // 2          # number of layers
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """One momentum step. For each parameter:

        v = beta * v + (1 - beta) * grad      # update the averaged gradient
        param = param - learning_rate * v     # step along the AVERAGE, not grad

    beta controls how much history to keep (0.9 averages ~the last 10
    gradients). When beta = 0 this reduces to plain gradient descent.
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        # 1) roll the new gradient into the running average (the velocity)
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        # 2) step along the velocity instead of the raw gradient
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]
    return parameters, v


# ===========================================================================
# SECTION 2 -- A deliberately stretched problem (the classic ravine)
# ===========================================================================
# We minimize a simple bowl that is stretched 25x in one direction:
#
#       J(W) = 0.5 * ( W1^2  +  25 * W2^2 )
#
# The gradient is [W1, 25*W2]. Because the W2 direction is 25x steeper, plain
# gradient descent overshoots in W2 and OSCILLATES -- the W2 value flips sign
# on nearly every step -- while it can only crawl along the gentle W1
# direction. That oscillation is what forces you to keep the learning rate
# small: push the rate too high and the W2 wobble explodes. Momentum averages
# the gradients, so the flip-flopping W2 pushes cancel while the consistent
# W1 push survives -- letting you use a rate that would tear plain GD apart.
# We store the one weight as parameters["W1"] (shape (2,1)) so the SAME two
# functions above drive it, with no special-casing.
STRETCH = 25.0


def grad_J(parameters):
    W = parameters["W1"]
    dW = np.array([[W[0, 0]], [STRETCH * W[1, 0]]])
    return {"dW1": dW, "db1": np.zeros((1, 1))}


def cost_J(parameters):
    W = parameters["W1"]
    return 0.5 * (W[0, 0] ** 2 + STRETCH * W[1, 0] ** 2)


def start_point():
    # a dummy b1 keeps the dictionary shape the functions expect
    return {"W1": np.array([[8.0], [1.5]]), "b1": np.zeros((1, 1))}


# ===========================================================================
# SECTION 3 -- Train the same problem and record the path
# ===========================================================================
def run(beta, learning_rate, steps=30):
    """beta = 0.0 is plain gradient descent; beta = 0.9 is momentum."""
    parameters = start_point()
    v = initialize_velocity(parameters)
    history = []
    for t in range(steps + 1):
        W = parameters["W1"]
        history.append((t, W[0, 0], W[1, 0], cost_J(parameters)))
        grads = grad_J(parameters)
        parameters, v = update_parameters_with_momentum(
            parameters, grads, v, beta, learning_rate)
    return history


def show(label, hist):
    print(f"\n--- {label} ---")
    print(f"{'step':>5} {'W1':>9} {'W2':>9} {'cost':>13}")
    for t, w1, w2, c in hist:
        if t <= 7 or t % 5 == 0:
            print(f"{t:>5} {w1:>9.3f} {w2:>9.3f} {c:>13.4f}")


def main():
    # -------------------------------------------------------------------
    # EXPERIMENT A: a SAFE learning rate -- both survive, paths differ
    # -------------------------------------------------------------------
    print("=" * 70)
    print(" EXPERIMENT A  --  safe learning rate (0.06): both converge")
    print(" Watch the W2 column. Plain GD flips its sign every step")
    print(" (oscillation); momentum's W2 swings smoothly and its W1 falls")
    print(" faster. Goal: W1 = [0, 0], cost = 0.")
    print("=" * 70)
    show("PLAIN GRADIENT DESCENT (beta = 0.0)", run(0.0, 0.06))
    show("WITH MOMENTUM        (beta = 0.9)", run(0.9, 0.06))

    # -------------------------------------------------------------------
    # EXPERIMENT B: an AGGRESSIVE rate -- plain GD diverges, momentum holds
    # -------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print(" EXPERIMENT B  --  aggressive learning rate (0.085)")
    print(" This rate is past plain GD's stability limit: its cost EXPLODES.")
    print(" Momentum, by averaging away the W2 oscillation, stays stable and")
    print(" still converges. THIS is why momentum lets you train faster --")
    print(" it tolerates a bigger step.")
    print("=" * 70)
    show("PLAIN GRADIENT DESCENT (beta = 0.0)  -- diverges", run(0.0, 0.085))
    show("WITH MOMENTUM        (beta = 0.9)  -- stable", run(0.9, 0.085))

    gd = run(0.0, 0.085)[-1]
    mom = run(0.9, 0.085)[-1]
    print("\n" + "=" * 70)
    print(" FINAL COST AT THE AGGRESSIVE RATE (after 30 steps)")
    print("=" * 70)
    print(f"   plain GD (beta=0.0):  cost = {gd[3]:>14.2f}   (blew up)")
    print(f"   momentum (beta=0.9):  cost = {mom[3]:>14.4f}   (converged)")


if __name__ == "__main__":
    main()
