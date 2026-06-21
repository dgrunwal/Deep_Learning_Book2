"""
The Adam Optimization Algorithm                                  adam_demo.py
==============================================================================
A small, self-contained experiment for Chapter 12 of
"Improving Deep Neural Networks."

Adam's one idea: combine the TWO ideas from the previous two chapters.
  - From Momentum (Ch 10): keep an exponentially weighted average of the
    gradients themselves -- the "first moment", here called v.
  - From RMSprop (Ch 11): keep an exponentially weighted average of the
    SQUARED gradients -- the "second moment", here called s.
Then bias-correct both (so the zero-initialized averages are not too small
in the first few steps) and take a step that is the momentum direction
DIVIDED BY the RMSprop scale.

This file does two things:
    1. defines initialize_adam() and update_parameters_with_adam() EXACTLY
       as you implement them in the assignment and as the Part IV capstone
       uses them -- nothing renamed, nothing simplified away
    2. runs all three optimizers -- plain GD, Momentum, and Adam -- on the
       SAME stretched bowl from the SAME start, so you can read straight off
       the numbers why Adam is the default

Only NumPy is needed. The parameter/gradient dictionaries use the same
"W1"/"b1" keys the rest of the book uses, so these functions drop straight
into any model.
"""

import numpy as np
np.random.seed(1)

# ===========================================================================
# SECTION 1 -- The two functions you build (assignment + capstone, verbatim)
# ===========================================================================
def initialize_adam(parameters):
    """Adam's two running averages, v (gradients) and s (squared gradients).
    Both start at zero, one array per parameter, matching the gradient shape.
    v is the Momentum term; s is the RMSprop term."""
    L = len(parameters) // 2          # number of layers
    v, s = {}, {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t,
                                learning_rate=0.01, beta1=0.9,
                                beta2=0.999, epsilon=1e-8):
    """The full Adam update: momentum average + RMSprop scale, bias-corrected.
    t is the step number (starts at 1); it powers the bias correction."""
    L = len(parameters) // 2
    v_corrected, s_corrected = {}, {}
    for l in range(1, L + 1):
        # 1) Momentum: running average of the gradient (the first moment)
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        # 2) Bias-correct it so early steps are not artificially small
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
        # 3) RMSprop: running average of the SQUARED gradient (second moment)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
        # 4) Bias-correct that too
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
        # 5) Step: momentum direction divided by the RMSprop scale
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
    return parameters, v, s


# ===========================================================================
# SECTION 2 -- The same stretched bowl used in Chapters 10 and 11
# ===========================================================================
# Cost surface: J = 0.5 * (w^2 / 25 + b^2 * 25)   -- gentle in w, steep in b.
# The gradient is (w/25, b*25). We reuse the "W1"/"b1" dictionary form so the
# generic Adam functions above apply unchanged.
def cost_and_grads(params):
    w = params["W1"][0, 0]
    b = params["b1"][0, 0]
    cost = 0.5 * (w * w / 25.0 + b * b * 25.0)
    grads = {"dW1": np.array([[w / 25.0]]), "db1": np.array([[b * 25.0]])}
    return cost, grads


def run(optimizer, lr, steps=40):
    params = {"W1": np.array([[8.0]]), "b1": np.array([[1.5]])}  # same start
    if optimizer == "momentum":
        v = {"dW1": np.zeros((1, 1)), "db1": np.zeros((1, 1))}
    elif optimizer == "adam":
        v, s = initialize_adam(params)
    t = 0
    rows = []
    for i in range(1, steps + 1):
        cost, grads = cost_and_grads(params)
        if optimizer == "gd":
            params["W1"] -= lr * grads["dW1"]
            params["b1"] -= lr * grads["db1"]
        elif optimizer == "momentum":
            beta = 0.9
            v["dW1"] = beta * v["dW1"] + (1 - beta) * grads["dW1"]
            v["db1"] = beta * v["db1"] + (1 - beta) * grads["db1"]
            params["W1"] -= lr * v["dW1"]
            params["b1"] -= lr * v["db1"]
        elif optimizer == "adam":
            t += 1
            params, v, s = update_parameters_with_adam(params, grads, v, s, t, lr)
        if i == 1 or i % 5 == 0:
            rows.append((i, params["W1"][0, 0], params["b1"][0, 0], cost))
    return rows


def show(title, rows):
    print(title)
    print(" step   w (gentle)    b (steep)         cost")
    for i, w, b, c in rows:
        print(f"{i:5d}   {w:10.4f}   {b:10.4f}   {c:10.5f}")
    print()


if __name__ == "__main__":
    # Plain GD and Momentum share a safe rate; Adam normalizes its own step
    # size, so it tolerates -- and benefits from -- a larger learning rate.
    show("--- PLAIN GRADIENT DESCENT (lr=0.07) ---", run("gd", 0.07))
    show("--- MOMENTUM (lr=0.07, beta=0.9) ---", run("momentum", 0.07))
    show("--- ADAM (lr=0.50, beta1=0.9, beta2=0.999, eps=1e-8) ---", run("adam", 0.50))
