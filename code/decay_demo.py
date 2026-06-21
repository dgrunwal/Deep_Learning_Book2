"""
Learning Rate Decay                                              decay_demo.py
==============================================================================
A small, self-contained experiment for Chapter 13 of
"Improving Deep Neural Networks."

The one idea: a FIXED learning rate forces a bad compromise -- big enough to
make progress early means too big to settle near the minimum; small enough to
settle means painfully slow at the start. LEARNING RATE DECAY removes the
compromise: start large for fast early progress, then shrink the rate over
time so the steps become careful as you approach the minimum.

This file does two things:
    1. defines update_lr() and schedule_lr_decay() EXACTLY as you implement
       them in the assignment and as the Part IV capstone uses them --
       nothing renamed, nothing simplified away
    2. applies decay to all THREE optimizers (plain GD, Momentum, Adam) on
       the same stretched bowl from Chapters 10-12, so you can read off the
       numbers what decay buys for each

Only NumPy is needed. The parameter/gradient dictionaries use the same
"W1"/"b1" keys the rest of the book uses.
"""

import numpy as np
np.random.seed(1)

# ===========================================================================
# SECTION 1 -- The two functions you build (assignment + capstone, verbatim)
# ===========================================================================
def update_lr(learning_rate0, epoch_num, decay_rate):
    """Smooth decay -- shrinks the learning rate a little every epoch.
    Implements  alpha = alpha0 / (1 + decay_rate * epoch_num)."""
    return learning_rate0 / (1 + decay_rate * epoch_num)


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """Staircase decay -- holds the rate steady, then drops it at fixed
    epoch intervals. np.floor turns the smooth schedule into steps:
    alpha = alpha0 / (1 + decay_rate * floor(epoch_num / time_interval))."""
    return learning_rate0 / (1 + decay_rate * np.floor(epoch_num / time_interval))


# ===========================================================================
# SECTION 2 -- The optimizer steps (from Chapters 8, 10, 12), unchanged
# ===========================================================================
def init_adam_state():
    return ({"dW1": np.zeros((1, 1)), "db1": np.zeros((1, 1))},
            {"dW1": np.zeros((1, 1)), "db1": np.zeros((1, 1))})


def cost_and_grads(params):
    """The stretched bowl from Chapters 10-12: gentle in w, 25x steep in b."""
    w, b = params["W1"][0, 0], params["b1"][0, 0]
    cost = 0.5 * (w * w / 25.0 + b * b * 25.0)
    grads = {"dW1": np.array([[w / 25.0]]), "db1": np.array([[b * 25.0]])}
    return cost, grads


def run(optimizer, lr0, decay=None, decay_rate=0.0, time_interval=10, steps=40):
    params = {"W1": np.array([[8.0]]), "b1": np.array([[1.5]])}   # same start
    v = {"dW1": np.zeros((1, 1)), "db1": np.zeros((1, 1))}
    v_adam, s_adam = init_adam_state()
    beta, beta1, beta2, eps = 0.9, 0.9, 0.999, 1e-8
    t = 0
    rows = []
    lr = lr0
    for i in range(1, steps + 1):
        # ---- compute this epoch's learning rate (the decay lives HERE) ----
        if decay is schedule_lr_decay:
            lr = decay(lr0, i, decay_rate, time_interval)
        elif decay is not None:
            lr = decay(lr0, i, decay_rate)
        cost, grads = cost_and_grads(params)
        if optimizer == "gd":
            params["W1"] -= lr * grads["dW1"]
            params["b1"] -= lr * grads["db1"]
        elif optimizer == "momentum":
            v["dW1"] = beta * v["dW1"] + (1 - beta) * grads["dW1"]
            v["db1"] = beta * v["db1"] + (1 - beta) * grads["db1"]
            params["W1"] -= lr * v["dW1"]
            params["b1"] -= lr * v["db1"]
        elif optimizer == "adam":
            t += 1
            for k in ("dW1", "db1"):
                v_adam[k] = beta1 * v_adam[k] + (1 - beta1) * grads[k]
                vc = v_adam[k] / (1 - beta1 ** t)
                s_adam[k] = beta2 * s_adam[k] + (1 - beta2) * (grads[k] ** 2)
                sc = s_adam[k] / (1 - beta2 ** t)
                key = "W1" if k == "dW1" else "b1"
                params[key] -= lr * vc / (np.sqrt(sc) + eps)
        if i == 1 or i % 10 == 0:
            rows.append((i, lr, cost))
    return rows, cost


def show(title, rows, final):
    print(title)
    print(" epoch    learning_rate         cost")
    for i, lr, c in rows:
        print(f"{i:5d}   {lr:14.6f}   {c:11.5f}")
    print(f"  -> final cost after 40 epochs: {final:.5f}\n")


if __name__ == "__main__":
    # For each optimizer: a FIXED large rate (overshoots) vs the SAME large
    # rate WITH decay (fast early, careful late). decay_rate chosen per method.
    print("====================  PLAIN GRADIENT DESCENT  ====================\n")
    r, f = run("gd", 0.09, decay=None)
    show("--- fixed lr = 0.09 (too large: never settles) ---", r, f)
    r, f = run("gd", 0.09, decay=update_lr, decay_rate=0.3)
    show("--- lr0 = 0.09 WITH update_lr decay (decay_rate=0.3) ---", r, f)

    print("====================  MOMENTUM (beta=0.9)  =======================\n")
    r, f = run("momentum", 0.09, decay=None)
    show("--- fixed lr = 0.09 ---", r, f)
    r, f = run("momentum", 0.09, decay=update_lr, decay_rate=0.3)
    show("--- lr0 = 0.09 WITH update_lr decay (decay_rate=0.3) ---", r, f)

    print("====================  ADAM  ======================================\n")
    r, f = run("adam", 0.50, decay=None)
    show("--- fixed lr = 0.50 ---", r, f)
    r, f = run("adam", 0.50, decay=schedule_lr_decay, decay_rate=1.0, time_interval=10)
    show("--- lr0 = 0.50 WITH schedule_lr_decay (staircase, interval=10) ---", r, f)
