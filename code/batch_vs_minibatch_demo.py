"""
Batch vs. Mini-batch vs. Stochastic                  batch_vs_minibatch_demo.py
==============================================================================
A self-contained teaching script for Chapter 8 of
"Improving Deep Neural Networks."

It trains the SAME tiny model three different ways so you can SEE how the
data-batching choice changes the path to the minimum:

    * Batch gradient descent      -> 1 weight update per epoch   (smooth, slow)
    * Mini-batch gradient descent -> many updates per epoch       (noisy, fast)
    * Stochastic gradient descent -> 1 update PER EXAMPLE          (very noisy)

All three use the EXACT same model, loss, and gradient. The ONLY thing that
changes between runs is how many examples we look at before each weight
update -- the mini-batch size. That single knob is the whole idea.

The mini-batch splitter below, random_mini_batches, is the SAME routine the
Part IV capstone uses to feed Momentum and Adam. Learning it here, on a model
you can hold in your head, means it is already familiar when the capstone
wraps the fancier optimizers around it.

Only NumPy is needed.

    python batch_vs_minibatch_demo.py
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# 1. MAKE SOME FAKE DATA
#    A simple linear relationship  y = 2*x + 1  with a little noise.
#    The model's JOB is to rediscover w = 2 and b = 1 from the data alone.
# ---------------------------------------------------------------------------
np.random.seed(0)
m = 2000                                  # number of training examples
X = np.random.randn(1, m)                 # shape (1 feature, m examples)
true_w, true_b = 2.0, 1.0
Y = true_w * X + true_b + 0.1 * np.random.randn(1, m)   # add a bit of noise


# ---------------------------------------------------------------------------
# 2. SPLIT A TRAINING SET INTO SHUFFLED MINI-BATCHES
#    This is the heart of the chapter -- and the exact function the capstone
#    reuses. Shuffle the examples, then slice them into chunks of
#    mini_batch_size. The final chunk may be smaller if m doesn't divide
#    evenly; that is fine and expected.
# ---------------------------------------------------------------------------
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """Split (X, Y) into a list of shuffled mini-batches."""
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: shuffle X and Y the SAME way, so pairs stay together.
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: carve out the full-size mini-batches.
    num_complete = math.floor(m / mini_batch_size)
    for k in range(num_complete):
        mb_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mb_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mb_X, mb_Y))

    # Step 3: the leftover partial mini-batch, if any.
    if m % mini_batch_size != 0:
        mb_X = shuffled_X[:, num_complete * mini_batch_size:]
        mb_Y = shuffled_Y[:, num_complete * mini_batch_size:]
        mini_batches.append((mb_X, mb_Y))

    return mini_batches


# ---------------------------------------------------------------------------
# 3. THE SHARED PIECES  (identical for all three methods)
#    model:  y_hat = w*x + b
#    loss :  mean squared error
#    grad :  derivatives of the loss w.r.t. w and b, over whatever slice of
#            data we hand it. SAME math regardless of how many examples --
#            which is exactly why the batching choice is independent of the
#            gradient itself.
# ---------------------------------------------------------------------------
def compute_cost(w, b, x, y):
    """Mean squared error over the examples in x, y."""
    y_hat = w * x + b
    return np.mean((y_hat - y) ** 2)


def compute_gradients(w, b, x, y):
    """Gradients of MSE w.r.t. w and b, averaged over the given examples."""
    n = x.shape[1]
    y_hat = w * x + b
    error = y_hat - y
    dw = (2.0 / n) * np.sum(error * x)
    db = (2.0 / n) * np.sum(error)
    return dw, db


# ---------------------------------------------------------------------------
# 4. ONE GENERIC TRAINER -- the batch SIZE is the only knob
#    batch_size = m  -> batch gradient descent      (one update per epoch)
#    batch_size = k  -> mini-batch gradient descent (about m/k updates/epoch)
#    batch_size = 1  -> stochastic gradient descent (m updates per epoch)
# ---------------------------------------------------------------------------
def train(batch_size, epochs=5, lr=0.1):
    w, b = 0.0, 0.0                       # start both parameters at zero
    cost_history = []                     # cost on the FULL set after each step

    for epoch in range(epochs):
        # New shuffle each epoch so mini-batches aren't always identical.
        mini_batches = random_mini_batches(X, Y, batch_size, seed=epoch)

        for (mb_X, mb_Y) in mini_batches:
            # One gradient-descent step using ONLY this mini-batch.
            dw, db = compute_gradients(w, b, mb_X, mb_Y)
            w -= lr * dw
            b -= lr * db

            # Record cost on the FULL set so the three curves are comparable.
            cost_history.append(compute_cost(w, b, X, Y))

    return w, b, cost_history


def curve_noisiness(history):
    """A rough 'how jagged is the cost curve' score: the average size of the
    step-to-step WRONG-WAY moves (where cost went UP instead of down).
    Smooth batch descent should be ~0; stochastic should be the largest."""
    ups = [max(0.0, history[i] - history[i - 1])
           for i in range(1, len(history))]
    return sum(ups) / len(ups) if ups else 0.0


# ---------------------------------------------------------------------------
# 5. RUN ALL THREE AND COMPARE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configs = [
        ("Batch       (size = m = 2000)", m),     # whole set per update
        ("Mini-batch  (size = 64)",       64),    # the practical default
        ("Stochastic  (size = 1)",        1),     # one example per update
    ]

    print("=" * 70)
    print(f" Goal: recover  w = {true_w},  b = {true_b}   (5 epochs, lr = 0.1)")
    print(" Same model, same gradient -- only the mini-batch size changes.")
    print("=" * 70)
    print(f"\n {'method':<30}{'updates':>9}{'learned w,b':>16}{'final cost':>13}")
    print(" " + "-" * 67)

    results = {}
    for name, size in configs:
        w, b, hist = train(batch_size=size)
        results[name] = hist
        print(f" {name:<30}{len(hist):>9}{f'{w:.3f}, {b:.3f}':>16}"
              f"{hist[-1]:>13.6f}")

    print("\n" + "=" * 70)
    print(" HOW NOISY WAS EACH PATH TO THE MINIMUM?")
    print("=" * 70)
    print(" (average 'wrong-way' move per step -- 0.0 means perfectly smooth)\n")
    for name, _ in configs:
        score = curve_noisiness(results[name])
        bar = "#" * min(40, int(score * 4000))
        print(f" {name:<30}{score:>9.5f}  {bar}")

    print("\n" + "=" * 70)
    print(" WHAT TO NOTICE")
    print("=" * 70)
    print(" * All three land near w=2, b=1 -- same model, same gradient math.")
    print(" * Batch made only 5 updates (1 per epoch). Mini-batch and")
    print("   stochastic made FAR more, reaching a good answer in fewer passes.")
    print(" * Batch's path is the smoothest (noise ~0), mini-batch is mildly")
    print("   noisy, stochastic is the noisiest -- the classic trade-off.")
    print(" * The ONLY thing that changed between runs was batch_size.")
