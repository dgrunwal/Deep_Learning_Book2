"""
============================================================================
  optimization_methods_teaching_demo.py
============================================================================
  A SELF-CONTAINED teaching script that demonstrates the three optimization
  methods from the course assignment, using every component you implemented:

      Mini-batch gradient descent (the foundation):
          * random_mini_batches            -> split data into shuffled batches
          * update_parameters_with_gd      -> the plain GD update rule

      Momentum (smooth the gradients):
          * initialize_velocity            -> the running average v, set to 0
          * update_parameters_with_momentum-> step along the smoothed gradient

      Adam (momentum + RMSprop, combined):
          * initialize_adam                -> two running averages v and s
          * update_parameters_with_adam    -> bias-corrected adaptive step

      Learning rate decay (let simpler methods catch up):
          * update_lr                      -> smooth decay every epoch
          * schedule_lr_decay              -> staircase decay at fixed intervals

  Unlike the course notebook, this file needs NO external helper modules
  (no opt_utils, no testCases). Everything it uses is defined right here,
  so you can run it immediately:

      python optimization_methods_teaching_demo.py

  THE BIG IDEA: all three optimizers solve the same problem -- "in which
  direction, and how far, should I step the weights?" -- but each makes a
  smarter use of the gradient history than the last. Mini-batch GD uses only
  the current gradient. Momentum averages the recent gradients to cancel out
  noise. Adam also scales each parameter's step by how large its gradients
  have been. We train the SAME tiny network three ways and watch the cost
  curves to see the difference.
============================================================================
"""

import numpy as np
import math


# ===========================================================================
# SHARED BUILDING BLOCKS  (the model every optimizer trains, identically)
# ===========================================================================
# A real assignment uses a 3-layer network; here we use a tiny 2-parameter
# linear model y_hat = w*x + b so the WHOLE focus stays on the optimizers,
# not the network. The update rules below are byte-for-byte the same ones
# you wrote for the assignment -- only the model they wrap is simplified.

def sigmoid(z):
    """Sigmoid activation: squashes any number into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def make_data(m=2000, seed=3):
    """Two features on VERY different scales (1x and ~50x), so the cost
    surface is a long, stretched ravine -- the classic setting where plain
    gradient descent oscillates and Momentum/Adam pull ahead. The model's
    JOB is to rediscover w = [3.0, 0.05] and b = 1.0 from noisy data."""
    np.random.seed(seed)
    X = np.vstack([np.random.randn(1, m), 8.0 * np.random.randn(1, m)])  # (2, m)
    true_w = np.array([[3.0], [0.05]])
    Y = true_w.T @ X + 1.0 + 0.1 * np.random.randn(1, m)
    return X, Y


def compute_cost(params, X, Y):
    """Mean squared error -- the J we are trying to minimize."""
    y_hat = params["W1"] @ X + params["b1"]
    return float(np.mean((y_hat - Y) ** 2))


def compute_gradients(params, X, Y):
    """Gradients of MSE w.r.t. W1 and b1, averaged over the given examples.
    This is the SAME math regardless of how many examples we pass in -- which
    is exactly why the batching choice is independent of the gradient."""
    n = X.shape[1]
    y_hat = params["W1"] @ X + params["b1"]
    error = y_hat - Y                                  # (1, n)
    grads = {
        "dW1": (2.0 / n) * (error @ X.T),              # (1, 2)
        "db1": (2.0 / n) * np.sum(error, axis=1, keepdims=True),  # (1, 1)
    }
    return grads


def initialize_parameters(seed=0):
    """Start the weights at zero so we can watch them converge."""
    return {"W1": np.zeros((1, 2)), "b1": np.zeros((1, 1))}


# ===========================================================================
# PART A -- MINI-BATCH GRADIENT DESCENT
# ===========================================================================
# Two components: split the data into mini-batches, and apply the plain GD
# update once per batch.

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """COMPONENT: split (X, Y) into a list of shuffled mini-batches.
    Shuffle first so each epoch sees the data in a new order, then slice
    into chunks of mini_batch_size. The last chunk may be smaller."""
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: shuffle the columns (examples) of X and Y the SAME way.
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: carve out the full-size batches.
    num_complete = math.floor(m / mini_batch_size)
    for k in range(num_complete):
        mb_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mb_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mb_X, mb_Y))

    # Step 3: the leftover examples form one smaller final batch.
    if m % mini_batch_size != 0:
        mb_X = shuffled_X[:, num_complete * mini_batch_size:]
        mb_Y = shuffled_Y[:, num_complete * mini_batch_size:]
        mini_batches.append((mb_X, mb_Y))

    return mini_batches


def update_parameters_with_gd(parameters, grads, learning_rate):
    """COMPONENT: one step of plain gradient descent.
    Step each parameter directly against its own gradient. No history."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


# ===========================================================================
# PART B -- MOMENTUM
# ===========================================================================
# Momentum keeps a running (exponentially weighted) average of the gradients
# called the velocity, and steps along THAT instead of the raw gradient. The
# averaging cancels the back-and-forth noise and accelerates steady descent.

def initialize_velocity(parameters):
    """COMPONENT: the velocity v, one zero array per parameter.
    Starts at zero, so the very first step is just (1-beta) * gradient."""
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """COMPONENT: update the velocity, then step along it.
    v = beta * v + (1 - beta) * grad        # smoothed gradient
    param = param - learning_rate * v       # step along the smooth direction"""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]
    return parameters, v


# ===========================================================================
# PART C -- ADAM
# ===========================================================================
# Adam combines Momentum (average of gradients, v) with RMSprop (average of
# SQUARED gradients, s), bias-corrects both, then takes a step that is large
# where gradients have been small and small where they have been large.

def initialize_adam(parameters):
    """COMPONENT: two running averages, v (gradients) and s (squared
    gradients), each a zero array per parameter."""
    L = len(parameters) // 2
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
    """COMPONENT: the full Adam update, assembled from pieces you know.
        1. v  -- momentum average of the gradient
        2. v_corrected -- bias-correct it (divide by 1 - beta1**t)
        3. s  -- RMSprop average of the SQUARED gradient
        4. s_corrected -- bias-correct it (divide by 1 - beta2**t)
        5. step against v_corrected / (sqrt(s_corrected) + epsilon)
    t is the step count; it makes the bias correction fade as training goes on."""
    L = len(parameters) // 2
    v_corrected, s_corrected = {}, {}
    for l in range(1, L + 1):
        # 1. momentum-style average of the gradient
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        # 2. bias correction for the warm-up period
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
        # 3. RMSprop-style average of the squared gradient
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
        # 4. bias correction for the squared average
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
        # 5. the combined step: direction from v, scale from s
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
    return parameters, v, s


# ===========================================================================
# PART D -- LEARNING RATE DECAY
# ===========================================================================
# A fixed learning rate either overshoots near the minimum or crawls at the
# start. Shrinking it over time lets training settle tightly into the bottom.

def update_lr(learning_rate0, epoch_num, decay_rate):
    """COMPONENT: smooth decay -- shrinks a little every epoch.
    Always divides the ORIGINAL rate, so any epoch's rate is reproducible."""
    return learning_rate0 / (1 + decay_rate * epoch_num)


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """COMPONENT: staircase decay -- holds the rate steady inside each
    interval and drops it only when epoch_num crosses a multiple of
    time_interval. The floor() is what creates the flat steps."""
    return learning_rate0 / (1 + decay_rate * np.floor(epoch_num / time_interval))


# ===========================================================================
# PART E -- ONE MODEL, THREE OPTIMIZERS
# ===========================================================================
# The whole point: the model, the data, and the gradient are identical. The
# ONLY thing that changes is which update rule we plug in. This is exactly
# how the capstone compares the optimizer trio on the same dataset.

def model(X, Y, optimizer, learning_rate=0.05, mini_batch_size=64,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
          num_epochs=200, decay=None, decay_rate=1.0, seed=0):
    """Train the tiny model with the chosen optimizer and return the cost
    history so we can plot/compare the convergence curves."""
    parameters = initialize_parameters(seed=seed)
    costs = []
    t = 0                                  # Adam step counter
    lr0 = learning_rate

    # Set up the optimizer's memory once, before training.
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for epoch in range(num_epochs):
        # Optionally decay the learning rate at the start of each epoch.
        if decay is not None:
            if decay is schedule_lr_decay:
                learning_rate = decay(lr0, epoch, decay_rate, time_interval=10)
            else:
                learning_rate = decay(lr0, epoch, decay_rate)

        # Reshuffle into fresh mini-batches every epoch (new seed each time).
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed + epoch)

        for mb_X, mb_Y in mini_batches:
            grads = compute_gradients(parameters, mb_X, mb_Y)
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters, v, s = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        costs.append(compute_cost(parameters, X, Y))

    return parameters, costs


# ===========================================================================
# MAIN -- run all the demos in order
# ===========================================================================
if __name__ == "__main__":

    X, Y = make_data()
    print("=" * 68)
    print(" The task: recover  y = 3*x1 + 0.05*x2 + 1  from noisy data.")
    print(" The two features have very different scales (1x vs ~8x), so the")
    print(" cost surface is a stretched ravine -- hard for plain GD.")
    print(" Same model, same data, same gradient -- only the optimizer changes.")
    print("=" * 68)

    # Each optimizer gets a learning rate suited to it. Momentum and plain GD
    # share one; Adam's adaptive scaling means it usually wants a larger rate.
    LR = {"gd": 0.01, "momentum": 0.01, "adam": 0.05}
    EPOCHS = 60

    # ---- Train with each optimizer ----
    results = {}
    for name in ("gd", "momentum", "adam"):
        params, costs = model(X, Y, optimizer=name, learning_rate=LR[name], num_epochs=EPOCHS)
        results[name] = (params, costs)
        print(f"\n {name.upper():8s} (lr={LR[name]:.2f})  ->  "
              f"W1 = [{params['W1'][0, 0]:.3f}, {params['W1'][0, 1]:.4f}]  "
              f"b1 = {params['b1'].item():.3f}")
        print(f"             cost @ epoch 1 = {costs[0]:7.3f}   "
              f"@ epoch 3 = {costs[2]:7.3f}   final = {costs[-1]:.4f}")
    print(f"\n   (target W1 = [3.000, 0.0500], b1 = 1.000)")

    print("\n" + "=" * 68)
    print(" Reading the numbers above:")
    print(" * All three reach the answer -- on a problem this simple, none is")
    print("   dramatically better at the end. The lesson is about the PATH:")
    print(" * Compare COST @ EPOCH 1 and @ EPOCH 3. Momentum descends a little")
    print("   faster than plain GD by smoothing out the ravine's oscillations.")
    print(" * Adam needs a larger learning rate to shine; with its default it")
    print("   can be SLOWER on an easy convex bowl. Its real advantage shows")
    print("   on the harder, non-convex networks of the capstone -- which is")
    print("   exactly why we test all three rather than trusting one.")
    print("=" * 68)

    # ---- Show that decay helps a method settle ----
    print("\n LEARNING RATE DECAY: settling into the minimum.")
    _, costs_fixed = model(X, Y, optimizer="gd", learning_rate=0.015, num_epochs=EPOCHS)
    _, costs_decay = model(X, Y, optimizer="gd", learning_rate=0.015, num_epochs=EPOCHS,
                           decay=schedule_lr_decay, decay_rate=1.0)
    print(f"   plain GD, fixed LR=0.015   -> final cost = {costs_fixed[-1]:.4f}")
    print(f"   plain GD, decayed LR=0.015 -> final cost = {costs_decay[-1]:.4f}")
    print("   A larger fixed rate keeps bouncing around the minimum; decaying")
    print("   it shrinks the step over time so training settles in -- good")
    print("   tuning lets a simpler optimizer reach a Momentum/Adam-like result.\n")

    print("=" * 68)
    print(" TAKEAWAYS")
    print("=" * 68)
    print(" * Mini-batch GD updates after every small batch, so it makes many")
    print("   noisy steps per epoch instead of one slow step over all the data.")
    print(" * Momentum averages the recent gradients (velocity v) to cancel")
    print("   noise and accelerate -- one extra hyperparameter, beta ~ 0.9.")
    print(" * Adam adds a per-parameter scale (s, the squared-gradient average)")
    print("   on top of momentum, with bias correction. Its defaults")
    print("   beta1=0.9, beta2=0.999, epsilon=1e-8 rarely need tuning.")
    print(" * Learning rate decay shrinks the step over time so training")
    print("   settles tightly into the minimum -- and can let plain GD or")
    print("   Momentum approach Adam's result with the right schedule.")
    print(" * This is the exact recipe the Part IV capstone scales up to a")
    print("   real 3-layer network on a 2-D dataset, comparing cost curves")
    print("   and decision boundaries for all three optimizers.")
