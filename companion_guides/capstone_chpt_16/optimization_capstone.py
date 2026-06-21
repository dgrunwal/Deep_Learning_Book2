"""
============================================================================
  optimization_capstone.py   --   PART IV CAPSTONE
============================================================================
  This is the capstone that the Part III teaching demo builds toward. It
  takes the EXACT SAME optimizer functions you implemented in the
  assignment -- random_mini_batches, update_parameters_with_gd,
  initialize_velocity, update_parameters_with_momentum, initialize_adam,
  update_parameters_with_adam, update_lr, schedule_lr_decay -- and instead
  of wrapping them around a tiny 2-parameter line, wraps them around a REAL
  3-layer neural network trained on a 2-D dataset you cannot separate with
  a straight line.

  Then it does the comparison the whole book has been pointing at:

      * train the SAME network with mini-batch gradient descent, then
        Momentum, then Adam -- on the SAME data, from the SAME start
      * plot all three cost curves on one chart
      * draw each optimizer's decision boundary so you can SEE what it learned
      * finally, add learning-rate-decay variants and show that good tuning
        lets a simpler optimizer approach Adam's result

  Self-contained: no opt_utils, no testCases. Just NumPy and Matplotlib.

      python optimization_capstone.py

  THE BIG IDEA (now on a hard problem): on the easy line of the teaching
  demo, all three optimizers tied. On a real non-convex network the gaps
  open up -- which is exactly why you test all three rather than trusting
  one. Everything below is the same machinery, finally on a problem worthy
  of it.
============================================================================
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")            # save figures to file; no display needed
import matplotlib.pyplot as plt


# ===========================================================================
# SECTION 1 -- THE OPTIMIZER TRIO  (identical to the assignment / teaching demo)
# ===========================================================================
# These eight functions are the heart of Part III. They are reproduced here
# UNCHANGED so the capstone proves a point: the very same code that trained a
# toy line now trains a real network. Nothing about an optimizer needs to know
# what model it is optimizing.

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """Split (X, Y) into a list of shuffled mini-batches."""
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete = math.floor(m / mini_batch_size)
    for k in range(num_complete):
        mb_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mb_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mb_X, mb_Y))

    if m % mini_batch_size != 0:
        mb_X = shuffled_X[:, num_complete * mini_batch_size:]
        mb_Y = shuffled_Y[:, num_complete * mini_batch_size:]
        mini_batches.append((mb_X, mb_Y))

    return mini_batches


def update_parameters_with_gd(parameters, grads, learning_rate):
    """Plain gradient descent: step each parameter against its gradient."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


def initialize_velocity(parameters):
    """Momentum's velocity v, one zero array per parameter."""
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """Update the velocity (smoothed gradient), then step along it."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]
    return parameters, v


def initialize_adam(parameters):
    """Adam's two running averages, v (gradients) and s (squared gradients)."""
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
    """The full Adam update: momentum average + RMSprop scale, bias-corrected."""
    L = len(parameters) // 2
    v_corrected, s_corrected = {}, {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
    return parameters, v, s


def update_lr(learning_rate0, epoch_num, decay_rate):
    """Smooth decay -- shrinks the learning rate a little every epoch."""
    return learning_rate0 / (1 + decay_rate * epoch_num)


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """Staircase decay -- drops the rate at fixed epoch intervals."""
    return learning_rate0 / (1 + decay_rate * np.floor(epoch_num / time_interval))


# ===========================================================================
# SECTION 2 -- THE REAL 3-LAYER NETWORK  (what the optimizers now train)
# ===========================================================================
# LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
# This is the model from the assignment, written out so the file stands alone.
# The optimizers above do not change; only the thing they optimize got bigger.

def sigmoid(z):
    # clip the input so np.exp never overflows on large negative z
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def initialize_parameters(layers_dims, seed=3):
    """He initialization: random weights scaled so signal survives depth."""
    np.random.seed(seed)
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = (np.random.randn(layers_dims[l], layers_dims[l - 1])
                                    * np.sqrt(2.0 / layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


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
    """Cross-entropy cost averaged over the examples in this batch."""
    m = Y.shape[1]
    # clip to avoid log(0); harmless and keeps the demo numerically calm
    a3 = np.clip(a3, 1e-12, 1 - 1e-12)
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    return (1.0 / m) * np.sum(logprobs)


def backward_propagation(X, Y, cache):
    """The gradients for every parameter, via backprop (already verified by
    gradient checking back in Part II -- so here we simply trust it)."""
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

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


def predict(X, parameters):
    """Class prediction (0 or 1) for each example."""
    a3, _ = forward_propagation(X, parameters)
    return (a3 > 0.5).astype(int)


def accuracy(X, Y, parameters):
    """Fraction of examples classified correctly."""
    preds = predict(X, parameters)
    return float(np.mean(preds == Y))


# ===========================================================================
# SECTION 3 -- A HARD, NON-LINEAR DATASET
# ===========================================================================
# Two interleaving half-moons. No straight line can separate them, so the
# network has to learn a curved boundary -- the kind of problem where the
# optimizer choice actually matters.

def load_moons(m=600, noise=0.20, seed=1):
    """Generate a 2-D two-moons dataset. Returns X (2, m) and Y (1, m)."""
    np.random.seed(seed)
    n = m // 2
    # upper moon
    t1 = np.linspace(0, np.pi, n)
    x1 = np.stack([np.cos(t1), np.sin(t1)])
    # lower moon, shifted
    t2 = np.linspace(0, np.pi, n)
    x2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    X = np.concatenate([x1, x2], axis=1)
    X += noise * np.random.randn(2, X.shape[1])
    Y = np.concatenate([np.zeros(n), np.ones(n)]).reshape(1, -1)
    # shuffle
    perm = np.random.permutation(X.shape[1])
    return X[:, perm], Y[:, perm].astype(int)


# ===========================================================================
# SECTION 4 -- THE MODEL LOOP  (same structure as the assignment's model())
# ===========================================================================
# One function, three optimizer modes, optional learning-rate decay. This is
# the assignment's model() -- trimmed of plotting, with decay folded in.

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
          num_epochs=5000, decay=None, decay_rate=1.0, time_interval=1000,
          record_every=10):
    """Train the network with the chosen optimizer; return parameters and the
    recorded cost history (one value every `record_every` epochs)."""
    L = len(layers_dims)
    costs = []
    t = 0                                   # Adam step counter
    seed = 10                               # reproducible mini-batch shuffles
    m = X.shape[1]
    lr0 = learning_rate

    parameters = initialize_parameters(layers_dims)

    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        # Optional learning-rate decay, computed once per epoch.
        if decay is not None:
            if decay is schedule_lr_decay:
                learning_rate = decay(lr0, i, decay_rate, time_interval)
            else:
                learning_rate = decay(lr0, i, decay_rate)

        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for (mb_X, mb_Y) in minibatches:
            a3, cache = forward_propagation(mb_X, parameters)
            cost_total += compute_cost(a3, mb_Y)
            grads = backward_propagation(mb_X, mb_Y, cache)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters, v, s = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        if i % record_every == 0:
            costs.append(cost_total / m)

    return parameters, costs


# ===========================================================================
# SECTION 5 -- PLOTTING  (cost curves + decision boundaries)
# ===========================================================================

def plot_cost_curves(histories, record_every, filename,
                     title="Cost vs. epoch -- same network, three optimizers"):
    """One chart, several cost curves -- a convergence comparison.
    Pass `title` to give a specific chart its own heading."""
    plt.figure(figsize=(7, 4.5))
    colors = {"Mini-batch GD": "#c0392b", "Momentum": "#2e86de", "Adam": "#27ae60",
              "GD, small fixed LR": "#2e86de", "GD, large decayed LR": "#e67e22"}
    # Distinct line styles/widths so curves that nearly overlap stay visible.
    # (At a small learning rate, Mini-batch GD and Momentum trace almost the
    #  same path; drawing Momentum thinner and dashed lets the GD line show
    #  through underneath instead of being hidden.)
    styles = {
        "Mini-batch GD":        {"linewidth": 3.0, "linestyle": "-",  "zorder": 1},
        "Momentum":             {"linewidth": 1.6, "linestyle": "--", "zorder": 2},
        "Adam":                 {"linewidth": 2.0, "linestyle": "-",  "zorder": 3},
    }
    for name, costs in histories.items():
        epochs = np.arange(len(costs)) * record_every
        style = styles.get(name, {"linewidth": 2.0, "linestyle": "-"})
        plt.plot(epochs, costs, label=name, color=colors.get(name), **style)
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()


def plot_decision_boundaries(models, X, Y, filename):
    """One row of panels: each optimizer's learned boundary over the data."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6))
    if n == 1:
        axes = [axes]

    # shared grid covering the data
    pad = 0.5
    x_min, x_max = X[0].min() - pad, X[0].max() + pad
    y_min, y_max = X[1].min() - pad, X[1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()].T

    for ax, (name, params) in zip(axes, models.items()):
        Z = predict(grid, params).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.30, cmap=plt.cm.coolwarm, levels=1)
        ax.scatter(X[0], X[1], c=Y.ravel(), s=14, cmap=plt.cm.coolwarm,
                   edgecolors="k", linewidths=0.3)
        acc = accuracy(X, Y, params)
        ax.set_title(f"{name}\ntrain accuracy = {acc:.1%}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Decision boundaries learned by each optimizer", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close()


# ===========================================================================
# MAIN -- run the full capstone comparison
# ===========================================================================
if __name__ == "__main__":

    print("=" * 70)
    print(" PART IV CAPSTONE: one network, three optimizers, one hard dataset")
    print("=" * 70)

    X, Y = load_moons(m=600, noise=0.30, seed=1)
    layers_dims = [2, 16, 8, 1]              # 2 inputs -> 16 -> 8 -> 1 output
    EPOCHS = 3000
    REC = 10
    print(f" data: {X.shape[1]} points, 2 features, 2 noisy interleaving moons")
    print(f" network: {layers_dims}   epochs: {EPOCHS}\n")

    # ---- Part A: the three optimizers, SAME small learning rate ----
    # Holding the learning rate fixed across all three is the fair test: any
    # difference is the optimizer, not the step size. We use the assignment's
    # small rate (0.0007), which is where the optimizer choice matters most.
    configs = [
        ("Mini-batch GD", "gd",       {"learning_rate": 0.0007}),
        ("Momentum",      "momentum", {"learning_rate": 0.0007, "beta": 0.9}),
        ("Adam",          "adam",     {"learning_rate": 0.0007}),
    ]

    histories = {}
    trained = {}
    for name, opt, kw in configs:
        params, costs = model(X, Y, layers_dims, optimizer=opt,
                              num_epochs=EPOCHS, record_every=REC, **kw)
        histories[name] = costs
        trained[name] = params
        lr = kw["learning_rate"]
        print(f" {name:14s} (lr={lr:<5}) -> final cost = {costs[-1]:.4f}   "
              f"train accuracy = {accuracy(X, Y, params):.1%}")

    plot_cost_curves(histories, REC, "capstone_cost_curves.png")
    plot_decision_boundaries(trained, X, Y, "capstone_decision_boundaries.png")

    print("\n" + "-" * 70)
    print(" Reading the result:")
    print(" * On this noisy, curved problem the optimizers now SEPARATE,")
    print("   unlike the easy line in the teaching demo where they tied.")
    print(" * At this small, shared learning rate Adam pulls clearly ahead --")
    print("   lower cost and higher accuracy -- because its per-parameter")
    print("   scaling makes far better use of each tiny step.")
    print(" * Momentum and plain GD are close here: at so small a step, the")
    print("   smoothing has little room to help. Momentum's edge grows when")
    print("   the learning rate is larger and the path oscillates more.")
    print(" * See capstone_cost_curves.png and capstone_decision_boundaries.png")
    print("-" * 70)

    # ---- Part B: learning-rate decay lets a simpler optimizer catch up ----
    # The honest lesson: decay lets plain GD safely START with a LARGE step
    # (fast early progress) and still settle (no bouncing at the end). We
    # compare the small fixed rate from Part A against a large decayed rate.
    print("\n LEARNING RATE DECAY: can good tuning close the gap to Adam?")
    params_small, costs_small = model(X, Y, layers_dims, optimizer="gd",
                                      learning_rate=0.0007, num_epochs=EPOCHS, record_every=REC)
    params_decay, costs_decay = model(X, Y, layers_dims, optimizer="gd",
                                      learning_rate=0.05, num_epochs=EPOCHS, record_every=REC,
                                      decay=schedule_lr_decay, decay_rate=1.0, time_interval=1000)
    print(f"   plain GD, small fixed LR=0.0007  -> final cost = {costs_small[-1]:.4f}   "
          f"accuracy = {accuracy(X, Y, params_small):.1%}")
    print(f"   plain GD, large decayed LR0=0.05 -> final cost = {costs_decay[-1]:.4f}   "
          f"accuracy = {accuracy(X, Y, params_decay):.1%}")
    print("   The large rate makes fast early progress; decaying it stops the")
    print("   step from overshooting near the minimum -- so plain GD reaches")
    print("   a much better answer, closing most of the gap to Adam.")

    # decay comparison chart
    plot_cost_curves(
        {"GD, small fixed LR": costs_small, "GD, large decayed LR": costs_decay},
        REC, "capstone_decay_comparison.png",
        title="Cost vs. epoch -- learning rate decay vs. a small fixed rate")

    print("\n" + "=" * 70)
    print(" TAKEAWAYS")
    print("=" * 70)
    print(" * The optimizer code did not change from the teaching demo -- only")
    print("   the model it wraps. The same eight functions now train a real")
    print("   3-layer network on a problem no straight line can solve.")
    print(" * On a hard, non-convex surface the optimizers genuinely differ:")
    print("   here Adam reached the cleanest boundary; how much Momentum helps")
    print("   depends on the learning rate and how much the path oscillates.")
    print(" * Learning-rate decay is the tuning lever that lets a simpler")
    print("   optimizer approach the result of a fancier one.")
    print(" * Three figures were saved:")
    print("     - capstone_cost_curves.png         (convergence comparison)")
    print("     - capstone_decision_boundaries.png (what each one learned)")
    print("     - capstone_decay_comparison.png    (decay vs fixed rate)")
    print("=" * 70)
