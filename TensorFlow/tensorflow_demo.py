"""
tensorflow_demo.py
==================
A Practical TensorFlow Program -- companion demo.

This single script walks through every core idea from the Week 3 assignment
and the TensorFlow lecture, in the order you meet them:

    Part 1 -- tf.Variable, tf.constant, and the computation graph
    Part 2 -- Minimizing a cost with GradientTape + an optimizer
    Part 3 -- The assignment building blocks (linear, sigmoid, one-hot)
    Part 4 -- Glorot initialization + a full forward pass
    Part 5 -- The total loss with categorical cross-entropy (from_logits=True)

Everything runs top to bottom and prints its results.
"""

# --- Quiet TensorFlow's startup noise. These MUST be set before importing tf. ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence the oneDNN floating-point notice

import warnings
warnings.filterwarnings("ignore")          # hide Python-level deprecation warnings

# absl writes the "cpu_feature_guard" notice to stderr at first-op time;
# raising its verbosity to ERROR suppresses it. Must happen before tf import.
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")          # quiet TF's Python logger too

print("TensorFlow version:", tf.__version__)
print("=" * 60)


# ----------------------------------------------------------------------
# PART 1 -- Variables, constants, and the computation graph
# ----------------------------------------------------------------------
# A tf.Variable holds state you intend to CHANGE (the weights you train).
# A tf.constant holds state you will NOT change (your fixed input data).
# TensorFlow records the operations between them into a computation graph,
# which is what lets it work out the gradients for you automatically.

print("\nPART 1 -- Variables vs constants")

W = tf.Variable(0.0, dtype=tf.float32)      # trainable: state can change
x = tf.constant([1.0, -10.0, 25.0])         # fixed data: state cannot change

print("  W starts as:", W.numpy())
print("  x (fixed data):", x.numpy())


# ----------------------------------------------------------------------
# PART 2 -- Minimizing a cost with GradientTape and an optimizer
# ----------------------------------------------------------------------
# The lecture's motivating problem: minimise J(w) = w^2 - 10w + 25,
# which is (w - 5)^2, so the answer is w = 5. We never compute the
# derivative ourselves -- we only write FORWARD prop (the cost), and
# GradientTape "records" it so TensorFlow can replay it backwards.

print("\nPART 2 -- Minimizing a cost function")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

def cost():
    # J = x0*w^2 + x1*w + x2, with x = [1, -10, 25]  ->  w^2 - 10w + 25
    return x[0] * W ** 2 + x[1] * W + x[2]

def train_step():
    with tf.GradientTape() as tape:        # record the forward computation
        J = cost()
    trainable = [W]                        # the variables to optimize
    grads = tape.gradient(J, trainable)    # replay backwards for gradients
    optimizer.apply_gradients(zip(grads, trainable))

for _ in range(1000):                      # run many small steps
    train_step()

print("  After 1000 steps, W is approximately:", round(float(W.numpy()), 4))
print("  (the true minimum of (w-5)^2 is w = 5)")


# ----------------------------------------------------------------------
# PART 3 -- Assignment building blocks
# ----------------------------------------------------------------------
print("\nPART 3 -- Linear function, sigmoid, one-hot")

def linear_function():
    """Compute Y = WX + b with randomly drawn tensors (Exercise 1)."""
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name="X")   # data -> constant
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    # matmul does the matrix product; add does the bias addition
    return tf.add(tf.matmul(W, X), b)

def sigmoid(z):
    """Squash any real number into (0, 1) -- the logistic function (Exercise 2)."""
    z = tf.cast(z, tf.float32)             # sigmoid needs a float dtype
    return tf.keras.activations.sigmoid(z)

def one_hot_matrix(label, C=6):
    """Turn an integer label into a one-hot column of length C (Exercise 3)."""
    return tf.reshape(tf.one_hot(label, C, axis=0), shape=[C, ])

print("  linear_function() ->\n", linear_function().numpy())
print("  sigmoid(0)  =", float(sigmoid(0.0)))
print("  sigmoid(-1) =", round(float(sigmoid(-1)), 6))
print("  one_hot(label=2, C=5) =", one_hot_matrix(2, 5).numpy())


# ----------------------------------------------------------------------
# PART 4 -- Glorot initialization and a full forward pass
# ----------------------------------------------------------------------
# A 3-layer network: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR.
# We DELIBERATELY stop before softmax -- the loss step applies it for us.

print("\nPART 4 -- Initialize parameters + forward propagation")

def initialize_parameters(layer_dims):
    """Create W/b for each layer using the GlorotNormal initializer (Exercise 4)."""
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    params = {}
    for l in range(1, len(layer_dims)):
        # tf.Variable so these weights can be trained
        params[f"W{l}"] = tf.Variable(
            initializer(shape=(layer_dims[l], layer_dims[l - 1])))
        params[f"b{l}"] = tf.Variable(
            initializer(shape=(layer_dims[l], 1)))
    return params

def forward_propagation(X, parameters):
    """LINEAR -> RELU -> LINEAR -> RELU -> LINEAR, TF API only (Exercise 5)."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)   # logits -- no softmax yet
    return Z3

# Small toy network: 4 input features -> 5 -> 4 -> 3 classes
params = initialize_parameters([4, 5, 4, 3])
X_batch = tf.constant(np.random.randn(4, 2), dtype=tf.float32)   # 2 examples
logits = forward_propagation(X_batch, params)
print("  forward pass logits (3 classes x 2 examples):\n", logits.numpy())


# ----------------------------------------------------------------------
# PART 5 -- Total loss with categorical cross-entropy
# ----------------------------------------------------------------------
# from_logits=True tells the loss to apply softmax internally -- which is
# exactly why Part 4 left it out. We sum over the examples in the batch.

print("\nPART 5 -- Total loss")

def compute_total_loss(logits, labels):
    """Sum the categorical cross-entropy over a batch (Exercise 6)."""
    # categorical_crossentropy expects shape (examples, classes),
    # so transpose both from (classes, examples).
    return tf.reduce_sum(
        tf.keras.losses.categorical_crossentropy(
            tf.transpose(labels), tf.transpose(logits), from_logits=True))

# Two example labels, one-hot encoded as (classes=3, examples=2)
labels = tf.constant([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 0.0]], dtype=tf.float32)
loss = compute_total_loss(logits, labels)
print("  total loss over the batch:", round(float(loss), 6))

print("\n" + "=" * 60)
print("Done -- every concept from the assignment ran end to end.")
