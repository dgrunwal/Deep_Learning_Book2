"""
tf_minimize_demo.py  --  Chapter 15 of "Improving Deep Neural Networks"
==============================================================================
The smallest possible TensorFlow program. We minimize  J(w) = w^2 - 10w + 25,
which equals (w - 5)^2, so the minimizer is w = 5. The lesson: we write ONLY
the forward cost. TensorFlow records it on a gradient tape and computes the
gradient (backprop) for us -- no calculus by hand.

Install (inside an activated virtual environment):  pip install tensorflow
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silences the oneDNN round-off notice

import tensorflow as tf

# w is the PARAMETER we want to optimize: a trainable variable, started at 0.
w = tf.Variable(0.0, dtype=tf.float32)

# Pick the optimizer in one line. Swapping it (e.g. to SGD) is a one-line change.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

def train_step():
    # 1. FORWARD ONLY: record the cost computation on the gradient tape.
    with tf.GradientTape() as tape:
        cost = w ** 2 - 10 * w + 25        # J(w) = (w - 5)^2
    # 2. TensorFlow plays the tape backward to get dCost/dw -- no hand calculus.
    grads = tape.gradient(cost, [w])
    # 3. The optimizer applies the gradient: one step downhill.
    optimizer.apply_gradients(zip(grads, [w]))

print('start: w =', w.numpy())            # 0.0
train_step()
print('after 1 step: w =', w.numpy())     # ~0.1

for _ in range(1000):                      # run many steps
    train_step()
print('after 1000 steps: w =', w.numpy()) # ~5.0  -- the minimum, found for us

