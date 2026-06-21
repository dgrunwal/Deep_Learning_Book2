"""
Bias / Variance Diagnosis                              bias_variance_diagnosis.py
==============================================================================
A tiny diagnostic tool for Chapter 2 of "Improving Deep Neural Networks."

Given a model's TRAINING error and DEVELOPMENT error -- plus an estimate of
the best error any model could reach on this problem (the "Bayes" or optimal
error) -- this script classifies the situation into one of four buckets:

    1. High bias                 (underfitting)
    2. High variance             (overfitting)
    3. High bias AND high variance
    4. Low bias and low variance ("just right")

It then prints a short, plain-language recommendation for what to try next,
echoing the basic recipe from this chapter.

No deep-learning libraries are needed -- the whole point is that diagnosis is
done with two error numbers, not with the model itself.
"""

# -------------------------------------------------------------------
# STEP 1: The decision rule
# -------------------------------------------------------------------
# We compare two gaps against a tolerance:
#
#   avoidable bias = train_error - optimal_error
#       (how far the model is from the best achievable on data it HAS seen)
#
#   variance       = dev_error   - train_error
#       (how much worse the model does on data it has NOT seen)
#
# "High" simply means the gap is larger than `threshold` percentage points.

def diagnose(train_error, dev_error, optimal_error=0.0, threshold=2.0):
    """Return a (label, recommendation) pair from two error numbers.

    All errors are percentages, e.g. 15.0 means 15% error.
    """
    avoidable_bias = train_error - optimal_error
    variance = dev_error - train_error

    high_bias = avoidable_bias > threshold
    high_variance = variance > threshold

    if high_bias and high_variance:
        label = "High bias AND high variance"
        advice = ("Bigger/deeper network AND more data or regularization -- "
                  "attack both problems.")
    elif high_bias:
        label = "High bias (underfitting)"
        advice = ("Train longer, use a bigger network, or try a better "
                  "architecture.")
    elif high_variance:
        label = "High variance (overfitting)"
        advice = ("Get more training data, add regularization (L2, dropout), "
                  "or simplify the model.")
    else:
        label = "Low bias and low variance (just right)"
        advice = "Looks healthy. Ship it, or push the optimal error lower."

    return label, advice


# -------------------------------------------------------------------
# STEP 2: A few example "models" to classify
# -------------------------------------------------------------------
# Each entry is (name, train_error %, dev_error %). We assume a near-zero
# optimal (human-level) error here, which is typical for, say, cat photos.
examples = [
    ("Model A", 1.0, 11.0),   # great on train, poor on dev
    ("Model B", 15.0, 16.0),  # poor on both, small gap
    ("Model C", 15.0, 30.0),  # poor on train AND big gap
    ("Model D", 0.5, 1.0),    # great on both
]

OPTIMAL_ERROR = 0.0   # best achievable error for this problem (%)
THRESHOLD = 2.0       # gap (in % points) above which we call something "high"


# -------------------------------------------------------------------
# STEP 3: Run the diagnosis and print a small report
# -------------------------------------------------------------------
def main():
    print(f"Assuming optimal (best-possible) error = {OPTIMAL_ERROR:.1f}%")
    print(f"Calling any gap above {THRESHOLD:.1f}% points \"high\".\n")

    header = f"{'Model':<8} {'Train':>7} {'Dev':>7}   Diagnosis"
    print(header)
    print("-" * len(header))

    for name, train_err, dev_err in examples:
        label, advice = diagnose(train_err, dev_err, OPTIMAL_ERROR, THRESHOLD)
        print(f"{name:<8} {train_err:>6.1f}% {dev_err:>6.1f}%   {label}")
        print(f"{'':<8} {'':>7} {'':>7}   -> {advice}\n")


if __name__ == "__main__":
    main()
