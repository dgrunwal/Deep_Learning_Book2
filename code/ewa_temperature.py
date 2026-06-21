"""
Exponentially Weighted Averages: the temperature curves        ewa_temperature.py
==============================================================================
A self-contained plotting script for Chapter 9 of
"Improving Deep Neural Networks."

It reproduces the classic London-temperature demonstration:

    * build a noisy daily-temperature series for one year
    * smooth it with an exponentially weighted average at three betas
          beta = 0.5   -> averages ~2 days   (jumpy, follows noise)
          beta = 0.9   -> averages ~10 days  (the usual sweet spot)
          beta = 0.98  -> averages ~50 days  (very smooth, laggy)
    * show what BIAS CORRECTION fixes: without it the curve starts near
      zero and has to "warm up"; with it the early days are right too.

The smoothing recursion here -- v = beta*v + (1-beta)*theta -- is the EXACT
machinery the Part IV capstone's Momentum and Adam optimizers run on the
GRADIENTS instead of on temperatures, and the 1/(1-beta**t) correction is
exactly Adam's bias-correction step. Learn it here on weather; recognize it
there on gradients.

Run it:
    python ewa_temperature.py
It saves ewa_temperature_curves.png and prints a small warm-up table.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")            # save to file; no display needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. A NOISY YEAR OF DAILY TEMPERATURES
#    A smooth seasonal curve (cold in winter, warm in summer) plus daily
#    noise. The smooth part is the "truth" the average is trying to recover;
#    the noise is what makes raw daily readings jumpy.
# ---------------------------------------------------------------------------
def make_temperatures(days=365, seed=1):
    np.random.seed(seed)
    t = np.arange(days)
    # seasonal swing: coldest near day 0 (January), warmest mid-year
    seasonal = 50 + 25 * np.sin(2 * np.pi * (t - 80) / 365.0)
    noise = 8.0 * np.random.randn(days)
    return seasonal + noise


# ---------------------------------------------------------------------------
# 2. THE EXPONENTIALLY WEIGHTED AVERAGE
#    v_t = beta * v_{t-1} + (1 - beta) * theta_t
#    Optionally apply bias correction: divide v_t by (1 - beta**t), which
#    matters only for the first handful of days and then fades to nothing.
# ---------------------------------------------------------------------------
def exp_weighted_average(theta, beta, bias_correction=False):
    v = 0.0
    out = np.zeros_like(theta, dtype=float)
    for t in range(len(theta)):
        v = beta * v + (1.0 - beta) * theta[t]          # the one-line recursion
        if bias_correction:
            out[t] = v / (1.0 - beta ** (t + 1))        # t+1 because t starts at 0
        else:
            out[t] = v
    return out


# ---------------------------------------------------------------------------
# 3. PLOT: three betas, and a bias-correction before/after panel
# ---------------------------------------------------------------------------
def main():
    temps = make_temperatures()
    days = np.arange(len(temps))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- LEFT: the three betas over the raw data --------------------------
    ax1.scatter(days, temps, s=6, color="#BBBBBB", label="raw daily temp")
    ax1.plot(days, exp_weighted_average(temps, 0.5),
             color="#E1A100", lw=1.6, label="beta = 0.5  (~2 days)")
    ax1.plot(days, exp_weighted_average(temps, 0.9),
             color="#C0392B", lw=2.0, label="beta = 0.9  (~10 days)")
    ax1.plot(days, exp_weighted_average(temps, 0.98),
             color="#2E7D32", lw=2.0, label="beta = 0.98 (~50 days)")
    ax1.set_title("Same data, three values of beta")
    ax1.set_xlabel("day of year")
    ax1.set_ylabel("temperature (deg F)")
    ax1.legend(loc="lower center", fontsize=8)

    # ---- RIGHT: bias correction, with and without, at beta = 0.98 ---------
    beta = 0.98
    no_corr = exp_weighted_average(temps, beta, bias_correction=False)
    corr = exp_weighted_average(temps, beta, bias_correction=True)
    ax2.scatter(days[:120], temps[:120], s=6, color="#BBBBBB",
                label="raw daily temp")
    ax2.plot(days[:120], no_corr[:120], color="#8E44AD", lw=2.0,
             label="beta = 0.98, NO bias correction")
    ax2.plot(days[:120], corr[:120], color="#2E7D32", lw=2.0,
             label="beta = 0.98, WITH bias correction")
    ax2.set_title("Bias correction fixes the warm-up (first ~120 days shown)")
    ax2.set_xlabel("day of year")
    ax2.set_ylabel("temperature (deg F)")
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig("ewa_temperature_curves.png", dpi=150)
    print("Saved ewa_temperature_curves.png")

    # ---- A small numeric table making the warm-up concrete ----------------
    print("\nThe warm-up problem, in numbers (beta = 0.98):")
    print(f"{'day':>5}{'raw temp':>11}{'no correction':>16}{'corrected':>12}")
    print("-" * 44)
    for d in [0, 1, 2, 4, 9, 19, 49, 99]:
        print(f"{d + 1:>5}{temps[d]:>11.1f}{no_corr[d]:>16.1f}{corr[d]:>12.1f}")
    print("\n Notice: without correction, day 1 reads far too low (the average\n"
          " starts at 0 and must climb). With correction it is right immediately.\n"
          " By day ~100 the two columns agree -- correction has faded to nothing.")


if __name__ == "__main__":
    main()
