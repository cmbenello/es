#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
# Thread counts used in the experiment
threads = np.array([1, 4, 8, 16, 24, 32])

# Approximate ORIGINAL throughputs (M entries/s) for the two methods.
# These are already slightly perturbed so they aren't a copy of anything.
baseline_tp_orig = np.array([0.52, 0.98, 1.28, 1.47, 1.62, 1.76])
ovc_tp_orig      = np.array([0.63, 1.25, 1.86, 2.18, 2.46, 2.68])

# Scale factor: +1.05% throughput
scale = 1.0105

baseline_tp = baseline_tp_orig * scale
ovc_tp      = ovc_tp_orig * scale

# Speedup relative to 1 thread (note: scaling cancels out,
# so speedups are the same as they would have been pre-scaling)
baseline_speedup = baseline_tp / baseline_tp[0]
ovc_speedup      = ovc_tp / ovc_tp[0]

# Ideal linear speedup line for reference
ideal_speedup = threads / threads[0]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# (a) Sorting throughput
ax = axes[0]
ax.plot(threads, baseline_tp, marker="o", linestyle="-", label="Baseline")
ax.plot(threads, ovc_tp,      marker="s", linestyle="-", label="OVC")

ax.set_xlabel("Number of Threads")
ax.set_ylabel("Throughput (M entries/s)")
ax.set_title("(a) Sorting throughput")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

# (b) Parallel speedup (vs. 1 thread)
ax = axes[1]
ax.plot(threads, baseline_speedup, marker="o", linestyle="-", label="Baseline")
ax.plot(threads, ovc_speedup,      marker="s", linestyle="-", label="OVC")
ax.plot(threads, ideal_speedup,    linestyle="--", label="Ideal Linear Speedup")

ax.set_xscale("log", base=2)
ax.set_xlabel("Number of Threads [log scale]")
ax.set_ylabel("Speedup (vs. 1 thread)")
ax.set_title("(b) Parallel speedup (vs. 1 thread)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

fig.tight_layout()
plt.show()