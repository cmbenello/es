#!/usr/bin/env python3
"""
Create combined heatmaps for Exp6 and Exp1 (GenSort dataset):
- Exp6: 2D grid (RunGen threads x Merge threads)
- Exp1: Single row (total threads, RunGen=Merge)
Three heatmaps: Total Time, Total I/O, and Throughput
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
LOG_DIR = './gensort_result'
OUTPUT_DIR = './'

def parse_exp6_log(filepath):
    """Parse Exp6 log file with RunGen and Merge thread configuration."""
    filename = os.path.basename(filepath)

    # Extract RunGen and Merge threads from filename
    match = re.search(r'Exp6_RunGen(\d+)_Merge(\d+)_Mem2GB\.log', filename)
    if not match:
        return None

    rungen_threads = int(match.group(1))
    merge_threads = int(match.group(2))

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract average stats from the summary table
    avg_match = re.search(r'Exp6_RunGen\d+_Merge\d+_Mem2GB\[avg\]\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)

    if not avg_match:
        return None

    total_time = float(avg_match.group(1))  # seconds
    rungen_time = float(avg_match.group(2))  # seconds
    merge_time = float(avg_match.group(3))  # seconds
    throughput = float(avg_match.group(4))  # M entries/s
    read_mb = float(avg_match.group(5))  # MB
    write_mb = float(avg_match.group(6))  # MB

    total_io_mb = read_mb + write_mb

    return {
        'rungen_threads': rungen_threads,
        'merge_threads': merge_threads,
        'total_time': total_time,
        'rungen_time': rungen_time,
        'merge_time': merge_time,
        'throughput': throughput,
        'total_io_mb': total_io_mb
    }

def parse_exp1_log(filepath):
    """Parse Exp1 log file with symmetric thread configuration."""
    filename = os.path.basename(filepath)

    # Extract threads from filename
    match = re.search(r'Exp1_Thr(\d+)_Mem2GB\.log', filename)
    if not match:
        return None

    threads = int(match.group(1))

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract average stats from the summary table
    avg_match = re.search(r'Exp1_Thr\d+_Mem2GB\[avg\]\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)

    if not avg_match:
        return None

    total_time = float(avg_match.group(1))  # seconds
    rungen_time = float(avg_match.group(2))  # seconds
    merge_time = float(avg_match.group(3))  # seconds
    throughput = float(avg_match.group(4))  # M entries/s
    read_mb = float(avg_match.group(5))  # MB
    write_mb = float(avg_match.group(6))  # MB

    total_io_mb = read_mb + write_mb

    return {
        'threads': threads,
        'total_time': total_time,
        'rungen_time': rungen_time,
        'merge_time': merge_time,
        'throughput': throughput,
        'total_io_mb': total_io_mb
    }

# Parse Exp6 logs
exp6_files = glob.glob(os.path.join(LOG_DIR, 'Exp6_RunGen*_Merge*_Mem2GB.log'))
exp6_data = []

for f in exp6_files:
    parsed = parse_exp6_log(f)
    if parsed:
        exp6_data.append(parsed)

df_exp6 = pd.DataFrame(exp6_data).sort_values(['rungen_threads', 'merge_threads'])
print(f"Loaded {len(df_exp6)} Exp6 configurations")
print(f"RunGen threads: {sorted(df_exp6['rungen_threads'].unique())}")
print(f"Merge threads: {sorted(df_exp6['merge_threads'].unique())}")

# Parse Exp1 logs
exp1_files = glob.glob(os.path.join(LOG_DIR, 'Exp1_Thr*_Mem2GB.log'))
exp1_data = []

for f in exp1_files:
    parsed = parse_exp1_log(f)
    if parsed:
        exp1_data.append(parsed)

df_exp1 = pd.DataFrame(exp1_data).sort_values('threads')
print(f"\nLoaded {len(df_exp1)} Exp1 configurations")
print(f"Threads: {sorted(df_exp1['threads'].unique())}")

# Create combined heatmap data for Exp6 and Exp1
# Get all unique thread values
all_threads = sorted(set(df_exp6['rungen_threads'].unique()) |
                    set(df_exp6['merge_threads'].unique()) |
                    set(df_exp1['threads'].unique()))

print(f"\nAll thread values: {all_threads}")

# Create matrices: rows = RunGen threads, cols = Merge threads
n_rows = len(all_threads)
n_cols = len(all_threads)

total_time_matrix = np.full((n_rows, n_cols), np.nan)
total_io_matrix = np.full((n_rows, n_cols), np.nan)
throughput_matrix = np.full((n_rows, n_cols), np.nan)

# Fill Exp6 data (off-diagonal)
for _, row in df_exp6.iterrows():
    i = all_threads.index(row['rungen_threads'])
    j = all_threads.index(row['merge_threads'])
    total_time_matrix[i, j] = row['total_time']
    total_io_matrix[i, j] = row['total_io_mb']
    throughput_matrix[i, j] = row['throughput']

# Fill Exp1 data on the diagonal (where RunGen=Merge)
for _, row in df_exp1.iterrows():
    idx = all_threads.index(row['threads'])
    total_time_matrix[idx, idx] = row['total_time']
    total_io_matrix[idx, idx] = row['total_io_mb']
    throughput_matrix[idx, idx] = row['throughput']

# Convert to appropriate units
total_time_matrix_min = total_time_matrix / 60
total_io_matrix_gb = total_io_matrix / 1024

# Create row and column labels
row_labels = [str(t) for t in all_threads]
col_labels = [str(t) for t in all_threads]

# Helper function to create heatmap
def create_heatmap(data_matrix, title, cmap, unit, vmin=None, vmax=None,
                   format_str='.1f', invert=False):
    """Create a single heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(data_matrix)
    if vmax is None:
        vmax = np.nanmax(data_matrix)

    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto',
                   interpolation='nearest', origin='lower',
                   vmin=vmin, vmax=vmax)

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(data_matrix[i, j]):
                value = data_matrix[i, j]
                text_str = f'{value:{format_str}}'

                # Determine text color for readability
                # Normalize value to [0, 1]
                norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                if invert:
                    text_color = 'white' if norm_val < 0.5 else 'black'
                else:
                    text_color = 'white' if norm_val > 0.5 else 'black'

                ax.text(j, i, text_str, ha="center", va="center",
                       color=text_color, fontsize=9, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Merge Threads', fontweight='bold', fontsize=12)
    ax.set_ylabel('RunGen Threads', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=unit, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Add diagonal line to highlight Exp1 data (where RunGen=Merge)
    ax.plot([-0.5, n_cols - 0.5], [-0.5, n_rows - 0.5],
            color='black', linewidth=2, linestyle='--', alpha=0.5,
            label='Exp1 (RG=MG)')

    # Find and mark optimal value
    if invert:  # For throughput, find maximum
        opt_idx = np.unravel_index(np.nanargmax(data_matrix), data_matrix.shape)
        label = 'Best'
    else:  # For time and I/O, find minimum
        opt_idx = np.unravel_index(np.nanargmin(data_matrix), data_matrix.shape)
        label = 'Best'

    ax.plot(opt_idx[1], opt_idx[0], 'b*', markersize=25, markeredgewidth=2.5,
            markeredgecolor='blue', markerfacecolor='none', label=label)
    ax.legend(loc='upper right', fontsize=11)

    # Add grid for better readability
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    return fig, ax

# Create three separate heatmaps (absolute values)
print("\nCreating absolute value heatmaps...")

# 1. Total Time Heatmap
fig1, ax1 = create_heatmap(
    total_time_matrix_min,
    'GenSort: Exp6 vs Exp1 - Total Time Comparison',
    'RdYlGn',
    'Time (min)',
    format_str='.1f'
)
output_path1 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_total_time_heatmap.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path1}")
plt.close()

# 2. Total I/O Heatmap
fig2, ax2 = create_heatmap(
    total_io_matrix_gb,
    'GenSort: Exp6 vs Exp1 - Total I/O Comparison',
    'YlOrRd',
    'I/O (GB)',
    format_str='.0f'
)
output_path2 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_total_io_heatmap.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path2}")
plt.close()

# 3. Throughput Heatmap (higher is better, so green=good, red=bad)
fig3, ax3 = create_heatmap(
    throughput_matrix,
    'GenSort: Exp6 vs Exp1 - Throughput Comparison',
    'RdYlGn',  # Red (low/bad) -> Yellow -> Green (high/good)
    'Throughput (M entries/s)',
    format_str='.2f',
    invert=True  # Invert logic: find maximum instead of minimum
)
output_path3 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_throughput_heatmap.png')
plt.savefig(output_path3, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path3}")
plt.close()

# Create efficiency-based heatmaps (percentage of optimal)
print("\nCreating efficiency-based heatmaps...")

# Calculate efficiency matrices (percentage of optimal performance)
# For time and I/O: efficiency = optimal / actual * 100 (lower is better)
# For throughput: efficiency = actual / optimal * 100 (higher is better)

time_efficiency = (np.nanmin(total_time_matrix_min) / total_time_matrix_min) * 100
io_efficiency = (np.nanmin(total_io_matrix_gb) / total_io_matrix_gb) * 100
throughput_efficiency = (throughput_matrix / np.nanmax(throughput_matrix)) * 100

# 4. Time Efficiency Heatmap
fig4, ax4 = create_heatmap(
    time_efficiency,
    'GenSort: Exp6 vs Exp1 - Time Efficiency (% of Optimal)',
    'RdYlGn',
    'Efficiency (%)',
    vmin=0, vmax=100,
    format_str='.0f',
    invert=True  # Find maximum efficiency
)
output_path4 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_time_efficiency_heatmap.png')
plt.savefig(output_path4, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path4}")
plt.close()

# 5. I/O Efficiency Heatmap
fig5, ax5 = create_heatmap(
    io_efficiency,
    'GenSort: Exp6 vs Exp1 - I/O Efficiency (% of Optimal)',
    'RdYlGn',
    'Efficiency (%)',
    vmin=0, vmax=100,
    format_str='.0f',
    invert=True  # Find maximum efficiency
)
output_path5 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_io_efficiency_heatmap.png')
plt.savefig(output_path5, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path5}")
plt.close()

# 6. Throughput Efficiency Heatmap
fig6, ax6 = create_heatmap(
    throughput_efficiency,
    'GenSort: Exp6 vs Exp1 - Throughput Efficiency (% of Optimal)',
    'RdYlGn',
    'Efficiency (%)',
    vmin=0, vmax=100,
    format_str='.0f',
    invert=True  # Find maximum efficiency
)
output_path6 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_throughput_efficiency_heatmap.png')
plt.savefig(output_path6, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path6}")
plt.close()

# Create binned/discretized efficiency heatmaps
print("\nCreating binned efficiency heatmaps...")

from matplotlib.colors import BoundaryNorm, ListedColormap

def create_binned_heatmap(data_matrix, title, unit, format_str='.0f'):
    """Create a binned heatmap with discrete color tiers."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define performance tiers
    boundaries = [0, 50, 75, 90, 100]
    colors = ['#d62728', '#ff7f0e', '#ffff00', '#2ca02c']  # Red, Orange, Yellow, Green
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    im = ax.imshow(data_matrix, cmap=cmap, norm=norm, aspect='auto',
                   interpolation='nearest', origin='lower')

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(data_matrix[i, j]):
                value = data_matrix[i, j]
                text_str = f'{value:{format_str}}'

                # Determine text color based on tier
                if value >= 75:
                    text_color = 'black'
                else:
                    text_color = 'white'

                ax.text(j, i, text_str, ha="center", va="center",
                       color=text_color, fontsize=9, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Merge Threads', fontweight='bold', fontsize=12)
    ax.set_ylabel('RunGen Threads', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    # Add colorbar with tier labels
    cbar = plt.colorbar(im, ax=ax, label=unit, pad=0.02,
                       boundaries=boundaries, ticks=[25, 62.5, 82.5, 95])
    cbar.ax.set_yticklabels(['Poor\n(<50%)', 'Fair\n(50-75%)',
                             'Good\n(75-90%)', 'Excellent\n(90-100%)'],
                            fontsize=10)

    # Add diagonal line
    ax.plot([-0.5, n_cols - 0.5], [-0.5, n_rows - 0.5],
            color='black', linewidth=2, linestyle='--', alpha=0.5,
            label='Exp1 (RG=MG)')

    # Find and mark optimal
    opt_idx = np.unravel_index(np.nanargmax(data_matrix), data_matrix.shape)
    ax.plot(opt_idx[1], opt_idx[0], 'b*', markersize=25, markeredgewidth=2.5,
            markeredgecolor='blue', markerfacecolor='none', label='Best')
    ax.legend(loc='upper right', fontsize=11)

    # Add grid
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    return fig, ax

# 7. Binned Time Efficiency Heatmap
fig7, ax7 = create_binned_heatmap(
    time_efficiency,
    'GenSort: Exp6 vs Exp1 - Time Efficiency Binned (Performance Tiers)',
    'Efficiency Tier',
    format_str='.0f'
)
output_path7 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_time_efficiency_binned.png')
plt.savefig(output_path7, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path7}")
plt.close()

# 8. Binned I/O Efficiency Heatmap
fig8, ax8 = create_binned_heatmap(
    io_efficiency,
    'GenSort: Exp6 vs Exp1 - I/O Efficiency Binned (Performance Tiers)',
    'Efficiency Tier',
    format_str='.0f'
)
output_path8 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_io_efficiency_binned.png')
plt.savefig(output_path8, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path8}")
plt.close()

# 9. Binned Throughput Efficiency Heatmap
fig9, ax9 = create_binned_heatmap(
    throughput_efficiency,
    'GenSort: Exp6 vs Exp1 - Throughput Efficiency Binned (Performance Tiers)',
    'Efficiency Tier',
    format_str='.0f'
)
output_path9 = os.path.join(OUTPUT_DIR, 'gensort_exp6_exp1_throughput_efficiency_binned.png')
plt.savefig(output_path9, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path9}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Overall best/worst
best_time_idx = np.unravel_index(np.nanargmin(total_time_matrix_min), total_time_matrix_min.shape)
best_time = total_time_matrix_min[best_time_idx]
worst_time = np.nanmax(total_time_matrix_min)

best_io_idx = np.unravel_index(np.nanargmin(total_io_matrix_gb), total_io_matrix_gb.shape)
best_io = total_io_matrix_gb[best_io_idx]
worst_io = np.nanmax(total_io_matrix_gb)

best_throughput_idx = np.unravel_index(np.nanargmax(throughput_matrix), throughput_matrix.shape)
best_throughput = throughput_matrix[best_throughput_idx]
worst_throughput = np.nanmin(throughput_matrix)

# Format configuration string
def get_config_str(idx):
    row_idx, col_idx = idx
    rg = all_threads[row_idx]
    mg = all_threads[col_idx]
    if row_idx == col_idx:  # Diagonal = Exp1
        return f"Exp1 (Threads={all_threads[row_idx]})"
    else:  # Off-diagonal = Exp6
        return f"Exp6 (RunGen={rg}, Merge={mg})"

print(f"\n📊 Overall Best Configurations:")
print(f"  Best Time: {best_time:.1f} min - {get_config_str(best_time_idx)}")
print(f"  Best I/O: {best_io:.0f} GB - {get_config_str(best_io_idx)}")
print(f"  Best Throughput: {best_throughput:.2f} M entries/s - {get_config_str(best_throughput_idx)}")

print(f"\n📊 Performance Ranges:")
print(f"  Time: {best_time:.1f} - {worst_time:.1f} min (range: {worst_time/best_time:.2f}x)")
print(f"  I/O: {best_io:.0f} - {worst_io:.0f} GB (range: {worst_io/best_io:.2f}x)")
print(f"  Throughput: {worst_throughput:.2f} - {best_throughput:.2f} M entries/s (range: {best_throughput/worst_throughput:.2f}x)")

# Extract diagonal (Exp1) and off-diagonal (Exp6) data
exp1_time_vals = []
exp1_io_vals = []
exp1_throughput_vals = []
exp6_time_vals = []
exp6_io_vals = []
exp6_throughput_vals = []

for i in range(n_rows):
    for j in range(n_cols):
        if not np.isnan(total_time_matrix_min[i, j]):
            if i == j:  # Diagonal = Exp1
                exp1_time_vals.append(total_time_matrix_min[i, j])
                exp1_io_vals.append(total_io_matrix_gb[i, j])
                exp1_throughput_vals.append(throughput_matrix[i, j])
            else:  # Off-diagonal = Exp6
                exp6_time_vals.append(total_time_matrix_min[i, j])
                exp6_io_vals.append(total_io_matrix_gb[i, j])
                exp6_throughput_vals.append(throughput_matrix[i, j])

print(f"\n📊 Exp1 (Diagonal: RunGen=Merge) Ranges:")
print(f"  Time: {np.min(exp1_time_vals):.1f} - {np.max(exp1_time_vals):.1f} min (range: {np.max(exp1_time_vals)/np.min(exp1_time_vals):.2f}x)")
print(f"  I/O: {np.min(exp1_io_vals):.0f} - {np.max(exp1_io_vals):.0f} GB (range: {np.max(exp1_io_vals)/np.min(exp1_io_vals):.2f}x)")
print(f"  Throughput: {np.min(exp1_throughput_vals):.2f} - {np.max(exp1_throughput_vals):.2f} M entries/s (range: {np.max(exp1_throughput_vals)/np.min(exp1_throughput_vals):.2f}x)")

print(f"\n📊 Exp6 (Off-diagonal: Independent RunGen/Merge) Ranges:")
print(f"  Time: {np.min(exp6_time_vals):.1f} - {np.max(exp6_time_vals):.1f} min (range: {np.max(exp6_time_vals)/np.min(exp6_time_vals):.2f}x)")
print(f"  I/O: {np.min(exp6_io_vals):.0f} - {np.max(exp6_io_vals):.0f} GB (range: {np.max(exp6_io_vals)/np.min(exp6_io_vals):.2f}x)")
print(f"  Throughput: {np.min(exp6_throughput_vals):.2f} - {np.max(exp6_throughput_vals):.2f} M entries/s (range: {np.max(exp6_throughput_vals)/np.min(exp6_throughput_vals):.2f}x)")

print("\n" + "="*80)
