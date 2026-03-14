#!/usr/bin/env python3
"""
Generate publication-quality figures from the parameter sweep data.

Produces:
  1. Efficiency vs exponent curve (the PEL transition)
  2. Savings distribution histograms per exponent
  3. Two-clock distance time series
  4. Pattern classification pie chart
  5. Segment overlap vs distance scatter
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'probability_sweep.csv')
rows = list(csv.DictReader(open(data_path)))
out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'figures')
os.makedirs(out_dir, exist_ok=True)

exponents = sorted(set(int(r['exponent']) for r in rows))

# ================================================================
# Figure 1: Efficiency vs Exponent (the PEL transition)
# ================================================================
print("Figure 1: Efficiency vs Exponent...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mean_eff = []
std_eff = []
pct_best = []
for exp in exponents:
    effs = [float(r['efficiency']) for r in rows if int(r['exponent']) == exp and r['clock'] == '0']
    mean_eff.append(np.mean(effs))
    std_eff.append(np.std(effs))
    ranks = [int(r['rank']) for r in rows if int(r['exponent']) == exp and r['clock'] == '0']
    pct_best.append(sum(1 for r in ranks if r == 1) / len(ranks) * 100)

ax1.errorbar(exponents, mean_eff, yerr=std_eff, fmt='o-', color='#2166ac',
             capsize=4, markersize=6, linewidth=2)
ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(28, color='#d6604d', linestyle=':', alpha=0.7, label='Original Mathematica (exp=28)')
ax1.set_xlabel('Savings Exponent (PEL strength)')
ax1.set_ylabel('Mean Efficiency (savings / best)')
ax1.set_title('Principle of Efficient Language Transition')
ax1.set_ylim(0.4, 1.05)
ax1.legend()
ax1.grid(alpha=0.3)

ax2.bar(exponents, pct_best, color='#4393c3', width=2)
ax2.axvline(28, color='#d6604d', linestyle=':', alpha=0.7, label='exp=28')
ax2.set_xlabel('Savings Exponent')
ax2.set_ylabel('% Steps Choosing Optimal Move')
ax2.set_title('Optimal Move Selection Rate')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig1_pel_transition.png'))
plt.close()

# ================================================================
# Figure 2: Savings distributions per exponent
# ================================================================
print("Figure 2: Savings distributions...")

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for idx, exp in enumerate(exponents):
    ax = axes[idx // 3][idx % 3]
    savings = [int(r['savings']) for r in rows if int(r['exponent']) == exp and r['clock'] == '0']
    ax.hist(savings, bins=30, color='#4393c3', alpha=0.7, edgecolor='white')
    ax.set_title(f'exp = {exp}', fontweight='bold')
    ax.set_xlabel('Segment Savings')
    ax.set_ylabel('Count')
    ax.axvline(np.mean(savings), color='#d6604d', linewidth=2, linestyle='--',
               label=f'mean={np.mean(savings):.0f}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Savings Distribution by PEL Exponent\n(segment-based, 779 vertices, 3789 segments)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig2_savings_distributions.png'))
plt.close()

# ================================================================
# Figure 3: Two-clock distance time series
# ================================================================
print("Figure 3: Distance time series...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
configs = ['center_vs_far', 'center_vs_mid', 'far_vs_mid']
config_titles = ['Center vs Far', 'Center vs Mid', 'Far vs Mid']

for ci, (config, title) in enumerate(zip(configs, config_titles)):
    ax = axes[ci]
    for exp in [3, 10, 28]:
        config_rows = [r for r in rows if r['config'] == config and int(r['exponent']) == exp
                      and r['clock'] == '0' and r['distance_3d']]
        # Average across runs for each step
        step_dists = defaultdict(list)
        for r in config_rows:
            step_dists[int(r['step'])].append(float(r['distance_3d']))
        steps = sorted(step_dists.keys())
        means = [np.mean(step_dists[s]) for s in steps]
        ax.plot(steps, means, label=f'exp={exp}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('3D Distance')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Two-Clock Distance Over Time (averaged over 10 runs)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig3_distance_timeseries.png'))
plt.close()

# ================================================================
# Figure 4: Pattern classification
# ================================================================
print("Figure 4: Pattern classification...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ei, exp in enumerate([3, 10, 28]):
    ax = axes[ei]
    runs_data = defaultdict(list)
    for r in rows:
        if int(r['exponent']) == exp and r['clock'] == '0':
            runs_data[r['run_id']].append(int(r['to_vertex']))

    patterns = {'Oscillation\n(period ≤ 4)': 0, 'Short cycle\n(period 5-10)': 0, 'Wandering': 0}
    for run_id, verts in runs_data.items():
        unique = len(set(verts))
        if unique <= 4:
            patterns['Oscillation\n(period ≤ 4)'] += 1
        elif unique <= 10:
            patterns['Short cycle\n(period 5-10)'] += 1
        else:
            patterns['Wandering'] += 1

    colors_pie = ['#2166ac', '#4393c3', '#d6604d']
    counts = list(patterns.values())
    labels = list(patterns.keys())
    if sum(counts) > 0:
        ax.pie(counts, labels=labels, colors=colors_pie, autopct='%1.0f%%',
               textprops={'fontsize': 9})
    ax.set_title(f'Exponent = {exp}', fontweight='bold')

plt.suptitle('Emergent Movement Patterns by PEL Strength',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig4_pattern_classification.png'))
plt.close()

# ================================================================
# Figure 5: Segment overlap vs distance scatter
# ================================================================
print("Figure 5: Overlap vs distance...")

fig, ax = plt.subplots(figsize=(8, 6))
for exp in [3, 10, 28]:
    overlap_rows = [r for r in rows if int(r['exponent']) == exp
                   and r['clock'] == '0' and r.get('segment_overlap') and r.get('distance_3d')]
    if not overlap_rows:
        continue
    overlaps = [int(r['segment_overlap']) for r in overlap_rows]
    dists = [float(r['distance_3d']) for r in overlap_rows]
    ax.scatter(dists, overlaps, alpha=0.2, s=10, label=f'exp={exp}')

ax.set_xlabel('3D Distance Between Clocks')
ax.set_ylabel('Segment Empire Overlap')
ax.set_title('Empire Overlap vs Spatial Distance\n(non-local interaction strength)')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(out_dir, 'fig5_overlap_vs_distance.png'))
plt.close()

print(f"\nAll figures saved to {out_dir}/")
print("Files:")
for f in sorted(os.listdir(out_dir)):
    print(f"  {f}")
