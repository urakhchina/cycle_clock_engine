#!/usr/bin/env python3
"""
Helix-mode parameter sweep for the three canonical two-particle behaviors.

Sweeps over all 3 presets × multiple runs × 60 steps.
Exports CSV with trajectory, savings, inter-emperor distance.
Generates figures comparing the three behaviors.
"""

import numpy as np
import csv
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.helix_game import HelixGame, PRESETS


def run_single(helix_game, preset_name, n_steps, run_id):
    """Run a single helix simulation and collect per-step data."""
    np.random.seed(run_id)

    helix_game.init_from_preset(preset_name)
    rows = []

    for step in range(n_steps):
        entry = helix_game.step()

        for ed in entry['emperors']:
            row = {
                'run_id': run_id,
                'preset': preset_name,
                'step': step + 1,
                'emperor': ed['emperor_id'],
                'from_x': ed['from'][0],
                'from_y': ed['from'][1],
                'from_z': ed['from'][2],
                'to_x': ed['to'][0],
                'to_y': ed['to'][1],
                'to_z': ed['to'][2],
                'savings': ed['savings'],
                'probability': round(ed.get('probability', 0), 6),
                'n_options': ed.get('n_options', 0),
                'best_savings': ed.get('best_savings', 0),
                'mean_savings': round(ed.get('mean_savings', 0), 1),
            }
            sn = ed.get('snapshot', {})
            row['chirality'] = sn.get('chirality', '?')
            row['exponent'] = sn.get('exponent', 0)
            rows.append(row)

        if 'interaction' in entry:
            inter = entry['interaction']
            for r in rows[-2:]:
                r['distance'] = round(inter['distance'], 4)
                r['chirality_match'] = inter['chirality_match']

    return rows


def generate_figures(all_rows, output_dir):
    """Generate comparison figures for the three behaviors."""

    presets = list(PRESETS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for pi, preset_name in enumerate(presets):
        ax = axes[pi]
        preset_rows = [r for r in all_rows
                       if r['preset'] == preset_name
                       and r['emperor'] == 0
                       and 'distance' in r]

        # Group by run
        runs = {}
        for r in preset_rows:
            rid = r['run_id']
            if rid not in runs:
                runs[rid] = {'steps': [], 'distances': []}
            runs[rid]['steps'].append(r['step'])
            runs[rid]['distances'].append(r['distance'])

        for rid, data in runs.items():
            ax.plot(data['steps'], data['distances'], alpha=0.3, linewidth=0.8)

        # Mean across runs
        if runs:
            max_step = max(max(d['steps']) for d in runs.values())
            mean_dist = []
            for s in range(1, max_step + 1):
                vals = [d['distances'][d['steps'].index(s)]
                        for d in runs.values() if s in d['steps']]
                mean_dist.append(np.mean(vals) if vals else 0)
            ax.plot(range(1, max_step + 1), mean_dist, 'k-', linewidth=2,
                    label='mean')

        info = PRESETS[preset_name]
        ax.set_title(f"{preset_name.replace('_', ' ').title()}\n"
                     f"(exp={info['exponent']})", fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Inter-emperor distance')
        ax.legend(fontsize=8)

    plt.suptitle('Helix Mode: Three Canonical Two-Particle Behaviors',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'helix_behaviors.png'), dpi=150)
    plt.close()
    print(f"  Saved helix_behaviors.png")

    # Savings distribution figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for pi, preset_name in enumerate(presets):
        ax = axes[pi]
        savings = [r['savings'] for r in all_rows
                   if r['preset'] == preset_name and r['emperor'] == 0]
        if savings:
            ax.hist(savings, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(preset_name.replace('_', ' ').title())
        ax.set_xlabel('Savings')
        ax.set_ylabel('Count')

    plt.suptitle('Helix Mode: Savings Distributions', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'helix_savings.png'), dpi=150)
    plt.close()
    print(f"  Saved helix_savings.png")


def main():
    print("=" * 60)
    print("HELIX MODE PARAMETER SWEEP")
    print("=" * 60)

    # Configuration
    n_runs = 10
    n_steps = 60
    empire_radius = 3000  # Start smaller for faster iteration

    # Build game (one-time cost)
    t0 = time.time()
    helix_game = HelixGame(empire_radius=empire_radius, verbose=True)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.1f}s")

    presets = list(PRESETS.keys())
    total_runs = len(presets) * n_runs
    print(f"\nSweep: {len(presets)} presets × {n_runs} runs = {total_runs} total")
    print(f"Steps per run: {n_steps}")
    print(f"Presets: {presets}")

    all_rows = []
    run_count = 0
    t_start = time.time()

    for preset_name in presets:
        for run in range(n_runs):
            run_id = run_count
            rows = run_single(helix_game, preset_name, n_steps, run_id)
            all_rows.extend(rows)
            run_count += 1

            if run_count % 5 == 0:
                elapsed = time.time() - t_start
                rate = run_count / elapsed if elapsed > 0 else 0
                remaining = (total_runs - run_count) / rate if rate > 0 else 0
                print(f"  Run {run_count}/{total_runs} "
                      f"({preset_name}, #{run}) "
                      f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t_start
    print(f"\nCompleted {run_count} runs in {elapsed:.1f}s")

    # Write CSV
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'helix_sweep.csv')

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {csv_path}")

    # Generate figures
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating figures...")
    generate_figures(all_rows, fig_dir)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for preset_name in presets:
        preset_rows = [r for r in all_rows
                       if r['preset'] == preset_name and r['emperor'] == 0]
        if not preset_rows:
            continue
        savings = [r['savings'] for r in preset_rows]
        distances = [r.get('distance', 0) for r in preset_rows if 'distance' in r]
        print(f"\n  {preset_name}:")
        print(f"    Mean savings: {np.mean(savings):.1f} ± {np.std(savings):.1f}")
        if distances:
            print(f"    Mean distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}")


if __name__ == '__main__':
    main()
