#!/usr/bin/env python3
"""
Large-scale parameter sweep for cycle clock probability distributions.

Runs multiple simulations across different:
  - Savings exponents (PEL strength)
  - Initial positions
  - Random seeds

Outputs CSV data and summary statistics for paper supplementary material.
"""

import numpy as np
import csv
import sys
import os
import time
from itertools import product as iprod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'CCT-StandardModel'))

from engine.game import Game
from engine.cycle_clock import ISVParams


def run_single(game, fig_v0, fig_v1, seed0, seed1, exponent, n_steps, run_id):
    """Run a single simulation and collect per-step data."""
    np.random.seed(run_id)

    # Reset game state
    game.clocks = []
    game.step_log = []
    game.step_count = 0

    c0 = game.add_clock(fig_vertex=fig_v0, coxeter_seed=seed0,
                        isv=ISVParams(savings_exponent=exponent))
    c1 = game.add_clock(fig_vertex=fig_v1, coxeter_seed=seed1,
                        isv=ISVParams(savings_exponent=exponent))

    rows = []
    for step in range(n_steps):
        entry = game.step()

        for ci, cd in enumerate(entry['clocks']):
            rows.append({
                'run_id': run_id,
                'step': step + 1,
                'clock': ci,
                'exponent': exponent,
                'from_vertex': cd['from'],
                'to_vertex': cd['to'],
                'savings': cd['savings'],
                'rank': cd['rank'],
                'n_options': cd['n_options'],
                'probability': round(cd['probability'], 6),
                'best_savings': cd['best_savings'],
                'worst_savings': cd.get('worst_savings', 0),
                'mean_savings': round(cd['mean_savings'], 1),
                'efficiency': round(cd['savings'] / max(cd['best_savings'], 1), 4),
                'chirality': cd['snapshot']['chirality'],
                'generation': cd['snapshot']['generation'],
                'coxeter_phase': cd['snapshot']['coxeter_phase'],
                'fiber': cd['snapshot']['fiber'],
            })

        if 'interaction' in entry:
            inter = entry['interaction']
            # Add interaction data to the last two rows
            for r in rows[-2:]:
                r['segment_overlap'] = inter['segment_overlap']
                r['distance_3d'] = round(inter['distance_3d'], 4)
                r['chirality_match'] = inter['chirality_match']

    return rows


def main():
    print("=" * 60)
    print("CYCLE CLOCK PARAMETER SWEEP")
    print("=" * 60)

    # Build game (one-time cost)
    t0 = time.time()
    game = Game(max_norm_sq=8, verbose=True)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.1f}s")

    # Configuration
    origin = game.fig.origin_idx
    dists = np.linalg.norm(
        game.fig.pos_3d - game.fig.pos_3d[origin], axis=1)
    candidates = [(i, dists[i]) for i in range(game.fig.n_vertices)
                  if game.empire.vertex_empire_sizes[i] > 200
                  and game.segs.degrees[i] >= 6]
    candidates.sort(key=lambda x: -x[1])
    far_vertex = candidates[0][0]
    mid_vertex = candidates[len(candidates)//2][0]

    # Sweep parameters
    exponents = [1, 3, 5, 8, 10, 15, 20, 28, 40]
    n_runs_per_config = 10
    n_steps = 60  # 2 full Coxeter cycles
    start_configs = [
        (origin, far_vertex, 0, 100, "center_vs_far"),
        (origin, mid_vertex, 0, 50, "center_vs_mid"),
        (far_vertex, mid_vertex, 20, 80, "far_vs_mid"),
    ]

    total_runs = len(exponents) * n_runs_per_config * len(start_configs)
    print(f"\nSweep: {len(exponents)} exponents × {n_runs_per_config} runs × "
          f"{len(start_configs)} configs = {total_runs} total runs")
    print(f"Steps per run: {n_steps}")
    print(f"Exponents: {exponents}")

    all_rows = []
    run_count = 0
    t_start = time.time()

    for exp in exponents:
        for v0, v1, s0, s1, config_name in start_configs:
            for run in range(n_runs_per_config):
                run_id = run_count
                rows = run_single(game, v0, v1, s0, s1, exp, n_steps, run_id)
                for r in rows:
                    r['config'] = config_name
                all_rows.extend(rows)
                run_count += 1

                if run_count % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = run_count / elapsed
                    remaining = (total_runs - run_count) / rate
                    print(f"  Run {run_count}/{total_runs} "
                          f"(exp={exp}, {config_name}, #{run}) "
                          f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t_start
    print(f"\nCompleted {run_count} runs in {elapsed:.1f}s "
          f"({run_count/elapsed:.1f} runs/s)")

    # Write CSV
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'probability_sweep.csv')

    fieldnames = list(all_rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {csv_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    for exp in exponents:
        exp_rows = [r for r in all_rows if r['exponent'] == exp and r['clock'] == 0]
        if not exp_rows:
            continue
        efficiencies = [r['efficiency'] for r in exp_rows]
        savings = [r['savings'] for r in exp_rows]
        ranks = [r['rank'] for r in exp_rows]
        chose_best = sum(1 for r in ranks if r == 1)

        print(f"\n  Exponent {exp}:")
        print(f"    Mean efficiency: {np.mean(efficiencies):.3f}")
        print(f"    Chose rank 1 (best): {chose_best}/{len(ranks)} ({chose_best/len(ranks)*100:.0f}%)")
        print(f"    Mean savings: {np.mean(savings):.0f}")
        print(f"    Savings std: {np.std(savings):.0f}")

    # Interaction summary
    print(f"\n  --- Two-clock interaction ---")
    for config_name in set(r['config'] for r in all_rows):
        config_rows = [r for r in all_rows if r['config'] == config_name
                      and r['clock'] == 0 and 'segment_overlap' in r]
        if not config_rows:
            continue
        overlaps = [r['segment_overlap'] for r in config_rows]
        distances = [r['distance_3d'] for r in config_rows]
        print(f"    {config_name}:")
        print(f"      Segment overlap: {np.mean(overlaps):.0f} ± {np.std(overlaps):.0f}")
        print(f"      Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}")


if __name__ == '__main__':
    main()
