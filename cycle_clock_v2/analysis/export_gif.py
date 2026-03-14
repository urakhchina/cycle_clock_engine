#!/usr/bin/env python3
"""
Export animated GIFs of the cycle clock simulations.

Generates:
  1. Segment-hop mode: empire coloring on the FIG, two clocks walking
  2. Helix mode: 3 canonical behaviors side-by-side
  3. Helix mode: single preset, detailed view with savings sparkline

Usage:
  python analysis/export_gif.py                # all GIFs
  python analysis/export_gif.py --segment      # segment mode only
  python analysis/export_gif.py --helix        # helix mode only
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'CCT-StandardModel'))

BG = '#0a0a1a'
GOLD = '#f0a030'
PURPLE = '#a060f0'
WHITE = '#e0e0e0'
DIM = '#334466'

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def export_helix_comparison_gif(n_steps=80, fps=6):
    """GIF: 3 canonical helix behaviors side by side."""
    from engine.helix_game import HelixGame, PRESETS

    print("Generating helix comparison GIF...")
    presets = ['teeter_totter', 'expansion_contraction', 'cycling_chasing']
    preset_labels = ['Teeter-Totter (L/R, exp=28)',
                     'Expansion-Contraction (L/L, exp=17)',
                     'Cycling-Chasing (L/L, exp=7)']

    # Run all 3 simulations first
    all_data = {}
    for preset_name in presets:
        game = HelixGame(empire_radius=8, verbose=False)
        game.init_from_preset(preset_name)

        positions = {0: [game.emperors[0].position.copy()],
                     1: [game.emperors[1].position.copy()]}
        distances = []
        savings = {0: [], 1: []}

        for _ in range(n_steps):
            step = game.step()
            for ei in range(2):
                positions[ei].append(game.emperors[ei].position.copy())
            if 'interaction' in step:
                distances.append(step['interaction']['distance'])
            for ed in step['emperors']:
                savings[ed['emperor_id']].append(ed['savings'])

        all_data[preset_name] = {
            'positions': positions,
            'distances': distances,
            'savings': savings,
        }

    # Create figure
    fig = plt.figure(figsize=(18, 10), facecolor=BG)

    # Top row: 3D trajectory, Bottom row: distance + savings
    axes_3d = []
    axes_dist = []
    axes_sav = []
    for col in range(3):
        ax3 = fig.add_subplot(2, 3, col + 1, projection='3d', facecolor=BG)
        ax3.set_title(preset_labels[col], color=WHITE, fontsize=10, pad=10)
        ax3.tick_params(colors='#555555', labelsize=7)
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor('#222233')
        ax3.yaxis.pane.set_edgecolor('#222233')
        ax3.zaxis.pane.set_edgecolor('#222233')
        ax3.grid(True, alpha=0.1)
        axes_3d.append(ax3)

        ax_d = fig.add_subplot(2, 6, 6 + col * 2 + 1, facecolor=BG)
        ax_d.set_ylabel('Distance', color='#888', fontsize=8)
        ax_d.tick_params(colors='#555', labelsize=7)
        ax_d.set_xlim(0, n_steps)
        axes_dist.append(ax_d)

        ax_s = fig.add_subplot(2, 6, 6 + col * 2 + 2, facecolor=BG)
        ax_s.set_ylabel('Savings', color='#888', fontsize=8)
        ax_s.tick_params(colors='#555', labelsize=7)
        ax_s.set_xlim(0, n_steps)
        axes_sav.append(ax_s)

    fig.suptitle('Pentagonal Helix Mode — Three Canonical Behaviors',
                 color=WHITE, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Compute global ranges for 3D axes
    all_pos = []
    for d in all_data.values():
        for ei in range(2):
            all_pos.extend(d['positions'][ei])
    all_pos = np.array(all_pos)
    pad = 2
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)
    zlim = (all_pos[:, 2].min() - pad, all_pos[:, 2].max() + pad)

    max_dist = max(max(d['distances']) for d in all_data.values() if d['distances'])
    max_sav = max(max(max(d['savings'][0]), max(d['savings'][1]))
                  for d in all_data.values() if d['savings'][0])

    for ax in axes_3d:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
    for ax in axes_dist:
        ax.set_ylim(0, max_dist * 1.1)
    for ax in axes_sav:
        ax.set_ylim(0, max(max_sav * 1.1, 1))

    def update(frame):
        for col, preset_name in enumerate(presets):
            data = all_data[preset_name]

            # 3D trajectory
            ax3 = axes_3d[col]
            ax3.cla()
            ax3.set_title(preset_labels[col], color=WHITE, fontsize=10, pad=5)
            ax3.set_xlim(*xlim)
            ax3.set_ylim(*ylim)
            ax3.set_zlim(*zlim)
            ax3.tick_params(colors='#555555', labelsize=6)
            ax3.xaxis.pane.fill = False
            ax3.yaxis.pane.fill = False
            ax3.zaxis.pane.fill = False
            ax3.grid(True, alpha=0.1)

            for ei, color in [(0, GOLD), (1, PURPLE)]:
                pos = np.array(data['positions'][ei][:frame + 2])
                if len(pos) >= 2:
                    ax3.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                             '-', color=color, alpha=0.5, linewidth=1)
                if len(pos) > 0:
                    ax3.scatter(*pos[-1], color=color, s=40, zorder=5,
                                edgecolors='white', linewidth=0.5)

            # Distance
            ax_d = axes_dist[col]
            ax_d.cla()
            ax_d.set_facecolor(BG)
            ax_d.set_xlim(0, n_steps)
            ax_d.set_ylim(0, max_dist * 1.1)
            ax_d.tick_params(colors='#555', labelsize=6)
            d_slice = data['distances'][:frame + 1]
            if d_slice:
                ax_d.plot(range(1, len(d_slice) + 1), d_slice,
                          color='#88bbff', linewidth=1.5)
                ax_d.fill_between(range(1, len(d_slice) + 1), d_slice,
                                  alpha=0.1, color='#88bbff')

            # Savings
            ax_s = axes_sav[col]
            ax_s.cla()
            ax_s.set_facecolor(BG)
            ax_s.set_xlim(0, n_steps)
            ax_s.set_ylim(0, max(max_sav * 1.1, 1))
            ax_s.tick_params(colors='#555', labelsize=6)
            for ei, color in [(0, GOLD), (1, PURPLE)]:
                s_slice = data['savings'][ei][:frame + 1]
                if s_slice:
                    ax_s.plot(range(1, len(s_slice) + 1), s_slice,
                              color=color, linewidth=1, alpha=0.8)

        return []

    anim = animation.FuncAnimation(fig, update, frames=n_steps,
                                    interval=1000 // fps, blit=False)
    out_path = os.path.join(OUT_DIR, 'helix_comparison.gif')
    print(f"  Saving {n_steps} frames at {fps} fps...")
    anim.save(out_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print(f"  saved -> {out_path}")


def export_segment_gif(n_steps=60, fps=4):
    """GIF: Segment-hop mode with empire overlap visualization."""
    from engine.game import Game
    from engine.cycle_clock import ISVParams

    print("Generating segment-hop GIF...")
    game = Game(verbose=True)

    origin = game.fig.origin_idx
    dists = np.linalg.norm(
        game.fig.pos_3d - game.fig.pos_3d[origin], axis=1)
    candidates = [(i, dists[i]) for i in range(game.fig.n_vertices)
                  if game.empire.vertex_empire_sizes[i] > 120
                  and game.segs.degrees[i] >= 6]
    candidates.sort(key=lambda x: -x[1])
    far_v = candidates[0][0]

    game.add_clock(fig_vertex=origin, coxeter_seed=0,
                   isv=ISVParams(savings_exponent=28.0))
    game.add_clock(fig_vertex=far_v, coxeter_seed=100,
                   isv=ISVParams(savings_exponent=28.0))

    pos3d = game.fig.pos_3d
    segs = game.segs.segments

    # Prerun simulation
    history = []
    for _ in range(n_steps):
        entry = game.step()
        c0v = game.clocks[0].vertex
        c1v = game.clocks[1].vertex
        emp0 = game.empire.segment_empire[c0v]
        emp1 = game.empire.segment_empire[c1v]
        overlap = emp0 & emp1
        history.append({
            'c0': c0v, 'c1': c1v,
            'emp0': emp0, 'emp1': emp1, 'overlap': overlap,
            'savings0': entry['clocks'][0]['savings'],
            'savings1': entry['clocks'][1]['savings'],
            'distance': entry['interaction']['distance_3d'],
            'seg_overlap': entry['interaction']['segment_overlap'],
        })

    # Create figure
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    ax3d = fig.add_subplot(121, projection='3d', facecolor=BG)
    ax_metrics = fig.add_subplot(222, facecolor=BG)
    ax_savings = fig.add_subplot(224, facecolor=BG)

    ax3d.tick_params(colors='#555', labelsize=6)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.grid(True, alpha=0.1)

    fig.suptitle('Segment-Hop Mode — Empire Dynamics on 779-vertex FIG',
                 color=WHITE, fontsize=13, fontweight='bold')

    def update(frame):
        h = history[frame]
        ax3d.cla()
        ax3d.set_facecolor(BG)
        ax3d.tick_params(colors='#555', labelsize=6)
        ax3d.grid(True, alpha=0.1)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

        # Draw all segments dim
        for a, b in segs:
            pa, pb = pos3d[a], pos3d[b]
            ax3d.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                      color='#111122', linewidth=0.3, alpha=0.3)

        # Draw empire segments
        for si in list(h['emp0'] - h['overlap'])[:200]:
            a, b = segs[si]
            pa, pb = pos3d[a], pos3d[b]
            ax3d.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                      color=GOLD, linewidth=0.6, alpha=0.4)

        for si in list(h['emp1'] - h['overlap'])[:200]:
            a, b = segs[si]
            pa, pb = pos3d[a], pos3d[b]
            ax3d.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                      color=PURPLE, linewidth=0.6, alpha=0.4)

        for si in list(h['overlap'])[:200]:
            a, b = segs[si]
            pa, pb = pos3d[a], pos3d[b]
            ax3d.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                      color='white', linewidth=0.8, alpha=0.6)

        # Emperor positions
        p0 = pos3d[h['c0']]
        p1 = pos3d[h['c1']]
        ax3d.scatter(*p0, color=GOLD, s=80, zorder=10, edgecolors='white')
        ax3d.scatter(*p1, color=PURPLE, s=80, zorder=10, edgecolors='white')

        ax3d.set_title(f'Step {frame + 1}  |  overlap={h["seg_overlap"]}',
                       color=WHITE, fontsize=10)

        # Distance + overlap metrics
        ax_metrics.cla()
        ax_metrics.set_facecolor(BG)
        ax_metrics.tick_params(colors='#555', labelsize=7)
        steps = range(1, frame + 2)
        dists = [history[i]['distance'] for i in range(frame + 1)]
        overlaps = [history[i]['seg_overlap'] for i in range(frame + 1)]
        ax_metrics.plot(steps, dists, color='#88bbff', linewidth=1.5, label='distance')
        ax_m2 = ax_metrics.twinx()
        ax_m2.plot(steps, overlaps, color='white', linewidth=1, alpha=0.6, label='overlap')
        ax_m2.tick_params(colors='#555', labelsize=7)
        ax_metrics.set_ylabel('Distance', color='#888', fontsize=8)
        ax_m2.set_ylabel('Seg Overlap', color='#888', fontsize=8)
        ax_metrics.set_xlim(0, n_steps)
        ax_metrics.legend(loc='upper left', fontsize=7)

        # Savings
        ax_savings.cla()
        ax_savings.set_facecolor(BG)
        ax_savings.tick_params(colors='#555', labelsize=7)
        s0 = [history[i]['savings0'] for i in range(frame + 1)]
        s1 = [history[i]['savings1'] for i in range(frame + 1)]
        ax_savings.plot(steps, s0, color=GOLD, linewidth=1.5, label='C0')
        ax_savings.plot(steps, s1, color=PURPLE, linewidth=1.5, label='C1')
        ax_savings.set_ylabel('Savings', color='#888', fontsize=8)
        ax_savings.set_xlabel('Step', color='#888', fontsize=8)
        ax_savings.set_xlim(0, n_steps)
        ax_savings.legend(loc='upper right', fontsize=7)

        return []

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    anim = animation.FuncAnimation(fig, update, frames=n_steps,
                                    interval=1000 // fps, blit=False)
    out_path = os.path.join(OUT_DIR, 'segment_empire.gif')
    print(f"  Saving {n_steps} frames at {fps} fps...")
    anim.save(out_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--helix', action='store_true')
    args = parser.parse_args()

    do_all = not args.segment and not args.helix

    t0 = time.time()

    if do_all or args.helix:
        export_helix_comparison_gif(n_steps=80, fps=6)

    if do_all or args.segment:
        export_segment_gif(n_steps=60, fps=4)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
