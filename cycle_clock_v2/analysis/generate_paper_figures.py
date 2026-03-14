#!/usr/bin/env python3
"""
Generate publication-quality figures for the "Kinematic Spinors" paper.

Produces:
  Figure A (fig5_coxeter_cycle.png)   — Coxeter cycle polar diagram with 60-step overlay
  Figure B (fig6_behavior_comparison.png) — Side-by-side helix behavior comparison
  Figure C (fig7_savings_heatmap.png) — Savings heatmap on FIG vertices
"""

import sys
import os
import numpy as np

# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
OUT_DIR = os.path.join(ROOT_DIR, 'data', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Engine imports
sys.path.insert(0, ROOT_DIR)
CCT_SM = os.path.join(ROOT_DIR, '..', 'CCT-StandardModel')
if os.path.isdir(CCT_SM) and CCT_SM not in sys.path:
    sys.path.insert(0, CCT_SM)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from engine.game import Game
from engine.cycle_clock import ISVParams, CycleClock
from engine.helix_game import HelixGame, PRESETS

# ── styling ──
BG = '#0a0a1a'
FG = '#e0e0e0'
GOLD = '#f0a030'
PURPLE = '#a060f0'
GRID_ALPHA = 0.15

STYLE = {
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'axes.edgecolor': FG,
    'axes.labelcolor': FG,
    'text.color': FG,
    'xtick.color': FG,
    'ytick.color': FG,
    'font.family': 'monospace',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': BG,
}
plt.rcParams.update(STYLE)


# ====================================================================
#  FIGURE A — Coxeter Cycle Polar Diagram  (paper fig 5)
# ====================================================================
def figure_a_coxeter_cycle():
    print("Figure A: Coxeter Cycle Visualization ...")

    # ── build game, add one clock ──
    game = Game(verbose=True)
    clock = game.add_clock(
        fig_vertex=game.fig.origin_idx,
        coxeter_seed=0,
        isv=ISVParams(savings_exponent=28.0),
    )
    e8 = game.e8

    # ── Coxeter eigenvalue exponents ──
    coxeter_exponents = [1, 7, 11, 13, 17, 19, 23, 29]

    # ── run 60 steps (2 full cycles) and record snapshots ──
    snapshots = [clock.snapshot()]
    for _ in range(60):
        game.step()
        snapshots.append(clock.snapshot())

    # ── extract per-step data ──
    phases = [s['coxeter_phase'] for s in snapshots]
    chiralities = [s['chirality'] for s in snapshots]
    fibers = [s['fiber'] for s in snapshots]
    generations = [s['generation'] for s in snapshots]

    # ── figure ──
    fig = plt.figure(figsize=(14, 7))

    # ------ Panel 1: ideal 30-step ring ------
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_facecolor(BG)

    angles_30 = np.linspace(0, 2 * np.pi, 30, endpoint=False)

    # Orbit 0: walk along the Coxeter circuit and record quantum numbers
    orbit0 = e8.orbits[0]
    orbit_fibers = [int(e8.root_fiber[r]) for r in orbit0]
    orbit_chiralities = [e8.quantum_numbers(r)['chirality'] for r in orbit0]
    orbit_generations = [e8.quantum_numbers(r)['coset'] for r in orbit0]

    # Radial position encodes fiber index for visual interest
    radii = np.array([1.0 + 0.08 * (of % 5) for of in orbit_fibers])

    # Color by chirality
    colors_ring = [GOLD if ch > 0 else PURPLE for ch in orbit_chiralities]

    # Draw connections (arc segments)
    for i in range(30):
        j = (i + 1) % 30
        ax1.plot(
            [angles_30[i], angles_30[j]],
            [radii[i], radii[j]],
            color=colors_ring[i], alpha=0.4, linewidth=1.5,
        )

    # Draw nodes
    ax1.scatter(angles_30, radii, c=colors_ring, s=90, zorder=5,
                edgecolors='white', linewidths=0.5)

    # Annotate every 5th step with fiber label
    for i in range(0, 30, 5):
        ax1.annotate(
            f'f{orbit_fibers[i]}',
            xy=(angles_30[i], radii[i]),
            xytext=(angles_30[i], radii[i] + 0.22),
            fontsize=7, ha='center', va='center', color=FG,
        )

    # Mark generation changes
    for i in range(30):
        if orbit_generations[i] != orbit_generations[(i - 1) % 30]:
            ax1.scatter(
                [angles_30[i]], [radii[i]], marker='D', s=30,
                color='#ff4060', zorder=6,
            )

    # Mark eigenvalue exponent positions
    for exp in coxeter_exponents:
        idx = exp % 30
        ax1.scatter(
            [angles_30[idx]], [radii[idx] + 0.13], marker='*',
            s=50, color='#40ff90', zorder=7,
        )

    ax1.set_title('Coxeter Element Orbit (order 30)', pad=18, fontsize=13)
    ax1.set_rticks([])
    ax1.set_thetagrids(np.degrees(angles_30), labels=[str(i) for i in range(30)],
                       fontsize=7, color=FG)
    ax1.grid(alpha=GRID_ALPHA)

    # ------ Panel 2: simulated 60-step path on the ring ------
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_facecolor(BG)

    # Map each simulation step's coxeter_phase to its angular position
    sim_angles = [angles_30[p] for p in phases]
    sim_radii_base = 1.0

    # Draw underlying ring faintly
    ax2.scatter(angles_30, [sim_radii_base] * 30, c='#333355', s=40, zorder=2)

    # Overlay simulation path — spiral outward slightly per step for visibility
    sim_radii = [sim_radii_base + 0.005 * t for t in range(len(phases))]
    sim_colors = [GOLD if ch > 0 else PURPLE for ch in chiralities]

    # Path segments
    for t in range(len(phases) - 1):
        ax2.plot(
            [sim_angles[t], sim_angles[t + 1]],
            [sim_radii[t], sim_radii[t + 1]],
            color=sim_colors[t], alpha=0.6, linewidth=1.0,
        )

    ax2.scatter(sim_angles, sim_radii, c=sim_colors, s=25, zorder=5,
                edgecolors='white', linewidths=0.3)

    # Mark cycle boundaries
    ax2.scatter([sim_angles[0]], [sim_radii[0]], marker='o', s=120,
                facecolors='none', edgecolors='#40ff90', linewidths=2, zorder=8,
                label='start')
    ax2.scatter([sim_angles[30]], [sim_radii[30]], marker='s', s=100,
                facecolors='none', edgecolors='#ff4060', linewidths=2, zorder=8,
                label='step 30')
    ax2.scatter([sim_angles[60]], [sim_radii[60]], marker='^', s=100,
                facecolors='none', edgecolors='#40a0ff', linewidths=2, zorder=8,
                label='step 60')

    ax2.set_title('Simulated Path (60 steps = 2 cycles)', pad=18, fontsize=13)
    ax2.set_rticks([])
    ax2.set_thetagrids(np.degrees(angles_30), labels=[str(i) for i in range(30)],
                       fontsize=7, color=FG)
    ax2.grid(alpha=GRID_ALPHA)
    ax2.legend(loc='lower right', fontsize=8, framealpha=0.3,
               labelcolor=FG, facecolor=BG)

    # ------ Top annotation ------
    fig.suptitle(
        'Coxeter Cycle  —  8 orbits of 30 roots,  fiber period = 15\n'
        f'eigenvalue exponents: {coxeter_exponents}',
        fontsize=14, y=1.02, color=FG,
    )

    # Legend for chirality colors
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=GOLD,
               markersize=8, label='chirality +1'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=PURPLE,
               markersize=8, label='chirality  -1'),
        Line2D([0], [0], marker='*', color='none', markerfacecolor='#40ff90',
               markersize=10, label='eigenvalue exp'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='#ff4060',
               markersize=6, label='generation change'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.3, facecolor=BG, labelcolor=FG)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = os.path.join(OUT_DIR, 'fig5_coxeter_cycle.png')
    plt.savefig(path)
    plt.close()
    print(f"  saved → {path}")


# ====================================================================
#  FIGURE B — Side-by-Side Behavior Comparison  (paper fig 6)
# ====================================================================
def figure_b_behavior_comparison():
    print("Figure B: Side-by-Side Behavior Comparison ...")

    preset_names = ['teeter_totter', 'expansion_contraction', 'cycling_chasing']
    preset_labels = ['Teeter-Totter', 'Expansion-Contraction', 'Cycling-Chasing']
    n_steps = 100
    n_trials = 5

    # ── collect data for each preset ──
    all_data = {}

    for pname in preset_names:
        print(f"  running {pname} ({n_trials} trials x {n_steps} steps) ...")
        trial_distances = []
        trial_savings_0 = []
        trial_savings_1 = []
        trial_positions_0 = []
        trial_positions_1 = []

        for trial in range(n_trials):
            hg = HelixGame(empire_radius=8, verbose=False)
            hg.init_from_preset(pname)

            distances = []
            sav0_list = []
            sav1_list = []
            pos0_list = [hg.emperors[0].position.copy()]
            pos1_list = [hg.emperors[1].position.copy()]

            for _ in range(n_steps):
                entry = hg.step()
                if 'interaction' in entry:
                    distances.append(entry['interaction']['distance'])
                emp_data = entry.get('emperors', [])
                sav0_list.append(emp_data[0]['savings'] if len(emp_data) > 0 else 0)
                sav1_list.append(emp_data[1]['savings'] if len(emp_data) > 1 else 0)
                pos0_list.append(hg.emperors[0].position.copy())
                pos1_list.append(hg.emperors[1].position.copy())

            trial_distances.append(distances)
            trial_savings_0.append(sav0_list)
            trial_savings_1.append(sav1_list)
            trial_positions_0.append(np.array(pos0_list))
            trial_positions_1.append(np.array(pos1_list))

        all_data[pname] = {
            'distances': trial_distances,
            'savings_0': trial_savings_0,
            'savings_1': trial_savings_1,
            'positions_0': trial_positions_0,
            'positions_1': trial_positions_1,
        }

    # ── figure: 3 rows x 3 columns ──
    fig = plt.figure(figsize=(18, 15))

    # Determine global axis limits across all presets for consistency
    all_dists_flat = []
    all_sav_flat = []
    for pname in preset_names:
        for d in all_data[pname]['distances']:
            all_dists_flat.extend(d)
        for s in all_data[pname]['savings_0']:
            all_sav_flat.extend(s)
        for s in all_data[pname]['savings_1']:
            all_sav_flat.extend(s)

    dist_ylim = (0, max(all_dists_flat) * 1.1) if all_dists_flat else (0, 10)
    sav_ylim = (0, max(all_sav_flat) * 1.1) if all_sav_flat else (0, 100)

    for ci, (pname, plabel) in enumerate(zip(preset_names, preset_labels)):
        data = all_data[pname]

        # ── Row 1: 3D trajectory ──
        ax3d = fig.add_subplot(3, 3, ci + 1, projection='3d')
        ax3d.set_facecolor(BG)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor(BG)
        ax3d.yaxis.pane.set_edgecolor(BG)
        ax3d.zaxis.pane.set_edgecolor(BG)

        # Plot first trial's trajectory
        p0 = data['positions_0'][0]
        p1 = data['positions_1'][0]
        ax3d.plot(p0[:, 0], p0[:, 1], p0[:, 2],
                  color=GOLD, linewidth=1.2, alpha=0.8, label='Emperor 0')
        ax3d.plot(p1[:, 0], p1[:, 1], p1[:, 2],
                  color=PURPLE, linewidth=1.2, alpha=0.8, label='Emperor 1')

        # Start / end markers
        ax3d.scatter(*p0[0], color=GOLD, s=60, marker='o', zorder=5)
        ax3d.scatter(*p0[-1], color=GOLD, s=60, marker='^', zorder=5)
        ax3d.scatter(*p1[0], color=PURPLE, s=60, marker='o', zorder=5)
        ax3d.scatter(*p1[-1], color=PURPLE, s=60, marker='^', zorder=5)

        ax3d.set_title(plabel, fontsize=13, pad=10)
        ax3d.tick_params(labelsize=7, colors=FG)
        if ci == 0:
            ax3d.legend(fontsize=7, loc='upper left', framealpha=0.3,
                        facecolor=BG, labelcolor=FG)

        # ── Row 2: Distance time series ──
        ax_d = fig.add_subplot(3, 3, ci + 4)
        for t in range(n_trials):
            ax_d.plot(data['distances'][t], color='#6688bb', alpha=0.3, linewidth=0.7)
        # Bold mean
        max_len = min(len(d) for d in data['distances'])
        mean_dist = np.mean([d[:max_len] for d in data['distances']], axis=0)
        ax_d.plot(mean_dist, color='#40c0ff', linewidth=2.0, label='mean')
        ax_d.set_ylim(dist_ylim)
        ax_d.set_xlabel('Step')
        ax_d.set_ylabel('Distance')
        ax_d.set_title(f'{plabel} — Distance', fontsize=11)
        ax_d.legend(fontsize=8, framealpha=0.3, facecolor=BG, labelcolor=FG)
        ax_d.grid(alpha=GRID_ALPHA)

        # ── Row 3: Savings time series ──
        ax_s = fig.add_subplot(3, 3, ci + 7)
        for t in range(n_trials):
            ax_s.plot(data['savings_0'][t], color=GOLD, alpha=0.25, linewidth=0.7)
            ax_s.plot(data['savings_1'][t], color=PURPLE, alpha=0.25, linewidth=0.7)
        max_len_s = min(len(s) for s in data['savings_0'])
        mean_sav0 = np.mean([s[:max_len_s] for s in data['savings_0']], axis=0)
        mean_sav1 = np.mean([s[:max_len_s] for s in data['savings_1']], axis=0)
        ax_s.plot(mean_sav0, color=GOLD, linewidth=2.0, label='Emp 0 mean')
        ax_s.plot(mean_sav1, color=PURPLE, linewidth=2.0, label='Emp 1 mean')
        ax_s.set_ylim(sav_ylim)
        ax_s.set_xlabel('Step')
        ax_s.set_ylabel('Savings')
        ax_s.set_title(f'{plabel} — Savings', fontsize=11)
        ax_s.legend(fontsize=8, framealpha=0.3, facecolor=BG, labelcolor=FG)
        ax_s.grid(alpha=GRID_ALPHA)

    fig.suptitle(
        'Behavior Comparison  —  HelixGame (empire_radius = 8)\n'
        f'{n_trials} trials, {n_steps} steps each',
        fontsize=15, y=1.01, color=FG,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, 'fig6_behavior_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  saved → {path}")


# ====================================================================
#  FIGURE C — Savings Heatmap on FIG  (paper fig 7)
# ====================================================================
def figure_c_savings_heatmap():
    print("Figure C: Savings Heatmap on FIG ...")

    game = Game(verbose=True)
    fig_b = game.fig
    pos = fig_b.pos_3d
    n = fig_b.n_vertices

    # ── pick three reference ("other emperor") positions ──
    # Sort vertices by distance from origin to choose near / mid / far
    origin = fig_b.origin_idx
    dists_from_origin = np.linalg.norm(pos - pos[origin], axis=1)
    sorted_by_dist = np.argsort(dists_from_origin)

    # near: ~10th percentile; mid: ~50th; far: ~90th
    ref_near = sorted_by_dist[max(1, n // 10)]
    ref_mid = sorted_by_dist[n // 2]
    ref_far = sorted_by_dist[int(n * 0.9)]
    refs = [ref_near, ref_mid, ref_far]
    ref_labels = [
        f'Near (d={dists_from_origin[ref_near]:.2f})',
        f'Mid  (d={dists_from_origin[ref_mid]:.2f})',
        f'Far  (d={dists_from_origin[ref_far]:.2f})',
    ]

    # ── for each ref vertex, compute savings for every other vertex ──
    savings_maps = []
    for ref_v in refs:
        sav_vals = np.zeros(n)
        ref_empire = game.empire.segment_empire[ref_v]
        for v in range(n):
            overlap = len(game.empire.segment_empire[v] & ref_empire)
            sav_vals[v] = overlap
        savings_maps.append(sav_vals)

    # ── figure: 3 panels ──
    fig_obj = plt.figure(figsize=(18, 6))

    # Global colorbar range
    vmin = 0
    vmax = max(sm.max() for sm in savings_maps)

    for pi, (sav_map, ref_v, ref_lbl) in enumerate(
            zip(savings_maps, refs, ref_labels)):
        ax = fig_obj.add_subplot(1, 3, pi + 1, projection='3d')
        ax.set_facecolor(BG)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BG)
        ax.yaxis.pane.set_edgecolor(BG)
        ax.zaxis.pane.set_edgecolor(BG)

        # Scatter all vertices colored by savings
        sc = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=sav_map, cmap='inferno', s=12, alpha=0.75,
            vmin=vmin, vmax=vmax, edgecolors='none',
        )

        # Mark reference (other emperor) with large marker
        ax.scatter(
            [pos[ref_v, 0]], [pos[ref_v, 1]], [pos[ref_v, 2]],
            color=PURPLE, s=200, marker='*', zorder=10,
            edgecolors='white', linewidths=1.0, label='Other emperor',
        )

        # Mark origin
        ax.scatter(
            [pos[origin, 0]], [pos[origin, 1]], [pos[origin, 2]],
            color=GOLD, s=150, marker='D', zorder=10,
            edgecolors='white', linewidths=1.0, label='Origin',
        )

        ax.set_title(ref_lbl, fontsize=12, pad=10)
        ax.tick_params(labelsize=7, colors=FG)

        if pi == 0:
            ax.legend(fontsize=8, loc='upper left', framealpha=0.3,
                      facecolor=BG, labelcolor=FG)

    # Shared colorbar
    cbar_ax = fig_obj.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig_obj.colorbar(sc, cax=cbar_ax)
    cb.set_label('Segment Empire Overlap (savings)', color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)

    fig_obj.suptitle(
        f'Savings Heatmap on FIG  ({n} vertices)\n'
        'Color = segment empire overlap with other emperor position',
        fontsize=14, y=1.02, color=FG,
    )

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    path = os.path.join(OUT_DIR, 'fig7_savings_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  saved → {path}")


# ====================================================================
#  Main
# ====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Kinematic Spinors — Paper Figure Generation")
    print("=" * 60)

    figure_a_coxeter_cycle()
    figure_b_behavior_comparison()
    figure_c_savings_heatmap()

    print("\n" + "=" * 60)
    print(f"  All figures saved to {OUT_DIR}/")
    print("  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith('fig'):
            print(f"    {f}")
    print("=" * 60)
