#!/usr/bin/env python3
"""
Export v2 engine data in v1 JSON format.

Generates fig_gameboard.json and game_history.json that the v1 viz
(cycle_clock_viz/index.html) can load directly. This replaces the
v1 vertex-approximation data with real segment-based empire data
from the v2 engine.

Usage:
  python analysis/export_v1_data.py
  # Then copy outputs to repo root:
  cp data/fig_gameboard_v2.json ../fig_gameboard.json
  cp data/game_history_v2.json ../game_history.json
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'CCT-StandardModel'))

from engine.game import Game
from engine.cycle_clock import ISVParams


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def export(n_steps=90, exponent=28.0):
    print("Building v2 engine...")
    game = Game(max_norm_sq=8, verbose=True)

    origin = game.fig.origin_idx

    # Find far vertex (same logic as v1)
    dists = np.linalg.norm(
        game.fig.pos_3d - game.fig.pos_3d[origin], axis=1)
    candidates = [(i, dists[i]) for i in range(game.fig.n_vertices)
                  if game.empire.vertex_empire_sizes[i] > 120
                  and game.segs.degrees[i] >= 6]
    candidates.sort(key=lambda x: -x[1])
    far_v = candidates[0][0]

    c0 = game.add_clock(fig_vertex=origin, coxeter_seed=0,
                        isv=ISVParams(savings_exponent=exponent))
    c1 = game.add_clock(fig_vertex=far_v, coxeter_seed=100,
                        isv=ISVParams(savings_exponent=exponent))

    # --- fig_gameboard.json ---
    # Compute edge_savings: for each segment, the static savings
    # (empire overlap of both endpoints)
    print("Computing edge savings...")
    edge_savings = []
    for si, (a, b) in enumerate(game.segs.segments):
        overlap = len(game.empire.segment_empire[a] & game.empire.segment_empire[b])
        edge_savings.append(overlap)

    # Empire members: for each vertex, list of vertices in its empire
    # (capped for JSON size)
    empire_members = {}
    for v in range(game.fig.n_vertices):
        members = sorted(list(game.empire.vertex_empire[v]))
        empire_members[str(v)] = members

    board_data = {
        'n_vertices': game.fig.n_vertices,
        'positions_3d': game.fig.pos_3d.tolist(),
        'perp_radii': game.fig.perp_radius.tolist(),
        'empire_sizes': game.empire.vertex_empire_sizes.tolist(),
        'degrees': game.segs.degrees.tolist(),
        'origin_idx': int(origin),
        'edges': game.segs.segments,
        'edge_savings': edge_savings,
        'empire_members': empire_members,
    }

    # --- game_history.json ---
    print(f"Running {n_steps} steps...")
    clock_histories = [[], []]

    # Record initial state
    for ci, clock in enumerate(game.clocks):
        sn = clock.snapshot()
        clock_histories[ci].append({
            'clock_id': ci,
            'step': 0,
            'vertex': int(sn['vertex']),
            'pos_3d': game.fig.pos_3d[sn['vertex']].tolist(),
            'perp_radius': float(game.fig.perp_radius[sn['vertex']]),
            'empire_size': int(game.empire.vertex_empire_sizes[sn['vertex']]),
            'coxeter_phase': int(sn['coxeter_phase']),
            'coxeter_root': int(sn.get('coxeter_root_idx', 0)),
            'fiber': int(sn['fiber']),
            'chirality': int(sn['chirality']),
            'generation': int(sn['generation']),
            'gauge_phase': 0,
            'amplitude': 1.0,
            'a2_fiber': int(sn.get('a2_fiber', 0)),
        })

    step_log = []
    for step in range(n_steps):
        entry = game.step()

        records = []
        for ci, cd in enumerate(entry['clocks']):
            sn = cd['snapshot']
            records.append({
                'from': int(cd['from']),
                'to': int(cd['to']),
                'probability': float(cd['probability']),
                'savings': int(cd['savings']),
                'chirality': int(sn['chirality']),
                'generation': int(sn['generation']),
                'phase': int(sn['coxeter_phase']),
            })

            # Record clock state
            clock_histories[ci].append({
                'clock_id': ci,
                'step': step + 1,
                'vertex': int(sn['vertex']),
                'pos_3d': game.fig.pos_3d[sn['vertex']].tolist(),
                'perp_radius': float(game.fig.perp_radius[sn['vertex']]),
                'empire_size': int(game.empire.vertex_empire_sizes[sn['vertex']]),
                'coxeter_phase': int(sn['coxeter_phase']),
                'coxeter_root': int(sn.get('coxeter_root_idx', 0)),
                'fiber': int(sn['fiber']),
                'chirality': int(sn['chirality']),
                'generation': int(sn['generation']),
                'gauge_phase': 0,
                'amplitude': 1.0,
                'a2_fiber': int(sn.get('a2_fiber', 0)),
            })

        # Interaction record
        if 'interaction' in entry:
            inter = entry['interaction']
            records.append({
                'empire_overlap': int(inter['segment_overlap']),
                'distance_3d': float(inter['distance_3d']),
                'chirality_match': bool(inter['chirality_match']),
            })

        step_log.append({
            'step': step + 1,
            'records': records,
        })

        if (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{n_steps}")

    history_data = {
        'n_steps': n_steps,
        'n_clocks': 2,
        'board': {
            'n_vertices': game.fig.n_vertices,
            'origin_idx': int(origin),
        },
        'clocks': [
            {'id': 0, 'history': clock_histories[0]},
            {'id': 1, 'history': clock_histories[1]},
        ],
        'step_log': step_log,
    }

    # Write files
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)

    board_path = os.path.join(out_dir, 'fig_gameboard_v2.json')
    with open(board_path, 'w') as f:
        json.dump(board_data, f, cls=NpEncoder)
    print(f"Wrote {board_path} ({game.fig.n_vertices} vertices, {len(game.segs.segments)} edges)")

    history_path = os.path.join(out_dir, 'game_history_v2.json')
    with open(history_path, 'w') as f:
        json.dump(history_data, f, cls=NpEncoder)
    print(f"Wrote {history_path} ({n_steps} steps)")

    # Also copy to repo root for v1 viz to find
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    import shutil
    shutil.copy(board_path, os.path.join(root, 'fig_gameboard.json'))
    shutil.copy(history_path, os.path.join(root, 'game_history.json'))
    print(f"Copied to repo root (overwrites v1 data)")


if __name__ == '__main__':
    export()
