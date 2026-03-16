"""
Helix-based simulation engine for the Non-Local Game of Life.

Port of the 2022 Mathematica simulation (FIG Dynamics Simulation - v6.wl).
Two emperor particles walk along pentagonal helix segments on the FIG,
with move probabilities driven by empire savings (hash-table overlap counting).

This is the helix-mode alternative to the segment-hop Game in game.py.
The key differences:
  - Movement: 6-vertex helix paths along 5-fold axes (not single edges)
  - Point set: 20G centroids (~15K-21K points, not 779)
  - Savings: Hash-table lookup of translated positions
  - Chirality: Geometric mirror reflection of helix handedness
"""

import numpy as np
import time
from .fig_icosagrid import FIGIcosagrid, TET_GROUPS
from .helix_builder import HelixBuilder


# Configuration presets for the three canonical behaviors
PRESETS = {
    'teeter_totter': {
        'emperor1': {'axis_group': 3, 'chirality': 'L', 'position': [0, 0, 0]},
        'emperor2': {'axis_group': 3, 'chirality': 'R', 'position': 'empire[2]'},
        'exponent': 28,
        'description': 'Opposite chirality → oscillatory distance',
    },
    'expansion_contraction': {
        'emperor1': {'axis_group': 3, 'chirality': 'L', 'position': [0, 0, 0]},
        'emperor2': {'axis_group': 3, 'chirality': 'L', 'position': 'empire[2]'},
        'exponent': 17,
        'description': 'Same chirality → expansion/contraction cycles',
    },
    'cycling_chasing': {
        'emperor1': {'axis_group': 1, 'chirality': 'L', 'position': [0, 0, 0]},
        'emperor2': {'axis_group': 1, 'chirality': 'L', 'position': 'empire[10]'},
        'exponent': 7,
        'description': 'Low exponent → cycling/chasing behavior',
    },
}


class Emperor:
    """A single emperor particle on the FIG."""

    def __init__(self, position, segments_L, segments_R, chirality='L',
                 exponent=28, emperor_id=0):
        self.position = np.asarray(position, dtype=np.float64)
        self.segments_L = segments_L  # left-handed helix segments
        self.segments_R = segments_R  # right-handed helix segments
        self.chirality = chirality
        self.exponent = exponent
        self.emperor_id = emperor_id

        # History
        self.position_history = [self.position.copy()]
        self.savings_history = []
        self.chosen_segment_history = []

    @property
    def segments(self):
        """Active segments based on current chirality."""
        return self.segments_L if self.chirality == 'L' else self.segments_R

    def snapshot(self):
        return {
            'emperor_id': self.emperor_id,
            'position': self.position.tolist(),
            'chirality': self.chirality,
            'exponent': self.exponent,
            'n_steps': len(self.position_history) - 1,
        }


class HelixGame:
    """Helix-based simulation engine matching the 2022 Mathematica code.

    Two emperors walk on pentagonal helix paths with move probabilities
    determined by empire savings (hash-table overlap counting).
    """

    def __init__(self, empire_radius=8, neighborhood_radius=500,
                 cylinder_length=1000, verbose=True):
        t0 = time.time()

        if verbose:
            print("Building FIG icosagrid...")
        self.grid = FIGIcosagrid()

        if verbose:
            print(f"Generating FIG empire vertices (radius={empire_radius})...")
        self.empire_radius = empire_radius
        self.empire = self.grid.vertices([0, 0, 0], empire_radius)
        if verbose:
            print(f"  Empire: {len(self.empire)} vertices")

        # Hash table for O(1) position lookup
        self.empire_lookup = set()
        for p in self.empire:
            self.empire_lookup.add(tuple(np.round(p, 8)))

        if verbose:
            print("Building helix segments...")
        self.helix_builder = HelixBuilder(self.grid)
        self.neighborhood_radius = neighborhood_radius
        self.cylinder_length = cylinder_length

        # Cache of helix segments per axis group
        self._segment_cache = {}

        self.emperors = []
        self.step_log = []
        self.step_count = 0

        build_time = time.time() - t0
        if verbose:
            print(f"HelixGame ready in {build_time:.1f}s")

    def _get_segments(self, axis_group, chirality):
        """Get (or build) helix segments for an axis group + chirality."""
        key = (axis_group, chirality)
        if key not in self._segment_cache:
            segments_L = self.helix_builder.build_axis_group_segments(
                center=np.zeros(3),
                axis_group_idx=axis_group,
                fig_vertices=self.empire,
                n_wafers=20,
                segment_length=6,
                pent_radius=0.25,
            )
            if chirality == 'L':
                self._segment_cache[key] = segments_L
            else:
                self._segment_cache[key] = self.helix_builder.chiral_reverse(segments_L)

            # Also cache the other chirality
            other_key = (axis_group, 'R' if chirality == 'L' else 'L')
            if other_key not in self._segment_cache:
                if chirality == 'L':
                    self._segment_cache[other_key] = \
                        self.helix_builder.chiral_reverse(segments_L)
                else:
                    self._segment_cache[other_key] = segments_L

        return self._segment_cache[key]

    def add_emperor(self, position, axis_group=3, chirality='L', exponent=28):
        """Add an emperor particle with helix movement."""
        position = np.asarray(position, dtype=np.float64)

        # Resolve symbolic positions like 'empire[2]'
        # (handled in init_from_preset)

        segments_L = self._get_segments(axis_group, 'L')
        segments_R = self._get_segments(axis_group, 'R')

        emperor = Emperor(
            position=position,
            segments_L=segments_L,
            segments_R=segments_R,
            chirality=chirality,
            exponent=exponent,
            emperor_id=len(self.emperors),
        )
        self.emperors.append(emperor)
        return emperor

    def _compute_savings(self, segment, other_position, return_hits=False):
        """Count empire overlaps for a translated helix segment.

        Translate segment by -other_position, flatten vertices,
        deduplicate, count how many land on empire positions.

        If return_hits=True, also returns list of indices into `segment`
        that produced hits (for visualization).
        """
        translated = segment - other_position
        rounded = np.round(translated, 8)

        seen = set()
        hits = 0
        hit_indices = []
        for idx, pt in enumerate(rounded):
            key = tuple(pt)
            if key not in seen:
                seen.add(key)
                if key in self.empire_lookup:
                    hits += 1
                    if return_hits:
                        hit_indices.append(idx)
        if return_hits:
            return hits, hit_indices
        return hits

    def step(self):
        """One simulation step.

        For each emperor:
          1. Translate all helix segments to current position
          2. For each candidate segment, compute savings vs other emperor
          3. weights = (savings + 1)^exponent
          4. RandomChoice(weights) → pick segment
          5. Move to endpoint of chosen segment
        """
        self.step_count += 1
        step_data = {'step': self.step_count, 'emperors': []}

        for ei, emperor in enumerate(self.emperors):
            # Get the other emperor (if any)
            other_pos = None
            if len(self.emperors) > 1:
                other = self.emperors[1 - ei]
                other_pos = other.position

            # Translate segments to current position
            translated_segments = []
            for seg in emperor.segments:
                translated_segments.append(seg + emperor.position)

            # Compute savings for each segment
            savings_list = []
            if other_pos is not None:
                for seg in translated_segments:
                    s = self._compute_savings(seg, other_pos)
                    savings_list.append(s)
            else:
                savings_list = [0] * len(translated_segments)

            # Compute weights: (savings + 1)^exponent
            savings_arr = np.array(savings_list, dtype=np.float64)
            weights = np.power(savings_arr + 1, emperor.exponent)
            total_weight = weights.sum()

            if total_weight == 0 or len(weights) == 0:
                # No valid moves — stay in place
                step_data['emperors'].append({
                    'emperor_id': ei,
                    'from': emperor.position.tolist(),
                    'to': emperor.position.tolist(),
                    'savings': 0,
                    'n_options': 0,
                    'probability': 0,
                })
                continue

            # Normalize to probabilities
            probs = weights / total_weight

            # RandomChoice
            chosen_idx = np.random.choice(len(probs), p=probs)
            chosen_segment = translated_segments[chosen_idx]
            chosen_savings = savings_list[chosen_idx]

            # Compute which vertices of the chosen segment hit empire points
            hit_indices = []
            if other_pos is not None:
                _, hit_indices = self._compute_savings(
                    chosen_segment, other_pos, return_hits=True)

            # The hit vertices in world coordinates (for viz)
            hit_positions = chosen_segment[hit_indices].tolist() if hit_indices else []

            # Move to endpoint of chosen segment
            old_pos = emperor.position.copy()
            emperor.position = chosen_segment[-1].copy()
            emperor.position_history.append(emperor.position.copy())
            emperor.savings_history.append(chosen_savings)
            emperor.chosen_segment_history.append(chosen_idx)

            step_data['emperors'].append({
                'emperor_id': ei,
                'from': old_pos.tolist(),
                'to': emperor.position.tolist(),
                'savings': chosen_savings,
                'probability': float(probs[chosen_idx]),
                'n_options': len(probs),
                'best_savings': int(savings_arr.max()),
                'mean_savings': float(savings_arr.mean()),
                'chosen_segment': chosen_segment.tolist(),
                'hit_positions': hit_positions,
                'snapshot': emperor.snapshot(),
            })

        # Interaction metrics
        if len(self.emperors) == 2:
            e0, e1 = self.emperors
            dist = float(np.linalg.norm(e0.position - e1.position))
            step_data['interaction'] = {
                'distance': dist,
                'chirality_match': e0.chirality == e1.chirality,
            }

        self.step_log.append(step_data)
        return step_data

    def run(self, n_steps, n_runs=1, verbose=True):
        """Run simulation(s) and collect trajectory data."""
        all_runs = []
        for run_idx in range(n_runs):
            if n_runs > 1:
                # Reset positions for new run
                for emp in self.emperors:
                    emp.position = emp.position_history[0].copy()
                    emp.position_history = [emp.position.copy()]
                    emp.savings_history = []
                    emp.chosen_segment_history = []
                self.step_log = []
                self.step_count = 0

            for step in range(n_steps):
                entry = self.step()
                if verbose and step % max(1, n_steps // 10) == 0:
                    self._print_status(entry, run_idx)

            all_runs.append(list(self.step_log))

        return all_runs if n_runs > 1 else self.step_log

    def _print_status(self, entry, run_idx=0):
        s = entry['step']
        parts = []
        for ed in entry['emperors']:
            sn = ed.get('snapshot', {})
            chi = sn.get('chirality', '?')
            sav = ed['savings']
            parts.append(f"E{ed['emperor_id']}(sav={sav},χ={chi})")
        status = ' | '.join(parts)
        extras = ''
        if 'interaction' in entry:
            inter = entry['interaction']
            extras = f" | d={inter['distance']:.2f}"
        print(f"  [run{run_idx} step{s:>4}] {status}{extras}")

    def init_from_preset(self, preset_name):
        """Initialize emperors from a named preset configuration."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available: {list(PRESETS.keys())}")

        preset = PRESETS[preset_name]
        self.emperors = []
        self.step_log = []
        self.step_count = 0

        for key in ['emperor1', 'emperor2']:
            cfg = preset[key]
            pos = cfg['position']

            # Resolve symbolic positions
            if isinstance(pos, str) and pos.startswith('empire['):
                idx = int(pos.split('[')[1].rstrip(']'))
                if idx < len(self.empire):
                    pos = self.empire[idx]
                else:
                    pos = self.empire[min(idx, len(self.empire) - 1)]

            self.add_emperor(
                position=pos,
                axis_group=cfg['axis_group'],
                chirality=cfg['chirality'],
                exponent=preset['exponent'],
            )

    def get_state(self):
        """Current state for front-end."""
        state = {
            'step': self.step_count,
            'mode': 'helix',
            'emperors': [e.snapshot() for e in self.emperors],
            'empire_size': len(self.empire),
        }
        if len(self.emperors) == 2:
            e0, e1 = self.emperors
            state['interaction'] = {
                'distance': float(np.linalg.norm(e0.position - e1.position)),
                'chirality_match': e0.chirality == e1.chirality,
            }
        if self.step_log:
            state['last_step'] = self.step_log[-1]
        return state
