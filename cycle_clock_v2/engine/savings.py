"""
Savings: segment-based probability computation.

At each step, for each candidate neighbor, compute:
  1. Segment savings (empire intersection)
  2. Interaction term (if another clock is present)
  3. Probability weight = (savings + 1)^exponent
  4. Full probability distribution over all neighbors

This matches the Mathematica code's approach:
  Power[Plus[savings, 1], 28.]
  RandomChoice[Rule[weights, Range[n]]]
"""

import numpy as np


class SavingsComputer:
    """Compute savings and probability distributions for each step."""

    def __init__(self, empire_computer, segment_set):
        self.empire = empire_computer
        self.segs = segment_set

        # Precompute static segment savings for all edges
        # (used as fast path when no other clock is present)
        self._precompute_static()

    def _precompute_static(self):
        """Precompute segment savings for all adjacent pairs."""
        self.static_savings = {}
        for seg_idx, (i, j) in enumerate(self.segs.segments):
            _, sav = self.empire.dynamic_empire_intersection(i, j)
            self.static_savings[(i, j)] = sav
            self.static_savings[(j, i)] = sav

    def get_static_savings(self, from_v, to_v):
        """Get precomputed savings (no other clock interaction)."""
        return self.static_savings.get((from_v, to_v), 0)

    def compute_move_options(self, from_vertex, exponent=28.0,
                             other_clock_vertex=None,
                             chirality_match=None,
                             coupling_strength=1.0):
        """Compute full probability distribution for all candidate moves.

        Returns list of dicts, one per neighbor, sorted by savings (descending):
        [
            {
                'vertex': neighbor_idx,
                'savings': segment_savings,
                'interaction_overlap': segments shared with other clock,
                'weight': raw weight before normalization,
                'probability': normalized probability,
                'rank': 1 = best, 2 = second best, etc.
            },
            ...
        ]
        """
        neighbors = self.segs.adjacency[from_vertex]
        if not neighbors:
            return []

        options = []
        for nb in neighbors:
            if other_clock_vertex is not None:
                sav, interaction = self.empire.dynamic_savings_for_move(
                    from_vertex, nb, other_clock_vertex)
            else:
                sav = self.get_static_savings(from_vertex, nb)
                interaction = 0

            options.append({
                'vertex': nb,
                'savings': sav,
                'interaction_overlap': interaction,
            })

        # Compute weights
        for opt in options:
            base_weight = (opt['savings'] + 1) ** exponent

            # Interaction modifier
            if other_clock_vertex is not None and coupling_strength > 0:
                modifier = 1.0
                if chirality_match is not None:
                    modifier = 1.0 + coupling_strength if chirality_match else 1.0 - coupling_strength * 0.5
                # Penalize moves that overlap with other clock's empire
                penalty = np.exp(-opt['interaction_overlap'] * coupling_strength * 0.01)
                base_weight *= modifier * penalty

            opt['weight'] = max(base_weight, 1e-30)

        # Normalize to probabilities
        total = sum(o['weight'] for o in options)
        for opt in options:
            opt['probability'] = opt['weight'] / total

        # Sort by savings descending, assign ranks
        options.sort(key=lambda x: -x['savings'])
        for i, opt in enumerate(options):
            opt['rank'] = i + 1

        return options

    def choose_move(self, from_vertex, exponent=28.0,
                    other_clock_vertex=None, chirality_match=None,
                    coupling_strength=1.0):
        """Choose a move by weighted random sampling.

        Returns (chosen_vertex, full_options_list).
        """
        options = self.compute_move_options(
            from_vertex, exponent, other_clock_vertex,
            chirality_match, coupling_strength)

        if not options:
            return from_vertex, []

        vertices = [o['vertex'] for o in options]
        probs = [o['probability'] for o in options]
        chosen = np.random.choice(vertices, p=probs)

        return chosen, options
