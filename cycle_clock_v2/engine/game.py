"""
Game: the simulation engine.

Runs one or two cycle clocks on the FIG with segment-based empire
dynamics and full probability tracking at each step.
"""

import numpy as np
from .e8_algebra import E8Algebra
from .fig_builder import FIGBuilder
from .segments import SegmentSet
from .empire import EmpireComputer
from .savings import SavingsComputer
from .cycle_clock import CycleClock, ISVParams


class Game:
    """The full cycle clock simulation.

    Builds the complete mathematical stack:
      E8 algebra → FIG → segments → empires → savings

    Then runs cycle clocks with step-by-step probability tracking.
    """

    def __init__(self, verbose=True):
        if verbose:
            print("Building E8 algebra...")
        self.e8 = E8Algebra()

        if verbose:
            print("Building FIG...")
        self.fig = FIGBuilder()

        if verbose:
            print(f"Building segments ({self.fig.n_vertices} vertices)...")
        self.segs = SegmentSet(self.fig)

        if verbose:
            print(f"Computing empires ({self.segs.n_segments} segments)...")
        self.empire = EmpireComputer(self.fig, self.segs)

        if verbose:
            print("Precomputing static savings...")
        self.savings = SavingsComputer(self.empire, self.segs)

        if verbose:
            print(f"Ready. {self.fig.n_vertices} vertices, "
                  f"{self.segs.n_segments} segments, "
                  f"segment empires {self.empire.segment_empire_sizes.min()}-"
                  f"{self.empire.segment_empire_sizes.max()}")

        self.clocks = []
        self.step_log = []
        self.step_count = 0

    def add_clock(self, fig_vertex=None, coxeter_seed=0, isv=None):
        if fig_vertex is None:
            fig_vertex = self.fig.origin_idx
        clock = CycleClock(
            self.e8, fig_vertex, coxeter_seed,
            isv=isv, clock_id=len(self.clocks))
        self.clocks.append(clock)
        return clock

    def step(self):
        """Advance one step with full probability tracking."""
        self.step_count += 1
        step_data = {'step': self.step_count, 'clocks': []}

        for ci, clock in enumerate(self.clocks):
            # Tick internal state
            clock.tick()

            # Determine other clock (if present)
            other_vertex = None
            chirality_match = None
            if len(self.clocks) > 1:
                other = self.clocks[1 - ci]
                other_vertex = other.vertex
                chirality_match = clock.chirality == other.chirality

            # Compute ALL options with probabilities
            chosen, options = self.savings.choose_move(
                clock.vertex,
                exponent=clock.isv.savings_exponent,
                other_clock_vertex=other_vertex,
                chirality_match=chirality_match,
                coupling_strength=clock.isv.coupling_strength,
            )

            old_vertex = clock.vertex
            clock.walk(chosen)
            clock.record()

            # Find the chosen option in the list
            chosen_opt = next((o for o in options if o['vertex'] == chosen), None)

            step_data['clocks'].append({
                'clock_id': ci,
                'from': old_vertex,
                'to': chosen,
                'savings': chosen_opt['savings'] if chosen_opt else 0,
                'probability': chosen_opt['probability'] if chosen_opt else 0,
                'rank': chosen_opt['rank'] if chosen_opt else 0,
                'n_options': len(options),
                'best_savings': options[0]['savings'] if options else 0,
                'best_vertex': options[0]['vertex'] if options else 0,
                'worst_savings': options[-1]['savings'] if options else 0,
                'mean_savings': np.mean([o['savings'] for o in options]) if options else 0,
                'all_options': options,
                'snapshot': clock.snapshot(),
            })

        # Interaction metrics
        if len(self.clocks) == 2:
            c0, c1 = self.clocks
            shared, overlap = self.empire.dynamic_empire_intersection(c0.vertex, c1.vertex)
            d3 = float(np.linalg.norm(
                self.fig.pos_3d[c0.vertex] - self.fig.pos_3d[c1.vertex]))
            step_data['interaction'] = {
                'segment_overlap': overlap,
                'distance_3d': d3,
                'chirality_match': c0.chirality == c1.chirality,
                'shared_segment_ids': list(shared)[:50],  # cap for JSON size
            }

        self.step_log.append(step_data)
        return step_data

    def run(self, n_steps, verbose=False):
        for i in range(n_steps):
            entry = self.step()
            if verbose and i % max(1, n_steps // 10) == 0:
                self._print_status(entry)

    def _print_status(self, entry):
        s = entry['step']
        parts = []
        for cd in entry['clocks']:
            sn = cd['snapshot']
            chi = '+' if sn['chirality'] > 0 else '-'
            parts.append(f"C{cd['clock_id']}@v{sn['vertex']}"
                         f"(sav={cd['savings']},rank={cd['rank']}/{cd['n_options']},"
                         f"χ={chi},L{sn['generation']})")
        status = ' | '.join(parts)
        extras = ''
        if 'interaction' in entry:
            inter = entry['interaction']
            extras = f" | seg_overlap={inter['segment_overlap']}, d={inter['distance_3d']:.2f}"
        print(f"  [{s:>4}] {status}{extras}")

    def get_state(self):
        """Get current state for live front-end."""
        state = {
            'step': self.step_count,
            'clocks': [c.snapshot() for c in self.clocks],
            'board': {
                'n_vertices': self.fig.n_vertices,
                'n_segments': self.segs.n_segments,
                'origin_idx': self.fig.origin_idx,
            },
        }
        if len(self.clocks) == 2:
            c0, c1 = self.clocks
            _, overlap = self.empire.dynamic_empire_intersection(c0.vertex, c1.vertex)
            state['interaction'] = {
                'segment_overlap': overlap,
                'distance_3d': float(np.linalg.norm(
                    self.fig.pos_3d[c0.vertex] - self.fig.pos_3d[c1.vertex])),
            }
        if self.step_log:
            state['last_step'] = self.step_log[-1]
        return state

    def update_isv(self, clock_id, params):
        """Live update ISV parameters from front-end."""
        if clock_id < len(self.clocks):
            clock = self.clocks[clock_id]
            for k, v in params.items():
                if hasattr(clock.isv, k):
                    setattr(clock.isv, k, v)
