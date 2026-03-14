"""
CycleClock: A kinematic spinor on the E8 gameboard.

A cycle clock is the central dynamical object of CCT — a discrete
trajectory of a Clifford rotor on the E8 root lattice. It carries:

  Position:      which E8 root (or FIG vertex) it occupies
  Coxeter phase: step k ∈ {0,...,29} in the Coxeter circuit
  ISV state:     intrinsic state variables (quantum numbers + scaling)

The clock TICKS by applying the Coxeter element, advancing k → k+1,
which simultaneously updates:
  - Chirality   (flips every 15 steps via C¹⁵)
  - Generation  (mixes A₈ cosets via C¹⁰)
  - Gauge       (rotates D₄ compound via C⁶)

Movement on the gameboard follows the LEAST CHANGE PRINCIPLE:
the clock walks to the neighbor whose empire overlap (savings)
is maximal — minimizing the disruption to its non-local field.

ISV (Intrinsic State Variables) from CCT Book §3.4.4:
  "Cycle Clocks are functions written generally in Cl(8) that describe
   intrinsic state variables (ISVs). ISVs come in two categories:
   one is the cyclic group data, the other is the scaling information."

Mathematical foundation:
  Irwin, "Kinematic Spinors on Division-Algebra Root Systems" (2026)
  Irwin, "Klee's Book on CCT" — Chapter 3: Cl(8) and cycle clocks
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from e8_gameboard import E8Gameboard


@dataclass
class ISVState:
    """Intrinsic State Variables — the internal quantum state of a cycle clock.

    Category 1: Cyclic group data (derived from Coxeter phase)
      chirality:   +1 or -1 (from Cl(8) bipartite: εᴬ·εᴮ)
      generation:  0, 1, or 2 (A₈ coset label L₀, L₁, L₂)
      gauge_phase: 0-4 (which of 5 D₄ compound orientations)

    Category 2: Scaling information (settable parameters)
      amplitude:   float > 0, the "size" of the clock's excitation
      frequency:   float > 0, rate of Coxeter stepping (for continuous-time extension)
      phase_offset: float, initial phase within the Coxeter circuit

    Category 3: Interaction parameters (for front-end control)
      savings_exponent: how strongly the least-change principle is enforced
      coupling_strength: how strongly this clock's empire affects other clocks
      chirality_coupling: whether same-chirality clocks attract or repel
    """
    # Cyclic group data (auto-updated from Coxeter phase)
    chirality: int = 1           # +1 or -1
    generation: int = 0          # 0, 1, 2
    gauge_phase: int = 0         # 0-4

    # Scaling parameters (user-settable ISVs)
    amplitude: float = 1.0       # excitation strength
    frequency: float = 1.0       # ticking rate
    phase_offset: float = 0.0    # initial Coxeter phase

    # Interaction parameters (front-end sliders)
    savings_exponent: float = 28.0    # PEL strength (higher = more deterministic)
    coupling_strength: float = 1.0    # empire interaction weight
    chirality_coupling: float = 1.0   # +1 = same-chirality attract, -1 = repel


class CycleClock:
    """A kinematic spinor living on the E8 gameboard.

    Usage:
        board = E8Gameboard()
        clock = CycleClock(board, root_idx=0)
        clock.tick()       # advance one Coxeter step
        clock.tick(n=10)   # advance 10 steps
        print(clock.state) # current quantum numbers
    """

    def __init__(self, board: E8Gameboard, root_idx: int = 0,
                 isv: Optional[ISVState] = None):
        """Initialize a cycle clock at a given root.

        Args:
            board: the E8Gameboard instance
            root_idx: starting root index (0-239)
            isv: initial ISV state (uses defaults if None)
        """
        self.board = board
        self.root_idx = root_idx
        self.isv = isv or ISVState()

        # Coxeter phase: which step in the 30-cycle
        self.coxeter_phase = 0

        # History
        self.history: List[Dict] = []
        self._record_state()

        # Update ISV from initial position
        self._update_isv_from_position()

    def _update_isv_from_position(self):
        """Update the cyclic ISVs from the current root's quantum numbers."""
        qn = self.board.root_quantum_numbers(self.root_idx)
        self.isv.chirality = qn['chirality_sign']
        self.isv.generation = qn['coset']
        self.isv.gauge_phase = qn['pair']

    def _record_state(self):
        """Record the current state in history."""
        self.history.append(self.snapshot())

    def snapshot(self) -> Dict:
        """Return a complete snapshot of the clock's state."""
        qn = self.board.root_quantum_numbers(self.root_idx)
        return {
            'step': self.coxeter_phase,
            'root_idx': self.root_idx,
            'root_8d': self.board.roots[self.root_idx].tolist(),
            'fiber': qn['fiber'],
            'pair': qn['pair'],
            'coset': qn['coset'],
            'chirality': qn['chirality_sign'],
            'isv': {
                'amplitude': self.isv.amplitude,
                'frequency': self.isv.frequency,
                'savings_exponent': self.isv.savings_exponent,
                'coupling_strength': self.isv.coupling_strength,
            }
        }

    @property
    def state(self) -> Dict:
        """Current state as a dict."""
        return self.snapshot()

    @property
    def fiber(self) -> int:
        return self.board.root_fiber[self.root_idx]

    @property
    def coset(self) -> int:
        return self.board.root_coset[self.root_idx]

    @property
    def chirality(self) -> int:
        return self.isv.chirality

    @property
    def empire(self) -> set:
        return self.board.empire[self.root_idx]['full']

    # === DYNAMICS ===

    def tick(self, n: int = 1):
        """Advance the Coxeter circuit by n steps.

        Each tick:
        1. Apply Coxeter element to advance internal phase
        2. Update quantum numbers
        3. Record state
        """
        for _ in range(n):
            # Advance root along Coxeter orbit
            self.root_idx = self.board.coxeter_perm[self.root_idx]
            self.coxeter_phase = (self.coxeter_phase + 1) % 30

            # Update cyclic ISVs
            self._update_isv_from_position()

            # Record
            self._record_state()

    def walk(self, target_root: int):
        """Walk the clock to a specific neighboring root.

        This is a SPATIAL move on the gameboard, distinct from ticking
        (which is an internal-state evolution).

        The walk changes the root but preserves the Coxeter phase.
        """
        if target_root not in self.board.adjacency[self.root_idx]:
            raise ValueError(f"Root {target_root} is not a neighbor of {self.root_idx}")

        self.root_idx = target_root
        self._update_isv_from_position()
        self._record_state()

    def step(self):
        """Combined tick + walk: advance the Coxeter phase AND move spatially.

        This is the full cycle clock dynamics from the Non-Local Game of Life:
        1. Tick the Coxeter circuit (internal state evolution)
        2. Choose a neighbor by savings-weighted probability
        3. Walk to that neighbor

        Returns the chosen neighbor and its probability.
        """
        # 1. Tick
        self.tick()

        # 2. Choose neighbor by savings
        probs = self.board.step_probabilities(
            self.root_idx,
            exponent=self.isv.savings_exponent
        )
        if not probs:
            return None, 0.0

        # 3. Weighted random choice
        indices = [idx for idx, _ in probs]
        weights = [p for _, p in probs]
        chosen = np.random.choice(indices, p=weights)
        chosen_prob = weights[indices.index(chosen)]

        # 4. Walk
        self.walk(chosen)

        return chosen, chosen_prob

    # === TWO-CLOCK INTERACTION ===

    def interaction_savings(self, other: 'CycleClock') -> int:
        """Compute the empire overlap between this clock and another.

        This is the "two-particle savings" from the Non-Local GoL paper:
        the size of the intersection of both clocks' empires.
        """
        return len(self.empire & other.empire)

    def interaction_modifier(self, other: 'CycleClock') -> float:
        """Compute the interaction modifier based on internal states.

        Two clocks interact differently depending on their quantum numbers:
        - Same chirality: attract (positive modifier)
        - Opposite chirality: repel (negative modifier)
        - Same generation: stronger coupling
        - Same gauge phase: resonance

        Returns a float multiplier for the interaction strength.
        """
        modifier = 1.0

        # Chirality interaction
        if self.chirality == other.chirality:
            modifier *= (1.0 + self.isv.chirality_coupling)
        else:
            modifier *= (1.0 - self.isv.chirality_coupling)

        # Generation resonance
        if self.coset == other.coset:
            modifier *= 1.5  # same generation = stronger coupling

        # Gauge resonance
        if self.board.fiber_pair[self.fiber] == self.board.fiber_pair[other.fiber]:
            modifier *= 1.2  # same perp pair = gauge resonance

        return modifier * self.isv.coupling_strength

    def step_with_interaction(self, other: 'CycleClock'):
        """Take a step influenced by another clock's empire.

        The transition probability is modified by:
        1. Own savings (as usual)
        2. Other clock's empire overlap (penalize moves into other's field)
        3. Interaction modifier (chirality/generation/gauge coupling)

        This implements the "two to tango" dynamics from the paper.
        """
        # Tick
        self.tick()

        # Get base probabilities
        neighbors = self.board.adjacency[self.root_idx]
        if not neighbors:
            return None, 0.0

        # Compute modified weights
        interaction_mod = self.interaction_modifier(other)
        other_empire = other.empire

        weights = []
        for j in neighbors:
            base_savings = self.board.savings(self.root_idx, j)

            # Penalty for moving into the other clock's empire
            # (empires should not overlap — from the paper's rules)
            overlap_with_other = len(self.board.empire[j]['local'] & other_empire)
            interaction_term = overlap_with_other * interaction_mod

            # Combined weight
            w = (base_savings + 1) ** self.isv.savings_exponent
            w *= np.exp(-interaction_term * 0.1)  # soft repulsion
            weights.append(max(w, 1e-10))

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        # Choose
        chosen = np.random.choice(neighbors, p=weights)
        chosen_prob = weights[list(neighbors).index(chosen)]

        # Walk
        self.walk(chosen)
        return chosen, chosen_prob


class GameSimulation:
    """Run a simulation of one or two cycle clocks on the E8 gameboard.

    This is the Python equivalent of the Mathematica FIGSimulation.
    """

    def __init__(self, board: Optional[E8Gameboard] = None):
        self.board = board or E8Gameboard()
        self.clocks: List[CycleClock] = []
        self.step_count = 0

    def add_clock(self, root_idx: int = 0, isv: Optional[ISVState] = None) -> CycleClock:
        """Add a cycle clock to the simulation."""
        clock = CycleClock(self.board, root_idx=root_idx, isv=isv)
        self.clocks.append(clock)
        return clock

    def step(self):
        """Advance the simulation by one step.

        If one clock: self-interaction (solo walk).
        If two clocks: mutual interaction (two to tango).
        """
        self.step_count += 1

        if len(self.clocks) == 1:
            self.clocks[0].step()

        elif len(self.clocks) == 2:
            # Both clocks step with awareness of each other
            self.clocks[0].step_with_interaction(self.clocks[1])
            self.clocks[1].step_with_interaction(self.clocks[0])

    def run(self, n_steps: int = 30):
        """Run the simulation for n steps."""
        for _ in range(n_steps):
            self.step()

    def export_history(self) -> Dict:
        """Export full simulation history for visualization."""
        return {
            'n_steps': self.step_count,
            'clocks': [
                {
                    'id': i,
                    'history': clock.history,
                    'isv_params': {
                        'amplitude': clock.isv.amplitude,
                        'frequency': clock.isv.frequency,
                        'savings_exponent': clock.isv.savings_exponent,
                        'coupling_strength': clock.isv.coupling_strength,
                        'chirality_coupling': clock.isv.chirality_coupling,
                    }
                }
                for i, clock in enumerate(self.clocks)
            ],
            'board_summary': {
                'n_roots': self.board.n_roots,
                'n_fibers': len(self.board.shells),
                'perp_pairs': self.board.perp_pairs,
            }
        }


# === Quick self-test ===
if __name__ == '__main__':
    print("Building gameboard...")
    board = E8Gameboard()

    # --- Solo clock ---
    print("\n=== SOLO CYCLE CLOCK ===")
    clock = CycleClock(board, root_idx=0)
    print(f"Initial state: {clock.state}")

    print("\nTicking through 30 Coxeter steps:")
    for k in range(30):
        clock.tick()
        s = clock.state
        print(f"  k={s['step']:>2}: root {s['root_idx']:>3}, "
              f"fiber C{s['fiber']}, coset L{s['coset']}, "
              f"chirality {'+'if s['chirality']>0 else '-'}")

    # --- Two clocks ---
    print("\n=== TWO-CLOCK INTERACTION ===")
    sim = GameSimulation(board)
    c1 = sim.add_clock(root_idx=0, isv=ISVState(savings_exponent=10))
    c2 = sim.add_clock(root_idx=100, isv=ISVState(savings_exponent=10))

    print(f"Clock 1: root {c1.root_idx}, fiber C{c1.fiber}, chirality {'+' if c1.chirality>0 else '-'}")
    print(f"Clock 2: root {c2.root_idx}, fiber C{c2.fiber}, chirality {'+' if c2.chirality>0 else '-'}")
    print(f"Empire overlap: {c1.interaction_savings(c2)}")
    print(f"Interaction modifier: {c1.interaction_modifier(c2):.3f}")

    print("\nRunning 10 steps...")
    for step in range(10):
        sim.step()
        overlap = c1.interaction_savings(c2)
        print(f"  Step {step+1}: C1@root{c1.root_idx}(C{c1.fiber}) "
              f"C2@root{c2.root_idx}(C{c2.fiber}) "
              f"overlap={overlap}")

    print("\nSimulation complete.")
    history = sim.export_history()
    print(f"History: {history['n_steps']} steps, {len(history['clocks'])} clocks")
