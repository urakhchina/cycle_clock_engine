"""
Cycle Clock Game: Kinematic spinors walking on the FIG gameboard.

This is the integration layer — cycle clocks (algebraic objects from E8)
moving on the FIG (physical 3D quasicrystal), driven by empire savings
and modulated by internal Coxeter state.

The key insight: the cycle clock has TWO dynamics running simultaneously:
  1. INTERNAL: the Coxeter circuit ticks through 30 steps on E8 roots
     (changing chirality, generation, gauge phase each step)
  2. SPATIAL: the clock walks on the FIG vertices, choosing neighbors
     by empire savings (least change principle)

The internal state MODULATES the spatial walk:
  - Chirality affects interaction with other clocks
  - Generation coset affects coupling strength
  - Savings exponent (ISV parameter) controls determinism

Architecture:
  FIGGameboard  → the physical board (287 vertices, empires, savings)
  E8Gameboard   → the algebraic backbone (240 roots, Coxeter element)
  CycleClock    → a piece: position on FIG + phase on Coxeter circuit
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import json

from fig_gameboard import FIGGameboard
from e8_gameboard import E8Gameboard


@dataclass
class ClockISV:
    """Intrinsic State Variables for front-end control.

    Cyclic data (auto-updated each tick):
      chirality, generation, gauge_phase

    Tunable parameters (front-end sliders):
      savings_exponent:    1-50, PEL strength (higher = more deterministic)
      coupling_strength:   0-2, how strongly this clock affects others
      chirality_coupling:  -1 to +1, same-chirality attract (+) or repel (-)
      amplitude:           0-2, visual size / excitation strength
    """
    # Auto-updated from Coxeter phase
    chirality: int = 1
    generation: int = 0
    gauge_phase: int = 0

    # Tunable (front-end sliders)
    savings_exponent: float = 15.0
    coupling_strength: float = 1.0
    chirality_coupling: float = 0.5
    amplitude: float = 1.0


class CycleClock:
    """A kinematic spinor walking on the FIG.

    Combines:
      - A position on the FIG gameboard (spatial)
      - A phase in the Coxeter circuit (internal, on E8)
    """

    def __init__(self, fig: FIGGameboard, e8: E8Gameboard,
                 fig_vertex: int = None, coxeter_seed: int = 0,
                 isv: ClockISV = None, clock_id: int = 0):
        self.fig = fig
        self.e8 = e8
        self.isv = isv or ClockISV()
        self.clock_id = clock_id

        # Spatial position on FIG
        self.vertex = fig_vertex if fig_vertex is not None else fig.origin_idx

        # Internal state: which root in the Coxeter orbit
        self.coxeter_seed = coxeter_seed
        self.coxeter_phase = 0
        self._coxeter_root = coxeter_seed

        # Compute orbit seed: the canonical first root of this orbit
        self._orbit_seed = self._find_orbit_seed(coxeter_seed)

        # Update ISV from initial state
        self._sync_isv()

        # History
        self.history: List[Dict] = []
        self._record()

    def _find_orbit_seed(self, root_idx):
        """Find the canonical seed of the Coxeter orbit containing this root."""
        # Walk the orbit and return the smallest root index
        orbit = {root_idx}
        idx = root_idx
        for _ in range(29):
            idx = self.e8.coxeter_perm[idx]
            orbit.add(idx)
        return min(orbit)

    def _sync_isv(self):
        """Sync cyclic ISVs from current Coxeter root."""
        qn = self.e8.root_quantum_numbers(self._coxeter_root)
        self.isv.chirality = qn['chirality_sign']
        self.isv.generation = qn['coset']
        self.isv.gauge_phase = qn['pair']

    def _record(self):
        """Record current state."""
        self.history.append(self.snapshot())

    def snapshot(self) -> Dict:
        """Complete state snapshot.

        Emits ALL fields consumed by the viz (Fix: Klee review #2).
        No notebook augmentation needed.
        """
        coxeter_root = int(self._coxeter_root)
        # Compute A2 fiber from root_a2_fiber map if available
        a2_fiber = self._get_a2_fiber(coxeter_root)
        # Local root index within the D4 fiber
        local_root = self._get_local_root(coxeter_root)

        return {
            'clock_id': self.clock_id,
            'step': len(self.history),
            'vertex': self.vertex,
            'pos_3d': self.fig.pos_3d[self.vertex].tolist(),
            'perp_radius': float(self.fig.perp_radius[self.vertex]),
            'empire_size': int(self.fig.empire_sizes[self.vertex]),
            'coxeter_phase': self.coxeter_phase,
            'coxeter_root': coxeter_root,
            'fiber': int(self.e8.root_fiber[coxeter_root]),
            'chirality': self.isv.chirality,
            'generation': self.isv.generation,
            'gauge_phase': self.isv.gauge_phase,
            'amplitude': self.isv.amplitude,
            'a2_fiber': a2_fiber,
            'local_root_in_d4': local_root,
            'orbit_seed': int(self._orbit_seed),
        }

    def _get_a2_fiber(self, root_idx):
        """Get A2 sub-fiber index for this root."""
        if not hasattr(self, '_a2_map'):
            try:
                import json
                with open('clock_geometry.json') as f:
                    geom = json.load(f)
                self._a2_map = {int(k): v for k, v in geom.get('root_a2_fiber', {}).items()}
            except:
                self._a2_map = {}
        return self._a2_map.get(root_idx, 0)

    def _get_local_root(self, root_idx):
        """Get local index (0-23) within the D4 fiber."""
        if not hasattr(self, '_local_map'):
            self._local_map = {}
            for fi, sh in enumerate(self.e8.shells):
                for local_i, global_i in enumerate(sh):
                    self._local_map[global_i] = local_i
        return self._local_map.get(root_idx, 0)

    @property
    def empire(self) -> set:
        return self.fig.empire[self.vertex]

    def tick(self):
        """Advance the Coxeter circuit by one step (internal evolution)."""
        self._coxeter_root = self.e8.coxeter_perm[self._coxeter_root]
        self.coxeter_phase = (self.coxeter_phase + 1) % 30
        self._sync_isv()

    def choose_neighbor(self, other: 'CycleClock' = None) -> Tuple[int, float]:
        """Choose the next FIG vertex to walk to.

        Without another clock: pure savings-weighted choice.
        With another clock: savings modified by empire interaction.

        Returns (chosen_vertex, probability).
        """
        neighbors = self.fig.adjacency[self.vertex]
        if not neighbors:
            return self.vertex, 1.0

        savings = np.array([self.fig.savings(self.vertex, j) for j in neighbors],
                           dtype=float)

        if other is not None:
            # Modify weights by interaction with the other clock
            other_empire = other.empire
            interaction_mod = self._interaction_modifier(other)

            for k, j in enumerate(neighbors):
                # How much does moving to j overlap with other's empire?
                j_empire = self.fig.empire[j]
                overlap = len(j_empire & other_empire)

                # Same chirality: overlap is attractive (bonus)
                # Opposite chirality: overlap is repulsive (penalty)
                if interaction_mod > 0:
                    savings[k] += overlap * interaction_mod * 0.1
                else:
                    savings[k] -= overlap * abs(interaction_mod) * 0.1

        # Savings-weighted probability
        weights = (np.maximum(savings, 0) + 1) ** self.isv.savings_exponent
        probs = weights / weights.sum()

        chosen_idx = np.random.choice(len(neighbors), p=probs)
        return neighbors[chosen_idx], float(probs[chosen_idx])

    def _interaction_modifier(self, other: 'CycleClock') -> float:
        """Compute interaction strength based on quantum number matching."""
        mod = self.isv.coupling_strength

        # Chirality coupling
        if self.isv.chirality == other.isv.chirality:
            mod *= self.isv.chirality_coupling
        else:
            mod *= -self.isv.chirality_coupling

        # Generation resonance
        if self.isv.generation == other.isv.generation:
            mod *= 1.5

        return mod

    def step(self, other: 'CycleClock' = None) -> Dict:
        """Full step: tick internal clock + walk on FIG.

        Returns the step record.
        """
        # 1. Tick the Coxeter circuit
        self.tick()

        # 2. Choose and walk
        chosen, prob = self.choose_neighbor(other)
        old_vertex = self.vertex
        self.vertex = chosen

        # 3. Record
        self._record()

        return {
            'from': old_vertex,
            'to': chosen,
            'probability': prob,
            'savings': self.fig.savings(old_vertex, chosen),
            'chirality': self.isv.chirality,
            'generation': self.isv.generation,
            'phase': self.coxeter_phase,
        }


class CycleClockGame:
    """The full simulation: one or two cycle clocks on the FIG.

    Implements:
      - Solo walk (self-interaction via own empire)
      - Two to tango (mutual empire interaction)
      - History tracking for visualization
      - JSON export for Three.js
    """

    def __init__(self):
        print("Building FIG gameboard...")
        self.fig = FIGGameboard()
        print("Building E8 backbone...")
        self.e8 = E8Gameboard()
        self.clocks: List[CycleClock] = []
        self.step_log: List[Dict] = []
        self.step_count = 0

    def add_clock(self, fig_vertex: int = None, coxeter_seed: int = 0,
                  isv: ClockISV = None) -> CycleClock:
        """Add a cycle clock to the game."""
        clock = CycleClock(
            self.fig, self.e8,
            fig_vertex=fig_vertex,
            coxeter_seed=coxeter_seed,
            isv=isv,
            clock_id=len(self.clocks),
        )
        self.clocks.append(clock)
        return clock

    def step(self) -> Dict:
        """Advance the game by one step."""
        self.step_count += 1
        records = []

        if len(self.clocks) == 1:
            rec = self.clocks[0].step()
            records.append(rec)

        elif len(self.clocks) >= 2:
            # Each clock steps with awareness of the other
            rec1 = self.clocks[0].step(other=self.clocks[1])
            rec2 = self.clocks[1].step(other=self.clocks[0])
            records.extend([rec1, rec2])

            # Compute interaction metrics
            overlap = len(self.clocks[0].empire & self.clocks[1].empire)
            d3 = np.linalg.norm(
                self.fig.pos_3d[self.clocks[0].vertex] -
                self.fig.pos_3d[self.clocks[1].vertex]
            )
            records.append({
                'empire_overlap': overlap,
                'distance_3d': float(d3),
                'chirality_match': self.clocks[0].isv.chirality == self.clocks[1].isv.chirality,
            })

        log_entry = {'step': self.step_count, 'records': records}
        self.step_log.append(log_entry)
        return log_entry

    def run(self, n_steps: int = 30, verbose: bool = False) -> List[Dict]:
        """Run the game for n steps."""
        for i in range(n_steps):
            entry = self.step()
            if verbose and i % max(1, n_steps // 10) == 0:
                self._print_status(entry)
        return self.step_log

    def _print_status(self, entry):
        """Print a concise status line."""
        s = entry['step']
        parts = []
        for c in self.clocks:
            sn = c.snapshot()
            chi = '+' if sn['chirality'] > 0 else '-'
            parts.append(f"C{c.clock_id}@v{sn['vertex']}(φ={sn['coxeter_phase']}"
                         f",χ={chi},L{sn['generation']},emp={sn['empire_size']})")
        status = ' | '.join(parts)

        extras = ''
        if len(self.clocks) >= 2 and len(entry['records']) > 2:
            inter = entry['records'][-1]
            extras = f" | overlap={inter['empire_overlap']}, d={inter['distance_3d']:.2f}"

        print(f"  [{s:>4}] {status}{extras}")

    def export_history(self, filepath: str = None) -> Dict:
        """Export complete game history for Three.js."""
        data = {
            'n_steps': self.step_count,
            'n_clocks': len(self.clocks),
            'board': {
                'n_vertices': self.fig.n_vertices,
                'origin_idx': self.fig.origin_idx,
            },
            'clocks': [
                {
                    'id': c.clock_id,
                    'history': c.history,
                    'isv': {
                        'savings_exponent': c.isv.savings_exponent,
                        'coupling_strength': c.isv.coupling_strength,
                        'chirality_coupling': c.isv.chirality_coupling,
                        'amplitude': c.isv.amplitude,
                    }
                }
                for c in self.clocks
            ],
            'step_log': self.step_log,
        }

        if filepath:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer,)): return int(obj)
                    if isinstance(obj, (np.floating,)): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super().default(obj)
            with open(filepath, 'w') as f:
                json.dump(data, f, cls=NpEncoder)

        return data


# === Self-test ===
if __name__ == '__main__':
    game = CycleClockGame()

    # === SOLO WALK ===
    print("\n" + "="*70)
    print("SOLO WALK: One cycle clock, self-interaction")
    print("="*70)
    c1 = game.add_clock(
        fig_vertex=game.fig.origin_idx,  # start at center emperor
        coxeter_seed=0,
        isv=ClockISV(savings_exponent=15.0),
    )
    print(f"Start: vertex {c1.vertex}, empire={c1.fig.empire_sizes[c1.vertex]}")

    game.run(30, verbose=True)

    # Analyze trajectory
    vertices_visited = [h['vertex'] for h in c1.history]
    unique_vertices = len(set(vertices_visited))
    perp_radii = [h['perp_radius'] for h in c1.history]
    savings_seq = [entry['records'][0]['savings']
                   for entry in game.step_log if entry['records']]

    print(f"\n  Vertices visited: {unique_vertices} unique out of {len(vertices_visited)}")
    print(f"  Perp radius range: {min(perp_radii):.3f} - {max(perp_radii):.3f}")
    print(f"  Savings per step: {min(savings_seq)}-{max(savings_seq)} "
          f"(mean {np.mean(savings_seq):.1f})")

    # Chirality sequence
    chi_seq = [h['chirality'] for h in c1.history]
    gen_seq = [h['generation'] for h in c1.history]
    print(f"  Chirality sequence: {''.join('+' if c>0 else '-' for c in chi_seq[:31])}")
    print(f"  Generation sequence: {''.join(str(g) for g in gen_seq[:31])}")

    # === TWO TO TANGO ===
    print("\n" + "="*70)
    print("TWO TO TANGO: Two cycle clocks, empire interaction")
    print("="*70)
    game2 = CycleClockGame()

    # Clock 1: center emperor, high savings exponent
    c1 = game2.add_clock(
        fig_vertex=game2.fig.origin_idx,
        coxeter_seed=0,
        isv=ClockISV(savings_exponent=12.0, chirality_coupling=0.5),
    )
    # Clock 2: offset vertex, different Coxeter seed
    c2 = game2.add_clock(
        fig_vertex=game2.fig.adjacency[game2.fig.origin_idx][5],
        coxeter_seed=50,
        isv=ClockISV(savings_exponent=12.0, chirality_coupling=0.5),
    )
    print(f"Clock 1: v{c1.vertex} (center), chirality {'+'if c1.isv.chirality>0 else '-'}")
    print(f"Clock 2: v{c2.vertex}, chirality {'+'if c2.isv.chirality>0 else '-'}")

    game2.run(60, verbose=True)

    # Analyze interaction
    overlaps = [entry['records'][-1]['empire_overlap']
                for entry in game2.step_log if len(entry['records']) > 2]
    distances = [entry['records'][-1]['distance_3d']
                 for entry in game2.step_log if len(entry['records']) > 2]
    chi_matches = [entry['records'][-1]['chirality_match']
                   for entry in game2.step_log if len(entry['records']) > 2]

    print(f"\n  Empire overlap: {min(overlaps)}-{max(overlaps)} (mean {np.mean(overlaps):.0f})")
    print(f"  3D distance: {min(distances):.2f}-{max(distances):.2f} (mean {np.mean(distances):.2f})")
    print(f"  Chirality matches: {sum(chi_matches)}/{len(chi_matches)}")

    # Export
    game2.export_history('game_history.json')
    print(f"\n  Exported to game_history.json")
    print("\nGame ready.")
