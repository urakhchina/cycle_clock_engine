"""
CycleClock: a kinematic spinor on the FIG gameboard.

Carries both spatial position (FIG vertex) and internal state
(Coxeter phase, chirality, generation, gauge).
"""

from dataclasses import dataclass


@dataclass
class ISVParams:
    """Intrinsic State Variables — tunable parameters."""
    savings_exponent: float = 28.0    # PEL strength (default matches old code)
    coupling_strength: float = 1.0
    chirality_coupling: float = 0.5
    amplitude: float = 1.0


class CycleClock:
    """A cycle clock walking on the FIG."""

    def __init__(self, e8_algebra, fig_vertex, coxeter_seed=0,
                 isv=None, clock_id=0):
        self.e8 = e8_algebra
        self.vertex = fig_vertex
        self.coxeter_seed = coxeter_seed
        self._coxeter_root = coxeter_seed
        self.coxeter_phase = 0
        self.isv = isv or ISVParams()
        self.clock_id = clock_id

        self._sync_quantum_numbers()
        self.history = [self.snapshot()]

    def _sync_quantum_numbers(self):
        qn = self.e8.quantum_numbers(self._coxeter_root)
        self.chirality = qn['chirality']
        self.generation = qn['coset']
        self.gauge_phase = qn['pair']
        self.fiber = qn['fiber']

    def tick(self):
        """Advance Coxeter circuit by one step."""
        self._coxeter_root = self.e8.coxeter_perm[self._coxeter_root]
        self.coxeter_phase = (self.coxeter_phase + 1) % 30
        self._sync_quantum_numbers()

    def walk(self, new_vertex):
        """Move to a new FIG vertex."""
        self.vertex = new_vertex

    def snapshot(self):
        return {
            'clock_id': self.clock_id,
            'vertex': self.vertex,
            'coxeter_phase': self.coxeter_phase,
            'coxeter_root': int(self._coxeter_root),
            'fiber': self.fiber,
            'chirality': self.chirality,
            'generation': self.generation,
            'gauge_phase': self.gauge_phase,
        }

    def record(self):
        self.history.append(self.snapshot())
