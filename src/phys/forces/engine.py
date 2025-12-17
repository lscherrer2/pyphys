from abc import ABC, abstractmethod
from phys.particle import Particle
import numpy as np
from astropy.units import Quantity, N

__all__ = ["Engine"]


class Engine(ABC):
    symmetric: bool = False

    @abstractmethod
    def force(self, particle: Particle, effector: Particle) -> Quantity:
        pass

    def _symmetric_interact(self, particles: list[Particle]) -> Quantity:
        count = len(particles)
        force_matrix = np.zeros((count, count, 3)) << N
        for p, particle in enumerate(particles):
            for e, effector in enumerate(particles[p + 1 :], p + 1):
                pe_force = self.force(particle, effector)
                force_matrix[p, e] = pe_force
                force_matrix[e, p] = -1.0 * pe_force
        net_forces = np.sum(force_matrix, axis=1)
        return net_forces

    def _asymmetric_interact(self, particles: list[Particle]) -> Quantity:
        count = len(particles)
        force_matrix = np.zeros((count, count, 3)) << N
        for p, particle in enumerate(particles):
            for e, effector in enumerate(particles):
                if particle is effector:
                    continue
                pe_force = self.force(particle, effector)
                force_matrix[p, e] = pe_force
        net_forces = np.sum(force_matrix, axis=1)
        return net_forces

    def interact(self, particles: list[Particle]) -> Quantity:
        return (
            self._symmetric_interact(particles)
            if self.symmetric
            else self._asymmetric_interact(particles)
        )
