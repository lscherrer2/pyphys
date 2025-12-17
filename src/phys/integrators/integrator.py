from abc import ABC, abstractmethod
from phys.particle import Particle
from phys.forces.engine import Engine
import astropy.units as u
from astropy.units import Quantity
import numpy as np

__all__ = ["Integrator"]


class Integrator(ABC):
    @staticmethod
    def forces(engines: list[Engine], particles: list[Particle]) -> Quantity:
        engine_forces = np.stack([e.interact(particles) for e in engines], axis=0)
        net_forces = np.sum(engine_forces, axis=0)
        return net_forces

    @staticmethod
    def accelerations(engines: list[Engine], particles: list[Particle]) -> Quantity:
        forces = Integrator.forces(engines, particles)
        masses = Quantity([particle.mass for particle in particles]).reshape(-1, 1)
        accels = forces / masses
        return accels

    @staticmethod
    @abstractmethod
    def integrate(
        engines: list[Engine],
        particles: list[Particle],
        timestep: u.Quantity,
    ): ...
