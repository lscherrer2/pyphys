from phys.forces.engine import Engine
from phys.particle import Particle
import astropy.units as u
import numpy as np

__all__ = ["Electromagnetism"]


class Electromagnetism(Engine):
    symmetric = True
    __slots__ = ("k",)

    def __init__(self, k: u.Quantity | float = 8.9875517923e9):
        self.k = (
            k.to(u.N * u.m**2 / u.C**2)
            if isinstance(k, u.Quantity)
            else k * (u.N * u.m**2 / u.C**2)
        )

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        r_vec: u.Quantity = effector.position - particle.position
        r_mag: u.Quantity = np.linalg.norm(r_vec)
        r_direction: u.Quantity = r_vec / r_mag

        f_mag: u.Quantity = self.k * particle.charge * effector.charge / (r_mag**2)
        f_mag = f_mag.to(u.N)

        f_vec = f_mag * r_direction
        return f_vec
