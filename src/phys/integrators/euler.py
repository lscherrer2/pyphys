from phys.integrators.integrator import Integrator
from phys.particle import Particle
from phys.forces.engine import Engine
import astropy.units as u

__all__ = ["Euler"]


class Euler(Integrator):
    def integrate(
        engines: list[Engine],
        particles: list[Particle],
        timestep: u.Quantity,
    ):
        accels = Integrator.accelerations(engines, particles)
        delta_vs = accels * timestep

        for i, particle in enumerate(particles):
            particle.buffer["position"] = (
                particle.position + particle.velocity * timestep
            )
            particle.buffer["velocity"] = particle.velocity + delta_vs[i]
