from phys import forces, Simulation, Particle, integrators
from astropy.units import kg, m, s, C

# Three particles
particles = [
    Particle(
        mass=1.0 << kg,
        charge=0.0 << C,
        position=[-1, 0, 0] << m,
        velocity=[0, -0.5, 0] << m / s,
    ),
    Particle(
        mass=1.0 << kg,
        charge=0.0 << C,
        position=[1, 0, 0] << m,
        velocity=[0, 0.5, 0] << m / s,
    ),
    Particle(
        mass=1.0 << kg,
        charge=0.0 << C,
        position=[1, 1, 0] << m,
        velocity=[0.28, 0.5, -0.38] << m / s,
    ),
]

# Set up and run the simulation
sim = Simulation(
    engines=[forces.Gravity(G=1.0)],
    particles=particles,
    integrator=integrators.Yoshida4(),
)
sim.simulate(
    sim_time=10.0 << s,
    timestep=0.001 << s,
    record=True,
    verbose=True,
)

# Plot particle paths
sim.plot()
