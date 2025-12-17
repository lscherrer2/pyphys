from phys import Gravity, Simulation, Particle, Yoshida4
from astropy.units import kg, m, s, C

# Solar system parameters
sun_mass = 1.989e30 << kg
earth_mass = 5.972e24 << kg
au = 1.496e11 << m
earth_orbital_velocity = 29780 << m / s

# Sun at the origin (stationary for simplicity)
sun = Particle(
    mass=sun_mass,
    charge=0.0 << C,
    position=[0, 0, 0] << m,
    velocity=[0, 0, 0] << m / s,
)
# Earth at 1 AU distance with orbital velocity
earth = Particle(
    mass=earth_mass,
    charge=0.0 << C,
    position=[au.value, 0, 0] << m,
    velocity=[0, earth_orbital_velocity.value, 0] << m / s,
)
particles = [sun, earth]


# Set up and run the simulation
sim = Simulation(
    engines=[Gravity()],
    particles=particles,
    integrator=Yoshida4(),
)

# Simulate for one Earth year (365.25 days)
earth_year = 365.25 * 24 * 3600 << s  # seconds in a year
sim.simulate(
    sim_time=2 * earth_year,
    timestep=3600 << s,  # 1 hour timestep
    record=True,
    verbose=True,
)

# Plot particle paths
sim.plot()
