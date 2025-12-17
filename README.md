
# phys

`phys` is a small particle simulation toolkit that lets you define particles and their properties (
mass/charge, position, velocity, etc.), and simulate them with one or more force "engines" and a 
time integrator.

Core ideas:

- **Particle** state (`mass`, `charge`, `position`, `velocity`) uses **Astropy quantities**.
- **Engines** calculate forces on particles (e.g. gravitational attraction).
- **Integrators** use engine forces to advance particle positions.
- **Simulations** coordinate stepping, snapshots, and plotting. 

## Install

From the repo root:

```bash
pip install -e .
```

### Recommended: `uv`

This repo is designed with the `uv` package manager/runner. To reproduce results, please use it.

On macOS you can install it with either:

```bash
brew install uv
```

or

```bash
pipx install uv
```

Runtime dependencies (from `pyproject.toml`): `numpy`, `astropy`, `tqdm`.

### Optional plotting dependency

`Simulation.plot()` uses Plotly (imported at call time). If you want plotting:

```bash
pip install plotly
```

## Quickstart

This is a minimal two-body gravity example.

```python
from phys import Gravity, Simulation, Particle, Yoshida4
from astropy.units import kg, m, s, C

sun = Particle(
	mass=1.989e30 << kg,
	charge=0.0 << C,
	position=[0, 0, 0] << m,
	velocity=[0, 0, 0] << m / s,
)
earth = Particle(
	mass=5.972e24 << kg,
	charge=0.0 << C,
	position=[1.496e11, 0, 0] << m,
	velocity=[0, 29_780, 0] << m / s,
)

sim = Simulation(
	engines=[Gravity()],
	particles=[sun, earth],
	integrator=Yoshida4(),
)

sim.simulate(
	sim_time=10.0 << s,
	timestep=0.1 << s,
	record=True,
	verbose=True,
)

# Optional (requires plotly)
sim.plot()
```

### Notes on units

This project uses Astropy’s quantity system throughout.

- Use the `<<` operator to attach units: `1.0 << kg`, `[1, 0, 0] << m`, `0.01 << s`.
- `Simulation.simulate()` expects `sim_time` and `timestep` as quantities.

## Included scripts

From the repo root:

```bash
python scripts/three-body.py
python scripts/earth-sun.py
```

- `scripts/three-body.py` simulates a simple 3-body setup and plots trajectories.
- `scripts/earth-sun.py` simulates an approximate Earth-Sun-like system.

## API overview

### `Particle`

Create particles with mass/charge and initial position/velocity:

```python
from phys import Particle
import astropy.units as u

p = Particle(
	mass=1.0 << u.kg,
	charge=0.0 << u.C,
	position=[0, 0, 0] << u.m,
	velocity=[1, 0, 0] << (u.m / u.s),
)
```

### Forces: `Engine` and `Gravity`

An engine produces forces on each particle. The built-in engine is:

- `Gravity(G=...)` in `phys.forces.gravity`.

To create your own force model, subclass `phys.forces.engine.Engine` and implement `force(self, particle, effector)`.
If you need special handling (e.g., a uniform external field), override `interact(self, particles)` to return an `(N, 3)` force array.

### Integrators

Integrators update particle state from forces:

- `Euler()`
- `Leapfrog()`
- `Yoshida4()`

### `Simulation`

Create a `Simulation(engines, particles, integrator)` and run:

```python
sim.simulate(sim_time=10.0 << u.s, timestep=0.01 << u.s, record=True)
data = sim.data  # list[dict] snapshots
```

If `record=True`, snapshots include `time` and each particle position keyed like `"Particle 0"`, `"Particle 1"`, etc.

## Running tests

Tests are plain Python unittest files under `test/`. A simple runner script is included:

```bash
./tool/test
```

This uses `uv`.

