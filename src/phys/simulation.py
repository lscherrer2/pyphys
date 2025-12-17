from phys.particle import Particle
from phys.forces.engine import Engine
from phys.integrators.integrator import Integrator
import astropy.units as u
from tqdm import tqdm

__all__ = ["Simulation"]


class SimTimer:
    def __init__(self, start_time: u.Quantity, end_time: u.Quantity, step: u.Quantity):
        self.start = start_time
        self.time = self.start.copy()
        self.end = end_time
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.time >= self.end:
            raise StopIteration()
        prev_time = self.time
        next_time = min(self.time + self.step, self.end)
        self.time = next_time
        return next_time - prev_time

    def __len__(self):
        return int(self.end // self.step).__ceil__()


class Simulation:
    def __init__(
        self, engines: list[Engine], particles: list[Particle], integrator: Integrator
    ):
        self.engines = engines
        self.particles = particles
        self.integrator = integrator
        self.snapshots = []

    def simulate(
        self,
        sim_time: u.Quantity,
        timestep: u.Quantity,
        record: bool = True,
        verbose: bool = False,
    ):
        timer = SimTimer(0.0 << u.s, sim_time.to(u.s), timestep.to(u.s))
        if record:
            self.record(timer.time)
        timer_iterable = timer if not verbose else tqdm(timer, colour="green")
        for current_step in timer_iterable:
            self.step(current_step)
            if record:
                self.record(timer.time)

    def step(self, timestep: u.Quantity):
        self.integrator.integrate(self.engines, self.particles, timestep)
        for particle in self.particles:
            particle.flush_buffer()

    def record(self, time: u.Quantity):
        self.snapshots.append(
            {"time": time}
            | {
                f"Particle {particle.id}": particle.position
                for particle in self.particles
            }
        )

    @property
    def data(self) -> list[dict]:
        return self.snapshots

    def plot(self):
        import plotly.graph_objects as go
        from astropy import units as u

        fig = go.Figure()

        # Collect particle names (excluding 'time')
        particle_names = [key for key in self.data[0] if key != "time"]

        for name in particle_names:
            xs, ys, zs = [], [], []

            for entry in self.data:
                # Convert position to meters
                pos = entry[name].to(u.m)
                xs.append(pos[0].value)
                ys.append(pos[1].value)
                zs.append(pos[2].value)

            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name))

        fig.update_layout(
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
            title="Particle Trajectories",
            showlegend=True,
        )

        fig.show()
