from unittest import TestCase
from phys.integrators.integrator import Integrator
from phys.integrators.euler import Euler
from phys.forces.gravity import Gravity
from phys.forces.engine import Engine
from phys.particle import Particle
import astropy.units as u
import numpy as np
import numpy.testing as npt


class ConstantFieldEngine(Engine):
    """An engine that applies a constant external field (like gravity or electric field)."""

    symmetric = False

    def __init__(self, field_strength: float = 1.0):
        super().__init__()
        self.field_strength = field_strength

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        return np.array([0.0, 0.0, 0.0]) << u.N

    def interact(self, particles: list[Particle]) -> u.Quantity:
        """Apply constant field force to all particles."""
        forces = np.zeros((len(particles), 3)) << u.N
        for i, particle in enumerate(particles):
            forces[i] = (
                np.array([self.field_strength * particle.mass.value, 0.0, 0.0]) << u.N
            )
        return forces


class InteractionEngine(Engine):
    """An engine for testing particle-particle interactions."""

    symmetric = True

    def __init__(self, strength: float = 1.0):
        super().__init__()
        self.strength = strength

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        return np.array([self.strength, 0.0, 0.0]) << u.N


class MockSymmetricEngine(Engine):
    """A mock engine that applies constant force between particles."""

    symmetric: bool = True

    def __init__(self, force_magnitude: float = 1.0):
        super().__init__()
        self.force_magnitude = force_magnitude

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        return np.array([self.force_magnitude, 0.0, 0.0]) << u.N


class MockAsymmetricEngine(Engine):
    """A mock engine that applies constant force to all particles independently."""

    symmetric: bool = False

    def __init__(self, force_magnitude: float = 1.0):
        super().__init__()
        self.force_magnitude = force_magnitude

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        return np.array([self.force_magnitude, 0.0, 0.0]) << u.N


class AsymmetricGravity(Engine):
    """Asymmetric version of gravity for comparison testing."""

    symmetric: bool = False

    def __init__(self, G: float = 1.0):
        super().__init__()
        self.G = G

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        r_vec = effector.position - particle.position
        r_mag = np.linalg.norm(r_vec)
        r_direction = (r_vec / r_mag).value
        f_mag = self.G * particle.mass.value * effector.mass.value / (r_mag.value**2)
        return f_mag * r_direction << u.N


class TestIntegrator(TestCase):
    def setUp(self):
        """Set up test particles."""
        self.particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        self.particle2 = Particle(
            mass=2.0 << u.kg,
            position=np.array([1.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 1.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )

    def test_forces_calculation_single_particle(self):
        """Test that the forces method correctly calculates net forces for single particle."""
        field_engine = ConstantFieldEngine(field_strength=1.0)
        engines = [field_engine]

        particle = Particle(
            mass=2.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle]

        forces = Integrator.forces(engines, particles)

        self.assertEqual(forces.shape, (1, 3))

        expected_force = np.array([2.0, 0.0, 0.0]) << u.N
        npt.assert_array_almost_equal(forces[0].value, expected_force.value)

    def test_forces_multiple_engines(self):
        """Test forces calculation with multiple engines."""
        engine1 = ConstantFieldEngine(field_strength=1.0)
        engine2 = ConstantFieldEngine(field_strength=0.5)
        engines = [engine1, engine2]

        particle = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle]

        forces = Integrator.forces(engines, particles)

        expected_force = np.array([1.5, 0.0, 0.0]) << u.N
        npt.assert_array_almost_equal(forces[0].value, expected_force.value)

    def test_forces_symmetric_engine(self):
        """Test forces calculation with symmetric interaction."""
        mock_engine = MockSymmetricEngine(force_magnitude=1.0)
        engines = [mock_engine]
        particles = [self.particle1, self.particle2]

        forces = Integrator.forces(engines, particles)

        self.assertEqual(forces.shape, (2, 3))

        expected_force1 = np.array([1.0, 0.0, 0.0]) << u.N
        expected_force2 = np.array([-1.0, 0.0, 0.0]) << u.N

        npt.assert_array_almost_equal(forces[0].value, expected_force1.value)
        npt.assert_array_almost_equal(forces[1].value, expected_force2.value)

    def test_forces_asymmetric_engine(self):
        """Test forces calculation with asymmetric interaction."""
        mock_engine = MockAsymmetricEngine(force_magnitude=1.0)
        engines = [mock_engine]
        particles = [self.particle1, self.particle2]

        forces = Integrator.forces(engines, particles)

        self.assertEqual(forces.shape, (2, 3))

        expected_force = np.array([1.0, 0.0, 0.0]) << u.N

        npt.assert_array_almost_equal(forces[0].value, expected_force.value)
        npt.assert_array_almost_equal(forces[1].value, expected_force.value)

    def test_forces_multiple_engines_symmetric(self):
        """Test forces calculation with multiple symmetric engines."""
        engine1 = MockSymmetricEngine(force_magnitude=1.0)
        engine2 = MockSymmetricEngine(force_magnitude=0.5)
        engines = [engine1, engine2]
        particles = [self.particle1, self.particle2]

        forces = Integrator.forces(engines, particles)

        expected_force1 = np.array([1.5, 0.0, 0.0]) << u.N
        expected_force2 = np.array([-1.5, 0.0, 0.0]) << u.N

        npt.assert_array_almost_equal(forces[0].value, expected_force1.value)
        npt.assert_array_almost_equal(forces[1].value, expected_force2.value)

    def test_euler_integration_simple(self):
        """Test Euler integration with a simple constant force."""
        field_engine = ConstantFieldEngine(field_strength=1.0)
        engines = [field_engine]

        particle = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle]

        initial_pos = particle.position.copy()
        initial_vel = particle.velocity.copy()

        timestep = 1.0 << u.s

        Euler.integrate(engines, particles, timestep)

        self.assertTrue("position" in particle.buffer)
        self.assertTrue("velocity" in particle.buffer)

        npt.assert_array_equal(particle.position.value, initial_pos.value)
        npt.assert_array_equal(particle.velocity.value, initial_vel.value)

        particle.flush_buffer()

        expected_velocity = np.array([1.0, 0.0, 0.0]) << (u.m / u.s)
        npt.assert_array_almost_equal(particle.velocity.value, expected_velocity.value)

        expected_position = np.array([0.0, 0.0, 0.0]) << u.m
        npt.assert_array_almost_equal(particle.position.value, expected_position.value)

    def test_euler_integration_with_initial_velocity(self):
        """Test Euler integration with non-zero initial velocity."""
        field_engine = ConstantFieldEngine(field_strength=2.0)
        engines = [field_engine]

        particle = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([3.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle]

        timestep = 0.5 << u.s

        Euler.integrate(engines, particles, timestep)
        particle.flush_buffer()

        expected_velocity = np.array([4.0, 0.0, 0.0]) << (u.m / u.s)
        npt.assert_array_almost_equal(particle.velocity.value, expected_velocity.value)

        expected_position = np.array([1.5, 0.0, 0.0]) << u.m
        npt.assert_array_almost_equal(particle.position.value, expected_position.value)

    def test_euler_integration_multiple_particles(self):
        """Test Euler integration with multiple particles."""
        field_engine = ConstantFieldEngine(field_strength=1.0)
        engines = [field_engine]

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=2.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        timestep = 1.0 << u.s

        Euler.integrate(engines, particles, timestep)
        for p in particles:
            p.flush_buffer()

        expected_velocity = np.array([1.0, 0.0, 0.0]) << (u.m / u.s)

        npt.assert_array_almost_equal(particle1.velocity.value, expected_velocity.value)
        npt.assert_array_almost_equal(particle2.velocity.value, expected_velocity.value)

    def test_gravity_integration(self):
        """Test integration with realistic gravity forces."""
        gravity_engine = Gravity(G=1.0)
        engines = [gravity_engine]

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([-0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        timestep = 0.1 << u.s

        Euler.integrate(engines, particles, timestep)
        for p in particles:
            p.flush_buffer()

        self.assertGreater(particle1.velocity[0].value, 0)
        self.assertLess(particle2.velocity[0].value, 0)

        npt.assert_almost_equal(
            abs(particle1.velocity[0].value), abs(particle2.velocity[0].value)
        )

    def test_euler_integration_with_gravity(self):
        """Test Euler integration with realistic gravity forces and buffering."""
        gravity_engine = Gravity(G=1.0)
        engines = [gravity_engine]

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([-0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        timestep = 0.1 << u.s

        initial_pos1 = particle1.position.copy()
        initial_pos2 = particle2.position.copy()

        Euler.integrate(engines, particles, timestep)

        self.assertTrue("position" in particle1.buffer)
        self.assertTrue("velocity" in particle1.buffer)
        self.assertTrue("position" in particle2.buffer)
        self.assertTrue("velocity" in particle2.buffer)

        npt.assert_array_equal(particle1.position.value, initial_pos1.value)
        npt.assert_array_equal(particle2.position.value, initial_pos2.value)

        for p in particles:
            p.flush_buffer()

        self.assertGreater(particle1.velocity[0].value, 0)
        self.assertLess(particle2.velocity[0].value, 0)

        npt.assert_almost_equal(
            abs(particle1.velocity[0].value), abs(particle2.velocity[0].value)
        )

    def test_euler_integration_buffering(self):
        """Test that Euler integration properly uses the buffering system."""
        gravity_engine = Gravity(G=1.0)
        engines = [gravity_engine]

        particle = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([1.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle]

        timestep = 0.1 << u.s
        initial_pos = particle.position.copy()
        initial_vel = particle.velocity.copy()

        Euler.integrate(engines, particles, timestep)

        self.assertTrue("position" in particle.buffer)
        self.assertTrue("velocity" in particle.buffer)

        npt.assert_array_equal(particle.position.value, initial_pos.value)
        npt.assert_array_equal(particle.velocity.value, initial_vel.value)

        expected_new_pos = initial_pos + initial_vel * timestep
        npt.assert_array_almost_equal(
            particle.buffer["position"].value, expected_new_pos.value
        )

    def test_symmetric_vs_asymmetric_engines(self):
        """Test that symmetric and asymmetric engines work correctly."""
        gravity_engine = Gravity(G=1.0)
        self.assertTrue(gravity_engine.symmetric)

        field_engine = ConstantFieldEngine(field_strength=1.0)
        self.assertFalse(field_engine.symmetric)

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([1.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        gravity_forces = gravity_engine.interact(particles)
        field_forces = field_engine.interact(particles)

        self.assertEqual(gravity_forces.shape, (2, 3))
        self.assertEqual(field_forces.shape, (2, 3))

    def test_gravity_symmetric_vs_asymmetric(self):
        """Test that gravity behaves symmetrically."""
        gravity_sym = Gravity(G=1.0)
        gravity_asym = AsymmetricGravity(G=1.0)

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([-0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        forces_sym = gravity_sym.interact(particles)
        forces_asym = gravity_asym.interact(particles)

        npt.assert_array_almost_equal(forces_sym.value, forces_asym.value, decimal=10)
