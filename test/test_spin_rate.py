"""End-to-end deterministic spin-rate and stopping checks (optional CUDA)."""

import math
from pathlib import Path
import subprocess
import tempfile

from test.jams_integration_test import JamsIntegrationtest


class SpinRateIntegrationTest(JamsIntegrationtest):
    def solvers(self):
        names = ["llg-heun-cpu", "llg-rkmk2-cpu", "llg-rkmk4-cpu"]
        if self.enable_gpu:
            names += ["llg-heun-gpu", "llg-rkmk2-gpu", "llg-rkmk4-gpu", "llg-rk4-gpu"]
        return names

    @staticmethod
    def config(solver, monitor="", *, temperature=0.0, spin="[1.0, 0.0, 0.0]",
               alpha=0.1, gilbert=False, t_min=0.0, t_max=1e-14, t_step=1e-15,
               hamiltonians=None, material_extra="", solver_extra="", physics_extra="", size=1):
        if hamiltonians is None:
            hamiltonians = '({module = "applied-field"; field = [0.0, 0.0, 100.0];})'
        return f"""
          sim = {{seed = 1234567890;}};
          materials = ({{name = "A"; moment = 1.0; alpha = {alpha}; spin = {spin};
                        {material_extra}}});
          unitcell = {{parameter = 3e-10;
            basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
            positions = (("A", [0.0, 0.0, 0.0]));}};
          lattice = {{size = [{size}, 1, 1]; periodic = [false, false, false];}};
          physics = {{temperature = {temperature}; {physics_extra}}};
          hamiltonians = {hamiltonians};
          solver = {{module = "{solver}"; t_step = {t_step}; t_max = {t_max};
                     t_min = {t_min}; gilbert_prefactor = {str(gilbert).lower()}; {solver_extra}}};
          monitors = (
            {{module = "magnetisation"; output_steps = 1; precision = 17;}},
            {{module = "hdf5"; output_steps = 100000;}}
            {monitor}
          );
        """

    def run_case(self, config, success=True):
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as directory:
            result = subprocess.run(
                [self.binary_path, "--name=jams", "--config", config, f"--output={directory}"],
                capture_output=True, text=True)
            if success:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((Path(directory) / "monitor_hdf5_final.h5").exists())
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout)
            rows = {}
            for path in Path(directory).glob("*.tsv"):
                rows[path.stem.removeprefix("monitor_")] = [
                    [float(x) for x in line.split()] for line in path.read_text().splitlines()[2:]]
            return result, rows

    def test_rates_use_current_state_for_all_solvers(self):
        # A coarser timestep makes an RK4 final-stage rate distinguishable from
        # the rate of its accepted, normalised state. Check that state directly.
        for solver in self.solvers():
            for alpha in (0.0, 0.3):
                for gilbert in (False, True):
                    with self.subTest(solver=solver, alpha=alpha, gilbert=gilbert):
                        _, rows = self.run_case(self.config(
                            solver, ', {module = "spin-rate"; output_steps = 1; precision = 17;}',
                            alpha=alpha, gilbert=gilbert, t_step=1e-14, t_max=1e-13))
                        rates = rows["spin-rate"]
                        spins = rows["magnetisation"]
                        gyro = 2.0023193043625 * 0.0578838181 / 0.6582119569
                        initial = 100.0 * gyro * math.sqrt(1 + alpha*alpha)
                        if gilbert:
                            initial /= 1 + alpha*alpha
                        self.assertAlmostEqual(rates[0][1], initial, delta=3e-6 * initial)
                        # Normalise to the initial analytic rate, eliminating
                        # material unit constants and mixed-precision rounding.
                        for rate, spin in zip(rates, spins):
                            x, y, z = spin[1:4]
                            norm2 = x*x + y*y + z*z
                            expected = rates[0][1] * math.sqrt(
                                (x*x + y*y) * (1 + alpha*alpha*norm2) / (1 + alpha*alpha))
                            self.assertAlmostEqual(rate[1], expected, delta=3e-6 * rates[0][1])
                        self.assertGreater(rates[0][1], 0)

    def test_solver_agreement_and_gilbert_scaling(self):
        reference = None
        for solver in self.solvers():
            measured = []
            for gilbert in (False, True):
                _, rows = self.run_case(self.config(
                    solver, ', {module = "spin-rate"; output_steps = 1; precision = 17;}',
                    alpha=0.5, gilbert=gilbert))
                measured.append(rows["spin-rate"][0][1])
            self.assertAlmostEqual(measured[1] / measured[0], 1 / 1.25, delta=3e-7)
            if reference is None:
                reference = measured[0]
            self.assertAlmostEqual(measured[0], reference, delta=3e-6 * reference)

    def test_physics_spin_changes_refresh_fields_before_convergence(self):
        # Pinning restores the same spin before every sample, after RKMK has
        # cached its final unpinned spins. A spin-dependent field must be rebuilt
        # from the pinned state without the test manually invalidating that cache.
        hamiltonians = '''({module = "uniaxial"; order = "K1"; energy_units = "meV";
                            anisotropies = (("A", [0.0, 0.0, 1.0], -1.0));})'''
        gyro = 2.0023193043625 * 0.0578838181 / 0.6582119569
        # H_z = 2 K S_z / mu; |dS/dt| = |gamma H_z| S_perp sqrt(1 + alpha^2).
        expected = gyro * (2.0 * 0.8 / 0.0578838181) * 0.6 * math.sqrt(2.0)
        for solver in self.solvers():
            with self.subTest(solver=solver):
                result, rows = self.run_case(self.config(
                    solver, ', {module = "spin-rate"; output_steps = 1; convergence = 4.1; precision = 17;}',
                    spin="[0.6, 0.0, 0.8]", alpha=1.0, t_step=1e-14, t_max=5e-14,
                    hamiltonians=hamiltonians,
                    physics_extra='module = "pinned_boundaries"; left_pinned_magnetisation = [0.6, 0.0, 0.8];'))
                # A stale field gives about 4.037 instead of 4.130 ps^-1 and
                # incorrectly stops mixed-precision CUDA RKMK at iteration 1.
                self.assertNotIn("stopping: monitor convergence", result.stdout)
                self.assertEqual(len(rows["spin-rate"]), 5)
                self.assertEqual(len(rows["magnetisation"]), 5)
                for rate, spin in zip(rows["spin-rate"], rows["magnetisation"]):
                    for measured, pinned in zip(spin[1:4], (0.6, 0.0, 0.8)):
                        self.assertAlmostEqual(measured, pinned, delta=1e-12)
                    self.assertAlmostEqual(rate[1], expected, delta=3e-6 * expected)

    def test_stopping_minimum_cadence_and_finalisation(self):
        for solver in self.solvers():
            with self.subTest(solver=solver):
                result, rows = self.run_case(self.config(
                    solver, ', {module = "spin-rate"; output_steps = 2; convergence = 1e-4; consecutive_samples = 2;}',
                    spin="[0.0, 0.0, 1.0]", t_min=3e-15))
                self.assertIn("stopping: monitor convergence at iteration 6", result.stdout)
                self.assertEqual(len(rows["magnetisation"]), 7)
                self.assertEqual(len(rows["spin-rate"]), 4)
                result, rows = self.run_case(self.config(
                    solver, ', {module = "spin-rate"; convergence = 1e-4;}',
                    spin="[0.0, 0.0, 1.0]"))
                self.assertIn("stopping: monitor convergence at iteration 0", result.stdout)
                self.assertEqual(len(rows["magnetisation"]), 1)

    def test_relaxation_and_and_semantics(self):
        for solver in self.solvers():
            with self.subTest(solver=solver):
                monitor = ', {module = "spin-rate"; output_steps = 10; convergence = 1.0; consecutive_samples = 3;}'
                result, rows = self.run_case(self.config(solver, monitor, alpha=1.0, t_max=1e-12))
                self.assertIn("stopping: monitor convergence", result.stdout)
                self.assertGreater(len(rows["magnetisation"]), 1)
                self.assertLess(len(rows["magnetisation"]), 1000)
                self.assertTrue(all(row[1] < 1.0 for row in rows["spin-rate"][-3:]))
                result, rows = self.run_case(self.config(
                    solver, monitor + ', {module = "spin-rate"; name = "strict"; output_steps = 1; convergence = 1e-30;}',
                    alpha=1.0, t_max=1e-14))
                self.assertNotIn("stopping: monitor convergence", result.stdout)
                self.assertEqual(len(rows["magnetisation"]), 10)

    def test_diagnostic_does_not_change_thermal_or_deterministic_trajectories(self):
        # Exchange and a time-dependent field exercise field/cache refreshes.
        hamiltonians = '''(
          {module = "exchange"; energy_units = "meV";
           interactions = (("A", "A", [1.0, 0.0, 0.0], 1.0));},
          {module = "applied-field"; type = "sinc-cos"; field = [0.0, 0.0, 100.0];
           time_center = 0.0; freq_bandwidth = 1e12; freq_center = 2e12;})'''
        for solver in self.solvers():
            for temperature in (0.0, 50.0):
                # CLI sim.seed is applied after CUDA RNG initialisation in JAMS.
                # GPU thermal preservation is tested in-process with an explicit
                # cuRAND seed by SpinRateCudaTrajectories instead.
                if temperature > 0.0 and solver.endswith("-gpu"):
                    continue
                with self.subTest(solver=solver, temperature=temperature):
                    kwargs = dict(temperature=temperature, hamiltonians=hamiltonians, size=2)
                    _, baseline = self.run_case(self.config(solver, **kwargs))
                    _, measured = self.run_case(self.config(
                        solver, ', {module = "spin-rate"; output_steps = 1;}', **kwargs))
                    self.assertEqual(baseline["magnetisation"], measured["magnetisation"])

    def test_zero_field_and_unsupported_solvers(self):
        for solver in self.solvers():
            _, rows = self.run_case(self.config(
                solver, ', {module = "spin-rate"; output_steps = 1;}', hamiltonians="()"))
            self.assertTrue(all(row[1] == 0 for row in rows["spin-rate"]))
        if self.enable_gpu:
            result, _ = self.run_case(self.config(
                "llg-sot-rk4-gpu", ', {module = "spin-rate";}',
                solver_extra='spin_polarisation = [0.0, 1.0, 0.0]; spin_hall_angle = 0.1; charge_current_density = 1e12;'),
                success=False)
            self.assertIn("spin-rate is not supported", result.stderr + result.stdout)

    def test_spin_perturbation_initial_state_matches_all_solvers(self):
        reference = None
        for solver in self.solvers():
            with self.subTest(solver=solver):
                config = self.config(
                    solver, ', {module = "spin-rate"; output_steps = 1; precision = 17;}',
                    spin="[0.0, 0.0, 1.0]", t_max=1e-15)
                initializer = 'initializer = {module = "spin-perturbation"; amplitude = 0.001; seed = 1234;};'
                _, rows = self.run_case(config + initializer)
                _, repeat = self.run_case(config + initializer)
                initial = rows["magnetisation"][0][1:4]
                self.assertEqual(initial, repeat["magnetisation"][0][1:4])
                if reference is None:
                    reference = initial
                self.assertEqual(initial, reference)
                self.assertGreater(math.hypot(*initial[:2]), 0.0)
                self.assertLessEqual(math.hypot(*initial[:2]) / initial[2], 0.001 + 1e-14)
                self.assertAlmostEqual(sum(x*x for x in initial), 1.0, delta=1e-14)

    def test_spin_perturbation_releases_unstable_equilibrium(self):
        for solver in self.solvers():
            with self.subTest(solver=solver):
                config = self.config(
                    solver, ', {module = "spin-rate"; output_steps = 1; convergence = 1e-4;}',
                    spin="[0.0, 0.0, 1.0]", alpha=1.0, t_step=1e-14, t_max=2e-12,
                    hamiltonians='({module = "applied-field"; field = [0.0, 0.0, -100.0];})')
                result, _ = self.run_case(config)
                self.assertIn("stopping: monitor convergence at iteration 0", result.stdout)
                result, rows = self.run_case(config +
                    'initializer = {module = "spin-perturbation"; amplitude = 0.001; seed = 1234;};')
                self.assertNotIn("stopping: monitor convergence at iteration 0", result.stdout)
                self.assertIn("stopping: monitor convergence", result.stdout)
                self.assertGreater(rows["spin-rate"][0][1], 1e-4)
                self.assertGreater(rows["magnetisation"][0][3], 0.999)
                self.assertLess(rows["magnetisation"][-1][3], -0.999)
