import re
import subprocess
import unittest

import libconf

from test.jams_integration_test import JamsIntegrationtest


def constrained_mc_config(*, temperature, solver):
    config = {
        "sim": {"seed": 24680},
        "materials": (
            {"name": "A", "moment": 2.0, "spin": [1.0, 0.0, 0.0]},
            {"name": "B", "moment": 1.0, "spin": [0.0, 1.0, 0.0]},
        ),
        "unitcell": {
            "parameter": 3.0e-10,
            "basis": (
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
            "positions": (
                ("A", [0.0, 0.0, 0.0]),
                ("B", [0.5, 0.0, 0.0]),
            ),
        },
        "lattice": {
            "size": [3, 2, 1],
            "periodic": [True, True, True],
        },
        "hamiltonians": (
            {"module": "applied-field", "field": [0.0, 0.0, 1.0]},
        ),
        "physics": {"module": "empty", "temperature": temperature},
        "solver": solver,
    }
    return libconf.dumps(config)


class TestConstrainedMonteCarlo(JamsIntegrationtest):
    def run_config(self, config, *, setup_only=False, check=True):
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            config,
            f"--output={self.temp_dir}",
        ]
        if setup_only:
            args.append("--setup-only")
        return subprocess.run(
            args,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_zero_wavevector_initializes_plane_constraints(self):
        config = constrained_mc_config(
            temperature=0.0,
            solver={
                "module": "monte-carlo-constrained-cpu",
                "max_steps": 1,
                "cmc_constraint_mode": "spin_spiral",
                "cmc_constraint_type": "magnetisation",
                "cmc_constraint_theta": 5.0,
                "cmc_constraint_phi": 0.0,
                "cmc_spiral_wavevector": [0.0, 0.0, 0.0],
                "cmc_spiral_axis": [0.0, 0.0, 1.0],
                "cmc_spiral_propagation_direction": 0,
            },
        )

        result = self.run_config(config, setup_only=True)
        self.assertIn("zero-wavevector plane constraint yes", result.stdout)
        self.assertIn("spiral propagation lattice direction 0", result.stdout)
        self.assertIn("constrained planes 3", result.stdout)
        self.assertEqual(result.stdout.count("4 magnetic spins"), 3)

    def test_finite_temperature_adaptation_freezes_after_burn_in(self):
        config = constrained_mc_config(
            temperature=300.0,
            solver={
                "module": "monte-carlo-constrained-cpu",
                "max_steps": 6,
                "output_write_steps": 100,
                "cmc_constraint_mode": "global",
                "cmc_constraint_type": "magnetisation",
                "cmc_constraint_theta": 5.0,
                "cmc_constraint_phi": 0.0,
                "move_angle_sigma": 0.01,
                "move_angle_adaptation": {
                    "enabled": True,
                    "target_acceptance": 0.1,
                    "interval_steps": 3,
                    "gain": 0.5,
                    "min_sigma": 1.0e-6,
                    "max_sigma": 0.1,
                    "burn_in_steps": 4,
                },
            },
        )

        result = self.run_config(config)
        self.assertIn("move_angle_adaptation: step 3", result.stdout)
        self.assertIn("move_angle_adaptation: step 4", result.stdout)
        self.assertNotIn("move_angle_adaptation: step 5", result.stdout)
        self.assertNotIn("move_angle_adaptation: step 6", result.stdout)
        frozen = re.findall(r"frozen production sigma ([0-9.eE+-]+)", result.stdout)
        self.assertEqual(len(frozen), 1)

    def test_finite_temperature_adaptation_rejects_missing_burn_in(self):
        config = constrained_mc_config(
            temperature=300.0,
            solver={
                "module": "monte-carlo-constrained-cpu",
                "max_steps": 6,
                "cmc_constraint_mode": "global",
                "cmc_constraint_type": "magnetisation",
                "cmc_constraint_theta": 5.0,
                "cmc_constraint_phi": 0.0,
                "move_angle_sigma": 0.01,
                "move_angle_adaptation": {
                    "enabled": True,
                    "target_acceptance": 0.1,
                    "interval_steps": 3,
                    "gain": 0.5,
                    "min_sigma": 1.0e-6,
                    "max_sigma": 0.1,
                },
            },
        )
        result = self.run_config(config, setup_only=True, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "move_angle_adaptation.burn_in_steps",
            result.stdout + result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
