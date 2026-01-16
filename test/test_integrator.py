import os
import subprocess
import tempfile
import unittest
import libconf
import numpy as np
import pandas as pd
import shutil
from test.jams_integration_test import JamsIntegrationtest

GYRO = 2.0023193043625 * 0.0578838181 / 0.6582119569 # per Tesla per picosecond with same precision as JAMS

def make_cfg(
        solver: str,
        dt_fs: float
) -> str:
    """
    Parameters
    ----------
    solver : str
        Value for solver.module.
    tstat : str
        Value for solver.thermostat.
    dt_fs : float
        Timestep in femtoseconds (fs). The file uses seconds:
        t_step = dt_fs * 1e-15

    Returns
    -------
    str
        libconfig text.
    """
    output_steps = int(1.0 / dt_fs)

    cfg = {
        "materials": (
            {
                "name": "A",
                "moment": 1.0,
                "alpha": 0.1,
                "spin": [1.0, 0.0, 0.0],
            },
        ),
        "unitcell": {
            "parameter": 0.3e-9,
            "basis": (
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
            "positions": (
                ("A", [0.0, 0.0, 0.0]),
            ),
        },
        "lattice": {
            "size": [1, 1, 1],
            "periodic": [False, False, False],
        },
        "hamiltonians": (
            {
                "module": "applied-field",
                "field": [0.0, 0.0, 100.0],
            },
        ),
        "solver": {
            "module": solver,
            "t_step": dt_fs * 1e-15,
            "t_max": 10.0e-12,
        },
        "monitors": (
            {"module": "magnetisation", "output_steps": output_steps},
            {"module": "energy", "output_steps": output_steps},
        ),
        "physics": {
            "temperature": 0.0,
        },
    }

    # libconf.dumps returns a libconfig-formatted string.
    return libconf.dumps(cfg)


class TestIntegrator(JamsIntegrationtest):
    def run_single_spin_case(self, solver: str, dt_fs: float) -> None:
        cfg = make_cfg(solver, dt_fs)
        mag_file = os.path.join(self.temp_dir, "jams_mag.tsv")
        if os.path.exists(mag_file):
            os.remove(mag_file)

        self.runJamsCfg(cfg)

        # Path to the output file
        # 1) Check the file exists
        self.assertTrue(
        os.path.exists(mag_file),
        f"Expected output file not found: {mag_file}",
        )

        def mx(t, alpha, H):
            omega = GYRO * H
            tau = 1 / (alpha * GYRO * H)
            return np.cos(omega * t) / np.cosh(t / tau)

        def my(t, alpha, H):
            omega = GYRO * H
            tau = 1 / (alpha * GYRO * H)
            return np.sin(omega * t) / np.cosh(t / tau)

        def mz(t, alpha, H):
            tau = 1 / (alpha * GYRO * H)
            return np.tanh(t / tau)

        df = pd.read_csv(mag_file, sep=r'\s+')

        df['A_mx_delta'] = mx(df['time'], 0.1, 100.0) - df['A_mx']
        df['A_my_delta'] = my(df['time'], 0.1, 100.0) - df['A_my']
        df['A_mz_delta'] = mz(df['time'], 0.1, 100.0) - df['A_mz']
        df['A_m_delta'] = 1.0 - df['A_m']

        max_delta = df[['A_mx_delta', 'A_my_delta', 'A_mz_delta', 'A_m_delta']].abs().max().max()

        self.assertLess(max_delta, 1e-4)

        artifact_name = f"mag_{solver}_dt{dt_fs:.3f}fs.tsv"
        df.to_csv(self.artifact_dir / artifact_name, sep=" ", index=False)

    def test_single_spin(self):
        self.keep_artifacts = True
        solvers = [
            "llg-heun-cpu",
            "llg-heun-gpu",
        ]
        time_steps_fs = [1.0, 0.5]

        for solver in solvers:
            if solver.endswith("-gpu") and not self.enable_gpu:
                continue

            for dt_fs in time_steps_fs:
                with self.subTest(solver=solver, dt_fs=dt_fs):
                    self.run_single_spin_case(solver, dt_fs)


if __name__ == "__main__":
    unittest.main()
