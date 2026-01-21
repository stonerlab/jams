import os
import subprocess
import tempfile
import unittest
import libconf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
import time
from test.jams_integration_test import JamsIntegrationtest

GYRO = 2.0023193043625 * 0.0578838181 / 0.6582119569 # per Tesla per picosecond with same precision as JAMS

def make_single_spin_cfg(
        solver: str,
        dt_fs: float,
        s0,
        t_max_ps: float = 10.0,
        alpha: float = 0.1,
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
    output_steps = max(int(1.0 / dt_fs), 1)

    cfg = {
        "materials": (
            {
                "name": "A",
                "moment": 1.0,
                "alpha": alpha,
                "spin": s0,
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
            "t_max": t_max_ps * 1e-12,
        },
        "monitors": (
            {"module": "magnetisation", "output_steps": output_steps},
            {"module": "energy", "output_steps": output_steps},
        ),
        "physics": {
            "temperature": 0.0,
        },
        "sim" : {
            "seed": 1234567890,
        }
    }

    # libconf.dumps returns a libconfig-formatted string.
    return libconf.dumps(cfg)

def make_multi_spin_cfg(
        solver: str,
        dt_fs: float,
        temperature: float,
        s0,
        t_max_ps: float = 10.0,
        alpha: float = 0.1,
        field: float = 10.0,
        exchange: float = 0.0,
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
    output_steps = max(int(1.0 / dt_fs), 1)

    cfg = {
        "materials": (
            {
                "name": "A",
                "moment": 1.0,
                "alpha": alpha,
                "spin": s0,
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
            "size": [16, 16, 16],
            "periodic": [True, True, True],
        },
        "hamiltonians": (
            {
                "module": "applied-field",
                "field": [0.0, 0.0, field],
            },
        ),
        "solver": {
            "module": solver,
            "t_step": dt_fs * 1e-15,
            "t_max": t_max_ps * 1e-12,
        },
        "monitors": (
            {"module": "magnetisation", "output_steps": output_steps},
            {"module": "energy", "output_steps": output_steps},
            {"module": "boltzmann", "output_steps": output_steps, "delay_time": 5e-12},
        ),
        "physics": {
            "temperature": temperature,
        },
        "sim" : {
            "seed": 1234567890,
        }
    }

    if exchange > 0.0:
        cfg["hamiltonians"] = (
            {
                "module": "applied-field",
                "field": [0.0, 0.0, field],
            },
            {
                "module": "exchange",
                "energy_units": "meV",
                "interactions": (
                    ("A", "A", [0.0, 0.0, 1.0], exchange),
                ),
            },
        )

    # libconf.dumps returns a libconfig-formatted string.
    return libconf.dumps(cfg)

class TestIntegrator(JamsIntegrationtest):
    def run_single_spin_relaxation_case(self, solver: str, dt_fs: float) -> None:
        cfg = make_single_spin_cfg(solver, dt_fs, s0=[1.0, 0.0, 0.0])
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

        df['A_mx_exact'] = mx(df['time'], 0.1, 100.0)
        df['A_my_exact'] = my(df['time'], 0.1, 100.0)
        df['A_mz_exact'] = mz(df['time'], 0.1, 100.0)
        df['A_mx_delta'] = df['A_mx_exact'] - df['A_mx']
        df['A_my_delta'] = df['A_my_exact'] - df['A_my']
        df['A_mz_delta'] = df['A_mz_exact'] - df['A_mz']
        df['A_m_delta'] = 1.0 - df['A_m']

        max_delta = df[['A_mx_delta', 'A_my_delta', 'A_mz_delta', 'A_m_delta']].abs().max().max()

        dt_label = f"{dt_fs:.3f}".replace(".", "p")
        artifact_name = f"mag_{solver}_dt{dt_fs:.3f}fs.tsv"
        df.to_csv(self.artifact_dir / artifact_name, sep=" ", index=False, float_format="%.8e")

        plot_name = f"mag_deltas_{solver}_dt{dt_label}fs.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["A_mx_delta"], label="mx_delta")
        ax.plot(df["time"], df["A_my_delta"], label="my_delta")
        ax.plot(df["time"], df["A_mz_delta"], label="mz_delta")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("delta")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        self.assertLess(max_delta, 1e-4)

    def run_single_spin_precession_case(self, solver: str, dt_fs: float) -> None:
        cfg = make_single_spin_cfg(solver, dt_fs, s0 = [1.0, 0.0, 0.0], t_max_ps=10, alpha=0.0)
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

        def mx(t, H):
            omega = GYRO * H
            return np.cos(omega * t)

        def my(t, H):
            omega = GYRO * H
            return np.sin(omega * t)

        def mz(t, H):
            return 0.0

        df = pd.read_csv(mag_file, sep=r'\s+')

        df['A_mx_exact'] = mx(df['time'], 100.0)
        df['A_my_exact'] = my(df['time'], 100.0)
        df['A_mz_exact'] = mz(df['time'], 100.0)
        df['A_mx_delta'] = df['A_mx_exact'] - df['A_mx']
        df['A_my_delta'] = df['A_my_exact'] - df['A_my']
        df['A_mz_delta'] = df['A_mz_exact'] - df['A_mz']
        df['A_m_delta'] = 1.0 - df['A_m']

        max_delta = df[['A_mx_delta', 'A_my_delta', 'A_mz_delta', 'A_m_delta']].abs().max().max()

        dt_label = f"{dt_fs:.3f}".replace(".", "p")
        artifact_name = f"mag_{solver}_dt{dt_fs:.3f}fs.tsv"
        df.to_csv(self.artifact_dir / artifact_name, sep=" ", index=False, float_format="%.8e")

        plot_name = f"mag_deltas_{solver}_dt{dt_label}fs.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["A_mx_delta"], label="mx_delta")
        ax.plot(df["time"], df["A_my_delta"], label="my_delta")
        ax.plot(df["time"], df["A_mz_delta"], label="mz_delta")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("delta")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        self.assertLess(max_delta, 1e-4)

    def run_exchange_case(self, solver: str, dt_fs: float) -> None:

        temperatures = [1.0, 300.0, 800.0]
        for T in temperatures:
            cfg = make_multi_spin_cfg(solver, dt_fs, temperature=T, s0 = [0.0, 0.0, 1.0], t_max_ps=100, alpha=0.1, exchange=20.0)

            mag_file = os.path.join(self.temp_dir, "jams_mag.tsv")

            self.runJamsCfg(cfg)

            if os.path.exists(mag_file):
                artifact_name = f"mag_{solver}_dt{dt_fs:.3f}fs_T{T:.3f}K.tsv"
                shutil.copyfile(mag_file, self.artifact_dir / artifact_name)

            self.assertTrue(
                os.path.exists(mag_file),
                f"Expected output file not found: {mag_file}",
            )

    def run_boltzmann_case(self, solver: str, dt_fs: float) -> None:

        temperatures = [1.0, 10.0, 100.0]
        for T in temperatures:
            cfg = make_multi_spin_cfg(solver, dt_fs, temperature=T, s0 = '"random"', t_max_ps=200, alpha=0.1)

            blt_file = os.path.join(self.temp_dir, "jams_blt.tsv")
            mag_file = os.path.join(self.temp_dir, "jams_mag.tsv")

            if os.path.exists(blt_file):
                os.remove(blt_file)

            self.runJamsCfg(cfg)

            if os.path.exists(blt_file):
                artifact_name = f"blt_{solver}_dt{dt_fs:.3f}fs_T{T:.3f}K.tsv"
                shutil.copyfile(blt_file, self.artifact_dir / artifact_name)

            if os.path.exists(mag_file):
                artifact_name = f"mag_{solver}_dt{dt_fs:.3f}fs_T{T:.3f}K.tsv"
                shutil.copyfile(mag_file, self.artifact_dir / artifact_name)

            df = pd.read_csv(blt_file, sep=r'\s+')

            def boltzmann_prob(theta, mu, H, T):
                kB = 1.38064852e-23
                muB = 9.27400968e-24
                beta = 1 / (kB * T)
                a = beta * mu * H * muB
                # The analytical PDF
                return (a / (2 * np.sinh(a))) * np.exp(a * np.cos(theta)) * np.sin(theta)

            # Bin width in radians (2 degrees)
            d_theta = np.deg2rad(2.0)
            df['exact'] = boltzmann_prob(np.deg2rad(df['theta_deg']), 1.0, 10.0, T) * d_theta
            df['delta'] = df['exact'] - df['probability']

            df.to_csv(self.artifact_dir / f"blt_deltas_{solver}_dt{dt_fs:.3f}fs_T{T:.3f}K.tsv", sep=" ", index=False, float_format="%.8e")

            max_delta = df[['delta']].abs().max().max()

            artifact_name = f"blt_deltas_{solver}_dt{dt_fs:.3f}fs_T{T:.3f}K.png"
            fig, ax = plt.subplots()
            ax.plot(df["theta_deg"], df["delta"], label="delta")
            ax.set_xlabel("theta (rad)")
            ax.set_ylabel("delta")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.artifact_dir / artifact_name)
            plt.close(fig)

            self.assertLess(max_delta, 1e-3)



    # def test_single_spin_relaxation(self):
    #             self.keep_artifacts = True
    #             solvers = [
    #                 "llg-heun-gpu",
    #                 "llg-rk4-gpu",
    #                 "llg-simp-gpu",
    #                 "llg-rkmk2-gpu",
    #                 "llg-rkmk4-gpu"
    #             ]
    #             time_steps_fs = [20.0, 10.0, 5.0, 1.0]
    #
    #             for solver in solvers:
    #                 if solver.endswith("-gpu") and not self.enable_gpu:
    #                     continue
    #
    #                 for dt_fs in time_steps_fs:
    #                     with self.subTest(solver=solver, dt_fs=dt_fs):
    #                         start = time.perf_counter()
    #                         self.run_single_spin_relaxation_case(solver, dt_fs)
    #                         elapsed = time.perf_counter() - start
    #                         print(f"subtest relax solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")
    #
    # def test_single_spin_precession(self):
    #         self.keep_artifacts = True
    #         solvers = [
    #             "llg-heun-gpu",
    #             "llg-rk4-gpu",
    #             "llg-simp-gpu",
    #             "llg-rkmk2-gpu",
    #             "llg-rkmk4-gpu"
    #         ]
    #         time_steps_fs = [20.0, 10.0, 5.0, 1.0]
    #
    #         for solver in solvers:
    #             if solver.endswith("-gpu") and not self.enable_gpu:
    #                 continue
    #
    #             for dt_fs in time_steps_fs:
    #                 with self.subTest(solver=solver, dt_fs=dt_fs):
    #                     start = time.perf_counter()
    #                     self.run_single_spin_precession_case(solver, dt_fs)
    #                     elapsed = time.perf_counter() - start
    #                     print(f"subtest precession solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")

    # def test_boltzmann_distribution(self):
        # self.keep_artifacts = True
        # solvers = [
        #     "llg-heun-cpu",
        #     "llg-rkmk2-gpu",
        #     "llg-heun-gpu",
        #     "llg-simp-gpu",
        #     "llg-rkmk4-gpu",
        #     "llg-rk4-gpu",
        # ]
        # time_steps_fs = [10.0, 5.0, 1.0]
        #
        # for solver in solvers:
        #     if solver.endswith("-gpu") and not self.enable_gpu:
        #         continue
        #
        #     for dt_fs in time_steps_fs:
        #         with self.subTest(solver=solver, dt_fs=dt_fs):
        #             start = time.perf_counter()
        #             self.run_boltzmann_case(solver, dt_fs)
        #             elapsed = time.perf_counter() - start
        #             print(f"subtest boltzmann solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")

    def test_magnetisation(self):
        self.keep_artifacts = True
        solvers = [
            "llg-rkmk2-gpu",
            "llg-heun-gpu",
            "llg-simp-gpu",
            "llg-rkmk4-gpu",
            "llg-rk4-gpu",
        ]
        time_steps_fs = [20.0, 10.0, 5.0, 1.0, 0.1]

        for solver in solvers:
            if solver.endswith("-gpu") and not self.enable_gpu:
                continue

            for dt_fs in time_steps_fs:
                with self.subTest(solver=solver, dt_fs=dt_fs):
                    start = time.perf_counter()
                    self.run_exchange_case(solver, dt_fs)
                    elapsed = time.perf_counter() - start
                    print(f"subtest exchange solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
