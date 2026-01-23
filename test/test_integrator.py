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
            {"module": "magnetisation", "precision": 16, "output_steps": output_steps},
            {"module": "energy", "precision": 16, "output_steps": output_steps},
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
            {"module": "magnetisation", "precision": 16, "output_steps": output_steps},
            {"module": "energy", "precision": 16, "output_steps": output_steps},
            {"module": "boltzmann", "precision": 16, "output_steps": output_steps, "delay_time": 5e-12},
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

def vector_angle(v1_cols, v2_cols):
    """
    Calculate the angle between two sets of vectors stored in pandas/numpy arrays.

    Avoids acos domain being restricted to [-1, 1] by using the formula
    atan2(|v1 x v2|, v1 . v2)
    
    :param v1_cols: tuple/list of (x, y, z) components (e.g. [df['mx'], df['my'], df['mz']])
    :param v2_cols: tuple/list of (x, y, z) components (e.g. [df['mx_e'], df['my_e'], df['mz_e']])
    :return: array of angles in radians
    """
    x1, y1, z1 = v1_cols
    x2, y2, z2 = v2_cols

    # Vectorized dot product: v1 . v2
    dot = x1 * x2 + y1 * y2 + z1 * z2

    # Vectorized cross product magnitude: |v1 x v2|
    # (y1*z2 - z1*y2)^2 + (z1*x2 - x1*z2)^2 + (x1*y2 - y1*x2)^2
    cross_mag = np.sqrt(
        (y1 * z2 - z1 * y2)**2 +
        (z1 * x2 - x1 * z2)**2 +
        (x1 * y2 - y1 * x2)**2
    )

    # atan2(y, x) is numerically stable and avoids acos domain errors
    return np.arctan2(cross_mag, dot)

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

        if os.path.exists(mag_file):
            artifact_name = f"mag_{solver}_dt{dt_fs:.3f}fsK.tsv"
            shutil.copyfile(mag_file, self.artifact_dir / artifact_name)

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

        df = pd.read_csv(mag_file, sep=r'\s+', comment="#")

        df['A_mx_exact'] = mx(df['time'], 0.1, 100.0)
        df['A_my_exact'] = my(df['time'], 0.1, 100.0)
        df['A_mz_exact'] = mz(df['time'], 0.1, 100.0)
        df['A_mx_delta'] = df['A_mx_exact'] - df['A_mx']
        df['A_my_delta'] = df['A_my_exact'] - df['A_my']
        df['A_mz_delta'] = df['A_mz_exact'] - df['A_mz']
        df['norm_error'] = np.abs(df['A_m']*df['A_m'] - 1.0)
        angular_projection = vector_angle((df['A_mx'], df['A_my'], df['A_mz']), (df['A_mx_exact'], df['A_my_exact'], df['A_mz_exact']))
        df['angular_err'] = np.rad2deg(angular_projection)

        max_norm_err = df['norm_error'].abs().max().max()
        max_angular_err = df['angular_err'].abs().max().max()

        dt_label = f"{dt_fs:.3f}".replace(".", "p")

        artifact_name = f"{solver}_dt{dt_fs:.3f}fs_mag.tsv"
        df.to_csv(self.artifact_dir / artifact_name, sep=" ", index=False, float_format="%.8e")

        plot_name = f"{solver}_dt{dt_label}fs_angular_error.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["angular_err"])
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("angular error (deg)")
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        plot_name = f"{solver}_dt{dt_label}fs_norm_error.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["norm_error"])
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("norm error (dimensionless)")
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        self.assertLess(max_norm_err, 1e-8)
        self.assertLess(max_angular_err, 0.01)


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

        df = pd.read_csv(mag_file, sep=r'\s+', comment="#")

        df['A_mx_exact'] = mx(df['time'], 100.0)
        df['A_my_exact'] = my(df['time'], 100.0)
        df['A_mz_exact'] = mz(df['time'], 100.0)
        df['A_mx_delta'] = df['A_mx_exact'] - df['A_mx']
        df['A_my_delta'] = df['A_my_exact'] - df['A_my']
        df['A_mz_delta'] = df['A_mz_exact'] - df['A_mz']
        df['norm_error'] = np.abs(df['A_m']*df['A_m'] - 1.0)

        omega = GYRO * 100.0  # rad / ps (given your time column is in ps)
        u = df["A_mx"].to_numpy() + 1j * df["A_my"].to_numpy()

        # Rotate into the co-rotating frame
        u_corot = u * np.exp(-1j * omega * df["time"].to_numpy())

        # Phase error in radians, wrapped to (-pi, pi]
        phase_err = np.angle(u_corot)

        df["phase_err_deg"] = np.rad2deg(phase_err)

        # unwrap to avoid jumps at -pi/pi boundaries
        phase_err_unwrapped = np.unwrap(phase_err)  # radians, continuous
        df["phase_err_unwrapped_deg"] = np.rad2deg(phase_err_unwrapped)

        # --- Drift + scatter metrics (phase error = intercept + slope*time + residual)
        t = df["time"].to_numpy(dtype=float)
        y = df["phase_err_unwrapped_deg"].to_numpy(dtype=float)

        # Linear least-squares fit y ≈ a + b t
        A = np.vstack([np.ones_like(t), t]).T
        a_deg, b_deg_per_ps = np.linalg.lstsq(A, y, rcond=None)[0]

        df["phase_err_fit_deg"] = a_deg + b_deg_per_ps * t
        df["phase_err_resid_deg"] = y - df["phase_err_fit_deg"].to_numpy(dtype=float)

        # Drift rate (systematic bias) and scatter (oscillations around drift)
        resid = df["phase_err_resid_deg"].to_numpy(dtype=float)
        scatter_rms_deg = float(np.sqrt(np.mean(resid * resid)))
        scatter_mad_deg = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
        scatter_max_abs_deg = float(np.max(np.abs(resid)))

        # Store as constant columns for convenient TSV export
        df["phase_drift_deg_per_ps"] = b_deg_per_ps
        df["phase_scatter_rms_deg"] = scatter_rms_deg
        df["phase_scatter_mad_deg"] = scatter_mad_deg
        df["phase_scatter_max_abs_deg"] = scatter_max_abs_deg

        max_norm_err = df['norm_error'].abs().max().max()
        phase_drift_err = b_deg_per_ps

        dt_label = f"{dt_fs:.3f}".replace(".", "p")

        artifact_name = f"{solver}_dt{dt_fs:.3f}fs_mag.tsv"
        df.to_csv(self.artifact_dir / artifact_name, sep=" ", index=False, float_format="%.8e")

        plot_name = f"{solver}_dt{dt_label}fs_phase_error.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["phase_err_unwrapped_deg"], label="unwrapped")
        ax.plot(df["time"], df["phase_err_fit_deg"], label="drift fit")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("phase error (deg)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        plot_name = f"{solver}_dt{dt_label}fs_phase_residual.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["phase_err_resid_deg"], label="residual")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("phase residual (deg)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        plot_name = f"{solver}_dt{dt_label}fs_norm_error.png"
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["norm_error"])
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("norm error (dimensionless)")
        fig.tight_layout()
        fig.savefig(self.artifact_dir / plot_name)
        plt.close(fig)

        self.assertLess(max_norm_err, 1e-8)
        self.assertLess(phase_drift_err, 0.01) # deg / ps

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

            df = pd.read_csv(blt_file, sep=r'\s+', comment="#")

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
            fig.tight_layout()
            fig.savefig(self.artifact_dir / artifact_name)
            plt.close(fig)

            self.assertLess(max_delta, 1e-3)



    def test_single_spin_relaxation(self):
        self.keep_artifacts = True
        solvers = [
            "llg-rkmk2-gpu",
            "llg-simp-gpu",
            "llg-heun-gpu",
            "llg-rkmk4-gpu",
            "llg-rk4-gpu",
        ]
        time_steps_fs = [5.0, 1.0, 0.5, 0.1]

        for solver in solvers:
            if solver.endswith("-gpu") and not self.enable_gpu:
                continue

            for dt_fs in time_steps_fs:
                with self.subTest(solver=solver, dt_fs=dt_fs):
                    start = time.perf_counter()
                    self.run_single_spin_relaxation_case(solver, dt_fs)
                    elapsed = time.perf_counter() - start
                    print(f"subtest relax solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")

    def test_single_spin_precession(self):
        self.keep_artifacts = True
        solvers = [
            "llg-rkmk2-gpu",
            "llg-simp-gpu",
            "llg-heun-gpu",
            "llg-rkmk4-gpu",
            "llg-rk4-gpu",
        ]
        time_steps_fs = [5.0, 1.0, 0.5, 0.1]

        for solver in solvers:
            if solver.endswith("-gpu") and not self.enable_gpu:
                continue

            for dt_fs in time_steps_fs:
                with self.subTest(solver=solver, dt_fs=dt_fs):
                    start = time.perf_counter()
                    self.run_single_spin_precession_case(solver, dt_fs)
                    elapsed = time.perf_counter() - start
                    print(f"subtest precession solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")

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

    # def test_magnetisation(self):
    #     self.keep_artifacts = True
    #     solvers = [
    #         "llg-rkmk2-gpu",
    #         "llg-heun-gpu",
    #         "llg-simp-gpu",
    #         "llg-rkmk4-gpu",
    #         "llg-rk4-gpu",
    #     ]
    #     time_steps_fs = [20.0, 10.0, 5.0, 1.0, 0.1]
    #
    #     for solver in solvers:
    #         if solver.endswith("-gpu") and not self.enable_gpu:
    #             continue
    #
    #         for dt_fs in time_steps_fs:
    #             with self.subTest(solver=solver, dt_fs=dt_fs):
    #                 start = time.perf_counter()
    #                 self.run_exchange_case(solver, dt_fs)
    #                 elapsed = time.perf_counter() - start
    #                 print(f"subtest exchange solver={solver} dt_fs={dt_fs} elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
