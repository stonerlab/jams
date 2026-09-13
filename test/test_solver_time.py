"""End-to-end checks for solver time and regular monitor sampling."""

from pathlib import Path
import re
import shutil
import subprocess

from test.jams_integration_test import JamsIntegrationtest


class SolverTimeIntegrationTest(JamsIntegrationtest):
    def test_monitor_times_and_sample_counts(self):
        solvers = ["llg-heun-cpu", "llg-rkmk2-cpu", "llg-rkmk4-cpu"]
        if self.enable_gpu:
            solvers += ["llg-heun-gpu", "llg-rkmk2-gpu", "llg-rkmk4-gpu",
                        "llg-rk4-gpu", "llg-dm-gpu", "llg-simp-gpu"]
        for solver in solvers:
            for stride in (1, 2):
                with self.subTest(solver=solver, stride=stride):
                    config = """
                    materials = ({name = "A"; moment = 1.0; alpha = 0.1;
                                  spin = [1.0, 0.0, 0.0];});
                    unitcell = {parameter = 3e-10;
                      basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
                      positions = (("A", [0.0, 0.0, 0.0]));};
                    lattice = {size = [1, 1, 1]; periodic = [false, false, false];};
                    physics = {temperature = 0.0;};
                    hamiltonians = ({module = "applied-field"; field = [0.0, 0.0, 1.0];});
                    """ + f"""
                    solver = {{module = "{solver}"; t_step = 1e-16; t_max = 3e-16;}};
                    monitors = (
                      {{module = "magnetisation"; precision = 3; output_steps = {stride};}},
                      {{module = "energy"; precision = 3; output_steps = {stride};}}
                    );
                    """
                    self.runJamsCfg(config)
                    times = []
                    for monitor in ("magnetisation", "energy"):
                        path, = Path(self.temp_dir).glob(f"*_{monitor}.tsv")
                        rows = path.read_text().splitlines()[2:]
                        times.append([float(row.split()[0]) for row in rows])
                    expected = [n * (1e-16 / 1e-12) for n in range(0, 3, stride)]
                    self.assertEqual(times[0], expected)
                    self.assertEqual(times[1], expected)

    def test_cuda_spin_current_time_metadata(self):
        if not self.enable_gpu:
            self.skipTest("CUDA integration tests are disabled")
        h5dump = shutil.which("h5dump")
        if not h5dump:
            self.skipTest("h5dump is required to inspect HDF5 time attributes")
        self.runJamsCfg("""
          materials = ({name = "A"; moment = 1.0; alpha = 0.1;
                        spin = [1.0, 0.0, 0.0];});
          unitcell = {parameter = 3e-10;
            basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
            positions = (("A", [0.0, 0.0, 0.0]));};
          lattice = {size = [2, 1, 1]; periodic = [false, false, false];};
          physics = {temperature = 0.0;};
          hamiltonians = ({module = "exchange";
            interactions = (("A", "A", [1.0, 0.0, 0.0], 1.0));});
          solver = {module = "llg-heun-gpu"; t_step = 1e-16; t_max = 3e-16;};
          monitors = ({module = "spin-current"; output_steps = 1; h5 = true;});
        """)
        xdmf_path, = Path(self.temp_dir).glob("*.xdmf")
        xdmf_times = [float(t) for t in re.findall(r'<Time Value="([^"]+)"', xdmf_path.read_text())]
        snapshots = sorted(Path(self.temp_dir).glob("*.h5"))
        expected_times = [n * (1e-16 / 1e-12) for n in range(3)]
        self.assertEqual(xdmf_times, expected_times)
        self.assertEqual(len(snapshots), 3)
        for iteration, path in enumerate(snapshots):
            for attribute, expected in (("time", expected_times[iteration]), ("iteration", iteration)):
                dumped = subprocess.check_output(
                    [h5dump, "-m", "%.17g", "-a", f"/spin_current/{attribute}", str(path)], text=True)
                match = re.search(r"\(0\):\s*([^\s}]+)", dumped)
                self.assertIsNotNone(match, dumped)
                self.assertEqual(float(match[1]), expected)
