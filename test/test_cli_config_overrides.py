import subprocess
from pathlib import Path

from test.jams_integration_test import JamsIntegrationtest


def make_minimal_cfg(include_second_hamiltonian: bool = False) -> str:
    second_hamiltonian = r"""
  {
    module = "exchange";
    interactions = (
      ("A", "A", [0.0, 0.0, 1.0], 1.0)
    );
  }
""" if include_second_hamiltonian else ""

    return r"""
materials = (
  {
    name = "A";
    moment = 1.0;
    alpha = 0.1;
    spin = [1.0, 0.0, 0.0];
  }
);
unitcell = {
  parameter = 3.0e-10;
  basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
  positions = (("A", [0.0, 0.0, 0.0]));
};
lattice = {
  size = [1, 1, 1];
  periodic = [false, false, false];
};
hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, 1.0];
  }
""" + ("," + second_hamiltonian if include_second_hamiltonian else "") + r"""
);
solver = {
  module = "llg-heun-cpu";
  t_step = 1.0e-15;
  t_max = 1.0e-15;
};
monitors = (
  {
    module = "magnetisation";
    output_steps = 1;
  }
);
physics = {
  temperature = 0.0;
};
sim = {
  seed = 1234567890;
};
"""


class TestCliConfigOverrides(JamsIntegrationtest):
    def load_combined_cfg(self):
        combined_cfg = Path(self.temp_dir) / "jams_combined.cfg"
        self.assertTrue(combined_cfg.exists(), f"Expected combined config at {combined_cfg}")
        return combined_cfg.read_text()

    def run_jams_with_args(self, args):
        result = subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result

    def run_jams_with_args_allow_failure(self, args):
        return subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_log_flag_redirects_stdout_to_file(self):
        log_path = Path(self.artifact_dir) / "stdout.log"
        result = self.run_jams_with_args([self.binary_path, "--help", f"--log={log_path}"])

        self.assertEqual("", result.stdout)
        self.assertEqual("", result.stderr)
        self.assertTrue(log_path.exists(), f"Expected log file at {log_path}")

        log_text = log_path.read_text()
        self.assertIn("Usage: jams", log_text)
        self.assertIn("--log=<path>", log_text)

    def test_log_flag_redirects_stderr_to_file(self):
        log_path = Path(self.artifact_dir) / "stderr.log"
        result = self.run_jams_with_args_allow_failure(
            [self.binary_path, f"--log={log_path}", "--unknown-flag"]
        )

        self.assertNotEqual(0, result.returncode)
        self.assertEqual("", result.stdout)
        self.assertEqual("", result.stderr)
        self.assertTrue(log_path.exists(), f"Expected log file at {log_path}")

        log_text = log_path.read_text()
        self.assertIn("Unknown flag '--unknown-flag'", log_text)
        self.assertIn("Usage: jams", log_text)

    def test_dotted_path_override_updates_nested_setting(self):
        cfg = make_minimal_cfg()
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            cfg,
            "--config",
            "physics.temperature = 100.0;",
            f"--output={self.temp_dir}",
            "--setup-only",
        ]

        self.run_jams_with_args(args)

        combined_cfg = self.load_combined_cfg()
        self.assertIn("physics = {", combined_cfg)
        self.assertIn("temperature = 100.0;", combined_cfg)

    def test_multiple_dotted_path_overrides_can_share_one_string(self):
        cfg = make_minimal_cfg()
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            cfg,
            "--config",
            'physics.temperature = 250.0; solver.module = "llg-heun-cpu";',
            f"--output={self.temp_dir}",
            "--setup-only",
        ]

        self.run_jams_with_args(args)

        combined_cfg = self.load_combined_cfg()
        self.assertIn("temperature = 250.0;", combined_cfg)
        self.assertIn('module = "llg-heun-cpu";', combined_cfg)

    def test_indexed_path_override_updates_specific_hamiltonian(self):
        cfg = make_minimal_cfg(include_second_hamiltonian=True)
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            cfg,
            "--config",
            "hamiltonians[1].module = \"applied-field\"; hamiltonians[1].field = [0.0, 0.0, 2.0];",
            f"--output={self.temp_dir}",
            "--setup-only",
        ]

        self.run_jams_with_args(args)

        combined_cfg = self.load_combined_cfg()
        self.assertIn('field = [ 0.0, 0.0, 1.0 ];', combined_cfg)
        self.assertIn('field = [ 0.0, 0.0, 2.0 ];', combined_cfg)
        self.assertEqual(combined_cfg.count('module = "applied-field";'), 2)

    def test_append_path_adds_new_hamiltonian(self):
        cfg = make_minimal_cfg()
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            cfg,
            "--config",
            'hamiltonians[] = { module = "applied-field"; field = [0.0, 0.0, 2.0]; };',
            f"--output={self.temp_dir}",
            "--setup-only",
        ]

        self.run_jams_with_args(args)

        combined_cfg = self.load_combined_cfg()
        self.assertIn('field = [ 0.0, 0.0, 1.0 ];', combined_cfg)
        self.assertIn('field = [ 0.0, 0.0, 2.0 ];', combined_cfg)
        self.assertEqual(combined_cfg.count('module = "applied-field";'), 2)

    def test_append_path_adds_new_monitor(self):
        cfg = make_minimal_cfg()
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            cfg,
            "--config",
            'monitors[] = { module = "energy"; output_steps = 10; };',
            f"--output={self.temp_dir}",
            "--setup-only",
        ]

        self.run_jams_with_args(args)

        combined_cfg = self.load_combined_cfg()
        self.assertIn('module = "magnetisation";', combined_cfg)
        self.assertIn('module = "energy";', combined_cfg)
        self.assertIn('output_steps = 10;', combined_cfg)

    def test_indexed_path_requires_existing_element(self):
        cfg = make_minimal_cfg()
        args = [
            self.binary_path,
            "--name=jams",
            "--config",
            cfg,
            "--config",
            'hamiltonians[1].field = [0.0, 0.0, 2.0];',
            f"--output={self.temp_dir}",
            "--setup-only",
        ]

        result = self.run_jams_with_args_allow_failure(args)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid config path assignment", result.stderr + result.stdout)
