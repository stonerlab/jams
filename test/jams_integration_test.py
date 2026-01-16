import os
import subprocess
import tempfile
import unittest
import shutil
import time
import sys
import argparse
from pathlib import Path

def _consume_binary_path_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binary-path")
    args, remaining = parser.parse_known_args(sys.argv[1:])
    if args.binary_path:
        sys.argv[:] = [sys.argv[0]] + remaining
        return args.binary_path
    return None

_BINARY_PATH_FROM_ARGS = _consume_binary_path_arg()
_BINARY_PATH_FROM_ENV = os.getenv("JAMS_BINARY_PATH")

class JamsIntegrationtest(unittest.TestCase):
    temp_dir = tempfile.mkdtemp()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    binary_path = _BINARY_PATH_FROM_ENV or _BINARY_PATH_FROM_ARGS
    keep_artifacts = False
    enable_gpu = os.getenv("JAMS_TEST_ENABLE_GPU", "").lower() in {"1", "true", "yes"}

    def setUp(self):
        """
        Setup before each test:
        - Create a temporary directory for input and output files.
        - Define the path to the external binary.
        """
        super().setUp()

        if not self.binary_path:
            raise FileNotFoundError(
                "JAMS binary path not set (use JAMS_BINARY_PATH=/path/to/jams or --binary-path=/path/to/jams)"
            )
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"JAMS binary not found: {self.binary_path}")

        # Make a unique dir per test
        stamp = time.strftime("%Y%m%d-%H%M%S")
        test_class = self.__class__.__name__
        test_method = self._testMethodName

        self.artifact_dir = (
                Path(self.temp_dir)
                / self.__class__.__name__
                / self._testMethodName
                / stamp
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _test_failed(self) -> bool:
        # Works in modern Python; outcome holds errors/failures
        outcome = getattr(self, "_outcome", None)
        if outcome is None:
            return False
        # outcome.errors is a list of (testcase, exc_info)
        return any(exc for _, exc in getattr(outcome, "errors", []) if exc is not None)


    def tearDown(self):
        """
        Cleanup after each test:
        - Remove the temporary directory and its contents.
        """
        keep = os.getenv("JAMS_TEST_KEEP_ARTIFACTS", "").lower() in {"1", "true", "yes"} or self.keep_artifacts
        if keep or self._test_failed():
            # Keep temp_dir if you want everything; or keep only artifact_dir
            print(f"Keeping artifacts in: {self.artifact_dir}")
        else:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def runJamsCfgFile(self, cfg_file, setup_only=False):
        args = [self.binary_path, cfg_file, f"--output={self.temp_dir}"]
        if setup_only:
            args.append("--setup-only")

        try:
            result = subprocess.run(
                args,
                check=True,  # Raise an exception if the binary fails
                text=True, # Ensures output is in string format (not bytes)
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"JAMS execution failed: {e.stderr}")

        return result

    def runJamsCfg(self, cfg_str, setup_only=False):
        args = [self.binary_path, "--name=jams", "--config", cfg_str, f"--output={self.temp_dir}"]
        if setup_only:
            args.append("--setup-only")

        try:
            result = subprocess.run(
                args,
                check=True,  # Raise an exception if the binary fails
                text=True, # Ensures output is in string format (not bytes)
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"JAMS execution failed: {e.stderr}")

    def checkResultForLine(self, result, expected_line):
        output_lines = result.stdout.splitlines()

        # Normalize whitespace and check for the expected line
        found = any(expected_line == line.strip() for line in output_lines)

        # Assert that the expected line is in the output
        self.assertTrue(found, f"Expected line '{expected_line}' not found in output:\n{result.stdout}")




if __name__ == "__main__":
    unittest.main()
