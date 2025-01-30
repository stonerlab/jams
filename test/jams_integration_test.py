import os
import subprocess
import tempfile
import unittest
import shutil

class JamsIntegrationtest(unittest.TestCase):
    temp_dir = tempfile.mkdtemp()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    binary_path = "/path/to/your/external/binary"

    def setUp(self):
        """
        Setup before each test:
        - Create a temporary directory for input and output files.
        - Define the path to the external binary.
        """
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"JAMS binary not found: {self.binary_path}")

    def tearDown(self):
        """
        Cleanup after each test:
        - Remove the temporary directory and its contents.
        """
        shutil.rmtree(self.temp_dir)

    def runJams(self, cfg_file, setup_only=False):
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
            self.fail(f"JAMS execution failed: {e.stderr.decode()}")

        return result

    def checkResultForLine(self, result, expected_line):
        output_lines = result.stdout.splitlines()

        # Normalize whitespace and check for the expected line
        found = any(expected_line == line.strip() for line in output_lines)

        # Assert that the expected line is in the output
        self.assertTrue(found, f"Expected line '{expected_line}' not found in output:\n{result.stdout}")




if __name__ == "__main__":
    unittest.main()