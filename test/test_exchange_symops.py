import os
import subprocess
import tempfile
import unittest
import shutil
from test.jams_integration_test import JamsIntegrationtest

class TestExchangeSymops(JamsIntegrationtest):
    binary_path = "cmake-build-debug/bin/jams"
    def test_local_point_group(self):
        # Starting with a hard test for the quite low symmetry system CrPS4. The local point group symmetry operations
        # must be found correctly for the Cr sites because the crystal point group would give non-existent interactions
        # due to the inversion operation.
        result = self.runJams(f"{self.test_dir}/test_exchange_symops.cfg")
        self.checkResultForLine(result, "computed interactions: 1536")


if __name__ == "__main__":
    unittest.main()