#include "gtest/gtest.h"
#include "jams/interactions.h"
#include "jams/containers/test_cell.h"
#include "jams/hamiltonian/test_dipole_cpu_bruteforce.h"
#include "jams/hamiltonian/test_dipole_fft.h"

#if HAS_CUDA
#include "jams/hamiltonian/test_dipole_gpu_bruteforce.h"
#include "jams/hamiltonian/test_cuda_dipole_fft.h"
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}