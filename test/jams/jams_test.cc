#include "gtest/gtest.h"
#include "jams/containers/test_synced_memory.h"
#include "jams/containers/test_multiarray.h"

#ifdef HAS_CUDA
#include "jams/hamiltonian/test_cuda_dipole_fft.h"
#endif
#include "jams/hamiltonian/test_dipole_cpu_tensor.h"
#include "jams/hamiltonian/test_dipole_cpu_fft.h"




int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}