#include "gtest/gtest.h"
#include "jams/test/containers/test_vector_set.h"
#include "jams/test/containers/test_unordered_vector_set.h"
//#include "jams/test/containers/test_synced_memory.h"
//#include "jams/test/containers/test_multiarray.h"

#ifdef HAS_CUDA
#include "jams/hamiltonian/test_cuda_dipole_fft.h"
#endif
//#include "jams/test/hamiltonian/test_dipole_cpu_tensor.h"
//#include "jams/test/hamiltonian/test_dipole_cpu_fft.h"
//#include "jams/test/hamiltonian/test_dipole_cpu_bruteforce.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}