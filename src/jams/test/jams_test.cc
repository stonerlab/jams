#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "jams/test/containers/test_mat3.h"
#include "jams/test/containers/test_vector_set.h"
#include "jams/test/containers/test_unordered_vector_set.h"
#include "jams/test/containers/test_neartree.h"
//#include "jams/test/containers/test_synced_memory.h"
#include "jams/test/containers/test_multiarray.h"
#include "jams/test/hamiltonian/test_dipole_cpu.h"

#ifdef HAS_CUDA
#include "jams/hamiltonian/test_cuda_dipole_fft.h"
#endif

int main(int argc, char **argv) {
  srand(time(NULL));

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}