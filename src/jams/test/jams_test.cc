#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "jams/test/containers/test_mat3.h"
#include "jams/test/containers/test_vector_set.h"
#include "jams/test/containers/test_unordered_vector_set.h"
#include "jams/test/containers/test_neartree.h"
#include "jams/test/containers/test_cell.h"
#include <jams/lattice/minimum_image.t.h>
#include <jams/lattice/interaction_neartree.t.h>

#include "jams/test/containers/test_synced_memory.h"
#include "jams/test/containers/test_multiarray.h"
#include "jams/test/hamiltonian/test_dipole.h"

#ifdef HAS_CUDA
#include <jams/cuda/cuda_array_reduction.t.h>
#endif

int main(int argc, char **argv) {
  srand(time(NULL));

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}