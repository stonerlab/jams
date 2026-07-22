#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <cstdlib>
#include <ctime>

#include "jams/test/containers/test_mat3.h"
#include "jams/test/containers/test_multiarray.h"
#include "jams/test/containers/test_name_id_map.h"
#include "jams/test/containers/test_synced_memory.h"
#include "jams/test/containers/test_vec3.h"
#include "jams/test/containers/test_block_sparse_interaction_matrix.h"
#include "jams/test/helpers/test_utils.h"
#include "jams/test/maths/test_angles.h"
#include "jams/test/maths/test_neutrons.h"
#include "jams/test/maths/test_tesseral_harmonics.h"

int main(int argc, char **argv) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
