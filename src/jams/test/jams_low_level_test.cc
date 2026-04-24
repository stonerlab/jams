#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <cstdlib>
#include <ctime>

#include "jams/test/containers/test_mat3.h"
#include "jams/test/containers/test_multiarray.h"
#include "jams/test/containers/test_synced_memory.h"
#include "jams/test/containers/test_vec3.h"

int main(int argc, char **argv) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
