#include "gtest/gtest.h"
#include "jams/containers/test_synced_memory.h"
#include "jams/containers/test_multiarray.h"


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}