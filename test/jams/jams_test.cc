#include "gtest/gtest.h"
#include "jams/hamiltonian/test_dipole_bruteforce.h"
#include "jams/hamiltonian/test_dipole_fft.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}