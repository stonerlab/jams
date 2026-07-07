#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "jams/test/containers/test_vector_set.h"
#include "jams/test/containers/test_unordered_vector_set.h"
#include "jams/test/containers/test_interaction_list.h"
#include "jams/test/containers/test_neartree.h"
#include "jams/test/containers/test_cell.h"
#include "jams/test/core/test_interactions.h"
#include "jams/test/core/test_magnetic_moments.h"
#include "jams/test/core/test_thermostat_temperature_profile.h"
#include "jams/test/initializer/test_damping_regions_initializer.h"
#include "jams/test/initializer/test_h5_initializer.h"
#include <jams/lattice/minimum_image.t.h>
#include <jams/lattice/interaction_neartree.t.h>
#include <jams/test/lattice/test_lattice_size.h>
#include <jams/test/monitors/test_magnetisation.h>
#include <jams/test/monitors/test_magnon_spectrum_cuda.h>
#include <jams/test/monitors/test_hdf5.h>
#include <jams/test/monitors/test_magnetisation_layers.h>
#include <jams/test/monitors/test_neutron_scattering.h>

#include "jams/test/hamiltonian/test_crystal_field.h"
#include "jams/test/hamiltonian/test_dipole.h"
#include "jams/test/hamiltonian/test_exchange_stencil.h"
#include "jams/test/solvers/test_cpu_rotations.h"
#include "jams/test/thermostats/test_quantum_spde_noise.h"

#ifdef HAS_CUDA
#include <jams/cuda/cuda_array_reduction.t.h>
#include "jams/test/hamiltonian/test_anisotropy_polynomial.h"
#include "jams/test/monitors/test_cuda_grouped_spin_reduction.h"
#include "jams/test/monitors/test_cuda_thermal_current.h"
#include "jams/test/thermostats/test_cuda_quantum_spde_noise.h"
#endif

int main(int argc, char **argv) {
  srand(time(NULL));

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
