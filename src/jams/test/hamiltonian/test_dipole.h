#include <gtest/gtest.h>

#include <ctime>

#include <libconfig.h++>
#include <memory>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/utils.h"
#include "jams/hamiltonian/dipole_bruteforce.h"
#include "jams/hamiltonian/dipole_fft.h"
#include "jams/hamiltonian/dipole_neartree.h"
#include "jams/hamiltonian/dipole_neighbour_list.h"
// WARNING: dipole tensor tests are disabled because they are very slow for
// automated testing on github
//#include "jams/hamiltonian/dipole_tensor.h"
#include "jams/helpers/output.h"
#include "jams/helpers/random.h"
#include "jams/test/hamiltonian/test_dipole_input.h"
#include "jams/test/output.h"


#if HAS_CUDA
#include "jams/hamiltonian/cuda_dipole_fft.h"
#include "jams/hamiltonian/cuda_dipole_bruteforce.h"
#endif

// Testing to validate the dipole Hamiltonians give correct results. We compare
// to analytic results from 2016-Johnston-PhysRevB.93.014421 as well as using
// random spin configurations to check consistency between classes. We also check
// the results are correct for more than 1 sublattice to check for bugs in FFT
// implementations.

//---------------------------------------------------------------------
// NOTE: The liberal use of #pragma nounroll_and_jam is to avoid a bug
//       in the Intel 2016.2 compiler which mangles these loops when
//       unrolling
//---------------------------------------------------------------------

// Dipole Hamiltonian classes to test. They should be #included above.



typedef testing::Types<
    DipoleNearTreeHamiltonian,
    DipoleNeighbourListHamiltonian,
    DipoleBruteforceHamiltonian,
    DipoleFFTHamiltonian
//    DipoleTensorHamiltonian
> DipoleHamiltonianCPUTypes;

TYPED_TEST_SUITE(DipoleHamiltonianCPUTests, DipoleHamiltonianCPUTypes);

#ifdef HAS_CUDA
typedef testing::Types<
    CudaDipoleFFTHamiltonian,
    CudaDipoleBruteforceHamiltonian
> DipoleHamiltonianGPUTypes;

TYPED_TEST_SUITE(DipoleHamiltonianGPUTests, DipoleHamiltonianGPUTypes);
#endif

template<typename T>
class DipoleHamiltonianTests : public ::testing::Test {
public:
    DipoleHamiltonianTests() {
    }

    ~DipoleHamiltonianTests() override = default;

    virtual void SetUp(const std::string &config_string) {

      jams::testing::toggle_cout();
      // create global objects
      ::globals::lattice = new Lattice();
      ::globals::config = std::make_unique<libconfig::Config>();
      // configure global objects and create lattice and solver
      ::globals::config->readString(config_string);
      ::globals::lattice->init_from_config(*::globals::config);
      ::globals::solver = Solver::create(globals::config->lookup("solver"));
      ::globals::solver->initialize(globals::config->lookup("solver"));
      ::globals::solver->register_physics_module(Physics::create(
          globals::config->lookup("physics")));

      // configure the current Hamiltonian for testing
      hamiltonian = std::make_unique<T>(::globals::config->lookup("hamiltonians.[0]"), globals::num_spins);

      jams::testing::toggle_cout();
    }

    virtual void TearDown() override {
      // reset global objects
      globals::num_spins = 0;
      globals::num_spins3 = 0;

      jams::util::force_deallocation(globals::s);
      jams::util::force_deallocation(globals::h);
      jams::util::force_deallocation(globals::ds_dt);
      jams::util::force_deallocation(globals::positions);
      jams::util::force_deallocation(globals::alpha);
      jams::util::force_deallocation(globals::mus);
      jams::util::force_deallocation(globals::gyro);

      if (::globals::solver) {
        delete ::globals::solver;
        ::globals::solver = nullptr;
      }

      ::globals::config = nullptr;

      if (globals::lattice) {
        delete ::globals::lattice;
        ::globals::lattice = nullptr;
      }
    }

    // test the total dipole energy for an ordered spin configuration
    // compared to an analytic eigen value
    void eigenvalue_test(const std::string &spin_config_name, const double &expected_eigenvalue) {
      jams::testing::toggle_cout();
      std::unique_ptr<DipoleNearTreeHamiltonian> reference_hamiltonian(
          new DipoleNearTreeHamiltonian(::globals::config->lookup("hamiltonians.[0]"), globals::num_spins));
      jams::testing::toggle_cout();

      double analytic = analytic_prefactor * expected_eigenvalue;
      double numeric = hamiltonian->calculate_total_energy(0) / double(globals::num_spins);
      double reference =
          reference_hamiltonian->calculate_total_energy(0) / double(globals::num_spins);

      std::cout << "spins:      " << spin_config_name << "\n";
      std::cout << "expected:   " << jams::fmt::sci << analytic << " meV/spin\n";
      std::cout << "actual:     " << jams::fmt::sci << numeric << " meV/spin\n";
      std::cout << "difference: " << jams::fmt::sci << std::abs(analytic - numeric) << " meV/spin\n";
      std::cout << "tolerance:  " << jams::fmt::sci << target_accuracy << " meV/spin\n" << std::endl;

      ASSERT_NEAR(numeric, analytic, target_accuracy);
      ASSERT_NEAR(numeric, reference, target_accuracy);

      hamiltonian->calculate_fields(0);
      reference_hamiltonian->calculate_fields(0);

      for (auto i = 0; i < globals::num_spins; ++i) {
        for (auto j = 0; j < 3; ++j) {
          ASSERT_NEAR(hamiltonian->field(i, j), reference_hamiltonian->field(i,j), target_accuracy);
        }
      }
    }

    // test the total dipole energy for a random spin configuration
    // compared to a reference Hamiltonian
    void random_spin_test() {
      jams::testing::toggle_cout();
      std::unique_ptr<DipoleNearTreeHamiltonian> reference_hamiltonian(
          new DipoleNearTreeHamiltonian(::globals::config->lookup("hamiltonians.[0]"), globals::num_spins));
      jams::testing::toggle_cout();

      pcg32 rng = pcg_extras::seed_seq_from<std::random_device>();
      for (unsigned int i = 0; i < globals::num_spins; ++i) {
        Vec3 spin = jams::uniform_random_sphere(rng);
        globals::s(i, 0) = spin[0];
        globals::s(i, 1) = spin[1];
        globals::s(i, 2) = spin[2];
      }

      double numeric = hamiltonian->calculate_total_energy(0) / double(globals::num_spins);
      double reference =
          reference_hamiltonian->calculate_total_energy(0) / double(globals::num_spins);

      std::cout << "spin:       random" << "\n";
      std::cout << "expected:   " << jams::fmt::sci << reference << " meV/spin\n";
      std::cout << "actual:     " << jams::fmt::sci << numeric << " meV/spin\n";
      std::cout << "difference: " << jams::fmt::sci << std::abs(reference - numeric) << " meV/spin\n";
      std::cout << "tolerance:  " << jams::fmt::sci << target_accuracy << " meV/spin\n" << std::endl;

      ASSERT_NEAR(numeric, reference, target_accuracy);

      hamiltonian->calculate_fields(0);
      reference_hamiltonian->calculate_fields(0);

      for (auto i = 0; i < globals::num_spins; ++i) {
        for (auto j = 0; j < 3; ++j) {
          ASSERT_NEAR(hamiltonian->field(i, j), reference_hamiltonian->field(i,j), target_accuracy);
        }
      }
    }

    // test the total dipole energy for an ordered spin configuration
    // compared to a reference Hamiltonian
    void ordered_spin_test(const Vec3& spin_direction) {
        jams::testing::toggle_cout();
        std::unique_ptr<DipoleNearTreeHamiltonian> reference_hamiltonian(
                new DipoleNearTreeHamiltonian(::globals::config->lookup("hamiltonians.[0]"), globals::num_spins));
        jams::testing::toggle_cout();

        for (unsigned int i = 0; i < globals::num_spins; ++i) {
            globals::s(i, 0) = spin_direction[0];
            globals::s(i, 1) = spin_direction[1];
            globals::s(i, 2) = spin_direction[2];
        }

        double numeric = hamiltonian->calculate_total_energy(0) / double(globals::num_spins);
        double reference =
                reference_hamiltonian->calculate_total_energy(0) / double(globals::num_spins);

        std::cout << "spin:       " << spin_direction << "\n";
        std::cout << "expected:   " << jams::fmt::sci << reference << " meV/spin\n";
        std::cout << "actual:     " << jams::fmt::sci << numeric << " meV/spin\n";
        std::cout << "difference: " << jams::fmt::sci << std::abs(reference - numeric) << " meV/spin\n";
        std::cout << "tolerance:  " << jams::fmt::sci << target_accuracy << " meV/spin\n" << std::endl;

        hamiltonian->calculate_fields(0);
        reference_hamiltonian->calculate_fields(0);

        ASSERT_NEAR(numeric, reference, target_accuracy);
        for (auto i = 0; i < globals::num_spins; ++i) {
          for (auto j = 0; j < 3; ++j) {
            ASSERT_NEAR(hamiltonian->field(i, j), reference_hamiltonian->field(i,j), target_accuracy);
          }
        }
    }

    std::unique_ptr<T> hamiltonian;

    // mus = 2.0 muB; a = 0.3 nm
    // -(0.5 * mu0 / (4 pi)) * (mus / a^3)^2 = -23595.95647978379 J/m^3
    // (-23595.95647978379 J/m^3) * (0.3e-9 m)^3 = -6.37090824954162e-25 J
    //                                           = -0.0039764081652071 meV
    const double analytic_prefactor = -((0.5 * kVacuumPermeabilityIU) / (4*kPi)) *  pow2(2 * kBohrMagnetonIU) / pow3(0.3e-9); // meV

    // target accuracy for total energy per spin in meV
    const double target_accuracy = 1e-5; // 10 neV accuracy
};

template<typename T>
class DipoleHamiltonianCPUTests : public DipoleHamiltonianTests<T> {};

#ifdef HAS_CUDA
template<typename T>
class DipoleHamiltonianGPUTests : public DipoleHamiltonianTests<T> {
public:
    void SetUp(const std::string &config_string) override {
      cudaDeviceReset();
      DipoleHamiltonianTests<T>::SetUp(config_string);
    }

    void TearDown() override {
      DipoleHamiltonianTests<T>::TearDown();
    }
};
#endif

// 1D ferromagnetic spin chain
TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_1D_FM_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  // S = (1, 0, 0) FM
  TestFixture::eigenvalue_test("FM x-aligned", 4.808228);

  // S = (0, 1, 0) FM
  #pragma nounroll_and_jam
  for (auto i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 1.0;
    globals::s(i, 2) = 0.0;
  }
  TestFixture::eigenvalue_test("FM y-aligned", -2.404114);

  // S = (0, 0, 1) FM
  #pragma nounroll_and_jam
  for (auto i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }
  TestFixture::eigenvalue_test("FM z-aligned", -2.404114);
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_1D_FM_GPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  // S = (1, 0, 0) FM
  TestFixture::eigenvalue_test("FM x-aligned", 4.808228);

  // S = (0, 1, 0) FM
  #pragma nounroll_and_jam
  for (auto i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 1.0;
    globals::s(i, 2) = 0.0;
  }
  TestFixture::eigenvalue_test("FM y-aligned", -2.404114);

  // S = (0, 0, 1) FM
  #pragma nounroll_and_jam
  for (auto i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }
  TestFixture::eigenvalue_test("FM z-aligned", -2.404114);
}
#endif

// 1D ferromagnetic spin chain with two atoms defined in unit cell
TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_two_atom_1D_FM_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc_2_atom
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  TestFixture::eigenvalue_test("FM z-aligned", 4.808228);
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_two_atom_1D_FM_GPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_sc_2_atom
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  TestFixture::eigenvalue_test("FM z-aligned", 4.808228);
}
#endif


// 1D ferromagnetic spin chain with random spin orientations
TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_1D_FM_random_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  TestFixture::random_spin_test();
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_1D_FM_random_GPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  TestFixture::random_spin_test();
}
#endif


// 1D antiferromagnetic spin chain
TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_1D_AFM_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  // S = (1, 0, 0) AFM
  #pragma nounroll_and_jam
  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    if (i % 2 == 0) {
      globals::s(i, 0) = -1.0;
    } else {
      globals::s(i, 0) = 1.0;
    }
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 0.0;
  }
  TestFixture::eigenvalue_test("AFM x-aligned", -3.60617);

  // S = (0, 1, 0) AFM
  #pragma nounroll_and_jam
  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    if (i % 2 == 0) {
      globals::s(i, 1) = -1.0;
    } else {
      globals::s(i, 1) = 1.0;
    }
    globals::s(i, 2) = 0.0;
  }
  TestFixture::eigenvalue_test("AFM y-aligned", 1.803085);
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_1D_AFM_GPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  // S = (1, 0, 0) AFM
  #pragma nounroll_and_jam
  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    if (i % 2 == 0) {
      globals::s(i, 0) = -1.0;
    } else {
      globals::s(i, 0) = 1.0;
    }
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 0.0;
  }
  TestFixture::eigenvalue_test("AFM x-aligned", -3.60617);

  // S = (0, 1, 0) AFM
  #pragma nounroll_and_jam
  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    if (i % 2 == 0) {
      globals::s(i, 1) = -1.0;
    } else {
      globals::s(i, 1) = 1.0;
    }
    globals::s(i, 2) = 0.0;
  }
  TestFixture::eigenvalue_test("AFM y-aligned", 1.803085);
}
#endif


TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_cubic_unitcell_FM_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_cubic_unitcell
      + config_lattice({5, 5, 5}, {true, true, true})
      + config_dipole("dipole", 2.0));

  TestFixture::ordered_spin_test({0.0, 0.0, 1.0});
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_cubic_unitcell_FM_GPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_cubic_unitcell
      + config_lattice({5, 5, 5}, {true, true, true})
      + config_dipole("dipole", 2.0));

  TestFixture::ordered_spin_test({0.0, 0.0, 1.0});
}
#endif


TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_non_cubic_unitcell_FM_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_non_cubic_unitcell
      + config_lattice({6, 6, 6}, {true, true, true})
      + config_dipole("dipole", 1.0));

  TestFixture::ordered_spin_test({0.0, 0.0, 1.0});
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_non_cubic_unitcell_FM_GPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_non_cubic_unitcell
      + config_lattice({6, 6, 6}, {true, true, true})
      + config_dipole("dipole", 1.0));

  TestFixture::ordered_spin_test({0.0, 0.0, 1.0});
}
#endif


TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_2D_FM_CPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc
      + config_lattice({256, 256, 1}, {true, true, false})
      + config_dipole("dipole", 128.0));

  // S = (0, 0, 1) FM
  #pragma nounroll_and_jam
  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }

  // Eigen value from fit function from 2016-Johnston-PhysRevB.93.014421 Fig. 17(a)
  TestFixture::eigenvalue_test("FM z-aligned", -9.033622 + 6.28356 * (1.0 / 128.0));
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_2D_FM_GPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_sc
      + config_lattice({256, 256, 1}, {true, true, false})
      + config_dipole("dipole", 128.0));

  // S = (0, 0, 1) FM
  #pragma nounroll_and_jam
  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }

  // Eigen value from fit function from 2016-Johnston-PhysRevB.93.014421 Fig. 17(a)
  TestFixture::eigenvalue_test("FM z-aligned", -9.033622 + 6.28356 * (1.0 / 128.0));
}
#endif


TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_2D_AFM_CPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc_AFM
      + config_lattice({256, 256, 1}, {true, true, false})
      + config_dipole("dipole", 128.0));

  // 2016-Johnston-PhysRevB.93.014421 Fig. 18(a)
  TestFixture::eigenvalue_test("FM z-aligned", 2.6458865);
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_2D_AFM_GPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_sc_AFM
      + config_lattice({256, 256, 1}, {true, true, false})
      + config_dipole("dipole", 128.0));

  // 2016-Johnston-PhysRevB.93.014421 Fig. 18(a)
  TestFixture::eigenvalue_test("FM z-aligned", 2.6458865);
}
#endif


TYPED_TEST(DipoleHamiltonianCPUTests, total_energy_two_atom_2D_FM_CPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_bcc_2_atom
      + config_lattice({32, 32, 1}, {true, true, false})
      + config_dipole("dipole", 16.0));

  TestFixture::random_spin_test();
}

#ifdef HAS_CUDA
TYPED_TEST(DipoleHamiltonianGPUTests, total_energy_two_atom_2D_FM_GPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_gpu
      + config_unitcell_bcc_2_atom
      + config_lattice({32, 32, 1}, {true, true, false})
      + config_dipole("dipole", 16.0));

  TestFixture::random_spin_test();
}
#endif


