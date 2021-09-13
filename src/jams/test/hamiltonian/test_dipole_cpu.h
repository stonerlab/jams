#include <gtest/gtest.h>

#include <ctime>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
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
> DipoleHamiltonianTypes;

TYPED_TEST_SUITE(DipoleHamiltonianTests, DipoleHamiltonianTypes);

template<typename T>
class DipoleHamiltonianTests : public ::testing::Test {
protected:
    DipoleHamiltonianTests() {
      // create global objects
      ::lattice = new Lattice();
      ::config.reset(new libconfig::Config());
    }

    ~DipoleHamiltonianTests() = default;

    void SetUp(const std::string &config_string) {
      jams::testing::toggle_cout();
      // configure global objects and create lattice and solver
      ::config->readString(config_string);
      ::lattice->init_from_config(*::config);
      ::solver = Solver::create(config->lookup("solver"));
      ::solver->initialize(config->lookup("solver"));
      ::solver->register_physics_module(Physics::create(config->lookup("physics")));

      // configure the current Hamiltonian for testing
      hamiltonian.reset(new T(::config->lookup("hamiltonians.[0]"), globals::num_spins));

      jams::testing::toggle_cout();
    }

    virtual void TearDown() {
      // destroy global objects
      delete ::solver;
      delete ::lattice;
    }

    // test the total dipole energy for an ordered spin configuration
    // compared to an analytic eigen value
    void eigenvalue_test(const std::string &spin_config_name, const double &expected_eigenvalue) {
      double analytic = analytic_prefactor * expected_eigenvalue;
      double numeric = hamiltonian->calculate_total_energy() / double(globals::num_spins);

      std::cout << "spins:      " << spin_config_name << "\n";
      std::cout << "expected:   " << jams::fmt::sci << analytic << " meV/spin\n";
      std::cout << "actual:     " << jams::fmt::sci << numeric << " meV/spin\n";
      std::cout << "difference: " << jams::fmt::sci << std::abs(analytic - numeric) << " meV/spin\n";
      std::cout << "tolerance:  " << jams::fmt::sci << target_accuracy << " meV/spin\n" << std::endl;

      ASSERT_NEAR(numeric, analytic, target_accuracy);
    }

    // test the total dipole energy for a random spin configuration
    // compared to a reference Hamiltonian
    void random_spin_test() {
      jams::testing::toggle_cout();
      std::unique_ptr<DipoleNearTreeHamiltonian> reference_hamiltonian(
          new DipoleNearTreeHamiltonian(::config->lookup("hamiltonians.[0]"), globals::num_spins));
      jams::testing::toggle_cout();

      pcg32 rng = pcg_extras::seed_seq_from<std::random_device>();
      for (unsigned int i = 0; i < globals::num_spins; ++i) {
        Vec3 spin = uniform_random_sphere(rng);
        globals::s(i, 0) = spin[0];
        globals::s(i, 1) = spin[1];
        globals::s(i, 2) = spin[2];
      }

      double numeric = hamiltonian->calculate_total_energy() / double(globals::num_spins);
      double reference =
          reference_hamiltonian->calculate_total_energy() / double(globals::num_spins);

      std::cout << "spin:       random" << "\n";
      std::cout << "expected:   " << jams::fmt::sci << reference << " meV/spin\n";
      std::cout << "actual:     " << jams::fmt::sci << numeric << " meV/spin\n";
      std::cout << "difference: " << jams::fmt::sci << std::abs(reference - numeric) << " meV/spin\n";
      std::cout << "tolerance:  " << jams::fmt::sci << target_accuracy << " meV/spin\n" << std::endl;

      ASSERT_NEAR(numeric, reference, target_accuracy);

      hamiltonian->calculate_fields();
      reference_hamiltonian->calculate_fields();

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
                new DipoleNearTreeHamiltonian(::config->lookup("hamiltonians.[0]"), globals::num_spins));
        jams::testing::toggle_cout();

        for (unsigned int i = 0; i < globals::num_spins; ++i) {
            globals::s(i, 0) = spin_direction[0];
            globals::s(i, 1) = spin_direction[1];
            globals::s(i, 2) = spin_direction[2];
        }

        double numeric = hamiltonian->calculate_total_energy() / double(globals::num_spins);
        double reference =
                reference_hamiltonian->calculate_total_energy() / double(globals::num_spins);

        std::cout << "spin:       " << spin_direction << "\n";
        std::cout << "expected:   " << jams::fmt::sci << reference << " meV/spin\n";
        std::cout << "actual:     " << jams::fmt::sci << numeric << " meV/spin\n";
        std::cout << "difference: " << jams::fmt::sci << std::abs(reference - numeric) << " meV/spin\n";
        std::cout << "tolerance:  " << jams::fmt::sci << target_accuracy << " meV/spin\n" << std::endl;

        ASSERT_NEAR(numeric, reference, target_accuracy);
    }

    std::unique_ptr<T> hamiltonian;

    // mus = 2.0 muB; a = 0.3 nm
    // -(0.5 * mu0 / (4 pi)) * (mus / a^3)^2 = -23595.95647978379 J/m^3
    // (-23595.95647978379 J/m^3) * (0.3e-9 m)^3 = -6.37090824954162e-25 J
    //                                           = -0.0039764081652071 meV
    const double analytic_prefactor = -((0.5 * kVacuumPermeabilityIU) / (4*kPi)) *  pow2(2 * kBohrMagnetonIU) / pow3(0.3e-9); // meV

    // target accuracy for total energy per spin in meV
    const double target_accuracy = 1e-6; // nano eV accuracy
};

// 1D ferromagnetic spin chain
TYPED_TEST(DipoleHamiltonianTests, total_energy_1D_FM_CPU) {
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

// 1D ferromagnetic spin chain with two atoms defined in unit cell
TYPED_TEST(DipoleHamiltonianTests, total_energy_two_atom_1D_FM_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc_2_atom
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  TestFixture::eigenvalue_test("FM z-aligned", 4.808228);
}

// 1D ferromagnetic spin chain with random spin orientations
TYPED_TEST(DipoleHamiltonianTests, total_energy_1D_FM_random_CPU) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc
      + config_lattice({2000, 1, 1}, {true, false, false})
      + config_dipole("dipole", 1000.0));

  TestFixture::random_spin_test();
}

// 1D antiferromagnetic spin chain
TYPED_TEST(DipoleHamiltonianTests, total_energy_1D_AFM_CPU) {
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

TYPED_TEST(DipoleHamiltonianTests, total_energy_CPU_2D_FM_SLOW) {
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

TYPED_TEST(DipoleHamiltonianTests, total_energy_2D_AFM_CPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_sc_AFM
      + config_lattice({256, 256, 1}, {true, true, false})
      + config_dipole("dipole", 128.0));

  // 2016-Johnston-PhysRevB.93.014421 Fig. 18(a)
  TestFixture::eigenvalue_test("FM z-aligned", 2.6458865);
}

TYPED_TEST(DipoleHamiltonianTests, total_energy_two_atom_2D_FM_CPU_SLOW) {
  using namespace jams::testing::dipole;
  TestFixture::SetUp(
      config_basic_cpu
      + config_unitcell_bcc_2_atom
      + config_lattice({128, 128, 1}, {true, true, false})
      + config_dipole("dipole", 64.0));

  TestFixture::random_spin_test();
}

TYPED_TEST(DipoleHamiltonianTests, total_energy_cubic_unitcell_FM_CPU) {
    using namespace jams::testing::dipole;
    TestFixture::SetUp(
            config_basic_cpu
            + config_cubic_unitcell
            + config_lattice({5, 5, 5}, {true, true, true})
            + config_dipole("dipole", 2.0));

    TestFixture::ordered_spin_test({0.0, 0.0, 1.0});
}

TYPED_TEST(DipoleHamiltonianTests, total_energy_non_cubic_unitcell_FM_CPU) {
    using namespace jams::testing::dipole;
    TestFixture::SetUp(
            config_basic_cpu
            + config_non_cubic_unitcell
            + config_lattice({6, 6, 6}, {true, true, true})
            + config_dipole("dipole", 1.0));

    TestFixture::ordered_spin_test({0.0, 0.0, 1.0});
}
