#include <ctime>

#include <libconfig.h++>

#include "jams/core/lattice.h"
#include "jams/core/output.h"
#include "jams/core/solver.h"
#include "jams/core/physics.h"
#include "jams/core/globals.h"
#include "jams/core/rand.h"

#include "jams/hamiltonian/test_dipole_input.h"
#include "jams/hamiltonian/dipole_fft.h"

namespace {
// The fixture for testing class Foo.
class DipoleHamiltonianFFTTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  DipoleHamiltonianFFTTest() {
    // You can do set-up work for each test here.
    cudaDeviceReset();
    ::lattice = new Lattice();
    ::output = new Output();
    ::config = new libconfig::Config();
    ::rng = new Random();

    ::output->disableConsole();
  }

  virtual ~DipoleHamiltonianFFTTest() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp(const std::string &config_string) {
    // Code here will be called immediately after the constructor (right
    // before each test).
    ::rng->seed(time(NULL));
    ::config->readString(config_string);
    ::lattice->init_from_config(*::config);
    ::physics_module = Physics::create(config->lookup("physics"));
    ::solver = Solver::create(config->lookup("sim.solver"));
    int argc = 0; char **argv; double dt = 0.1;
    ::solver->initialize(argc, argv, dt);
    ::solver->register_physics_module(physics_module);
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
    if (::physics_module != nullptr) {
      delete ::physics_module;
      ::physics_module = nullptr;
    }
    if (::solver != nullptr) {
      delete ::solver;
      ::solver = nullptr;
    }
    if (::lattice != nullptr) {
      delete ::lattice;
      ::lattice = nullptr;
    }
    if (::output != nullptr) {
      delete ::output;
      ::output = nullptr;
    }
    if (::config != nullptr) {
      delete ::config;
      ::config = nullptr;
    }
    if (::rng != nullptr) {
      delete ::rng;
      ::rng = nullptr;
    }
  }

  // Objects declared here can be used by all tests in the test case for Foo.
};

//---------------------------------------------------------------------
// CPU
//---------------------------------------------------------------------

TEST_F(DipoleHamiltonianFFTTest, total_energy_CPU_1D_FM) {
  SetUp(  config_basic_cpu + config_unitcell_sc + config_lattice_1D + config_dipole_fft_1000);

  auto h = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);
 
  // S = (1, 0, 0) FM

  // 1D FM spin chain with spins aligned along chain axis
  double eigenvalue = 4.808228;

  double analytic = analytic_prefactor * eigenvalue;
  double numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

  ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
  ASSERT_NEAR(numeric/analytic, 1.0, 1e-6);

  // S = (0, 1, 0) FM

  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 1.0;
    globals::s(i, 2) = 0.0;
  }

  eigenvalue = -2.404114;
  analytic = analytic_prefactor * eigenvalue;
  numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

  ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
  ASSERT_NEAR(numeric/analytic, 1.0, 1e-6);

  // S = (0, 0, 1) FM

  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }

  eigenvalue = -2.404114;
  analytic = analytic_prefactor * eigenvalue;
  numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

  ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
  ASSERT_NEAR(numeric/analytic, 1.0, 1e-6);
}

//---------------------------------------------------------------------

  TEST_F(DipoleHamiltonianFFTTest, total_energy_CPU_2D_FM_SLOW) {
    SetUp(  config_basic_cpu + config_unitcell_sc + config_lattice_2D + config_dipole_fft_128);

    auto h = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

    // S = (0, 0, 1) FM

    for (unsigned int i = 0; i < globals::num_spins; ++i) {
      globals::s(i, 0) = 0.0;
      globals::s(i, 1) = 0.0;
      globals::s(i, 2) = 1.0;
    }

    // Fit function from 2016-Johnston-PhysRevB.93.014421 Fig. 17(a)
    double eigenvalue = -9.033622 + 6.28356 * (1.0 / 128.0);
    double analytic = analytic_prefactor * eigenvalue;
    double numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

    ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
    ASSERT_NEAR(numeric/analytic, 1.0, 1e-5);
  }

  //---------------------------------------------------------------------

  TEST_F(DipoleHamiltonianFFTTest, total_energy_CPU_1D_AFM) {
    SetUp(  config_basic_cpu + config_unitcell_sc + config_lattice_1D + config_dipole_fft_1000);

    auto h = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

    // S = (1, 0, 0) AFM

    for (unsigned int i = 0; i < globals::num_spins; ++i) {
      if (i % 2 == 0) {
        globals::s(i, 0) = -1.0;
      } else {
        globals::s(i, 0) = 1.0;
      }
      globals::s(i, 1) = 0.0;
      globals::s(i, 2) = 0.0;
    }

    double eigenvalue = -3.60617;
    double analytic = analytic_prefactor * eigenvalue;
    double numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

    ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
    ASSERT_NEAR(numeric/analytic, 1.0, 1e-6);

    // S = (0, 1, 0) AFM

    for (unsigned int i = 0; i < globals::num_spins; ++i) {
      globals::s(i, 0) = 0.0;
      if (i % 2 == 0) {
        globals::s(i, 1) = -1.0;
      } else {
        globals::s(i, 1) = 1.0;
      }
      globals::s(i, 2) = 0.0;
    }

    eigenvalue = 1.803085;
    analytic = analytic_prefactor * eigenvalue;
    numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

    ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
    ASSERT_NEAR(numeric/analytic, 1.0, 1e-6);
  }

//---------------------------------------------------------------------

  TEST_F(DipoleHamiltonianFFTTest, total_energy_CPU_2D_AFM_SLOW) {
    SetUp(  config_basic_cpu + config_unitcell_sc_AFM + config_lattice_2D + config_dipole_fft_128);

    auto h = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

    // 2016-Johnston-PhysRevB.93.014421 Fig. 18(a)
    double eigenvalue = 2.6458865;
    double analytic = analytic_prefactor * eigenvalue;
    double numeric =  numeric_prefactor * h->calculate_total_energy() / double(globals::num_spins) ;

    ASSERT_EQ(std::signbit(numeric), std::signbit(analytic));
    ASSERT_NEAR(numeric/analytic, 1.0, 1e-5);
  }

}

//---------------------------------------------------------------------

TEST_F(DipoleHamiltonianFFTTest, total_energy_CPU_1D_FM_RAND) {
  SetUp(  config_basic_cpu + config_unitcell_sc + config_lattice_1D + config_dipole_bruteforce_1000);

  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    rng->sphere(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
  }

  auto h_bruteforce = new DipoleHamiltonianBruteforce(::config->lookup("hamiltonians.[0]"), globals::num_spins);
  auto h_fft = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

  double result_bruteforce =  numeric_prefactor * h_bruteforce->calculate_total_energy() / double(globals::num_spins) ;
  double result_fft =  numeric_prefactor * h_fft->calculate_total_energy() / double(globals::num_spins);

  ASSERT_EQ(std::signbit(result_bruteforce), std::signbit(result_fft));
  ASSERT_NEAR(result_bruteforce/result_fft, 1.0, 1e-6);
}

//---------------------------------------------------------------------

TEST_F(DipoleHamiltonianFFTTest, total_energy_two_atom_CPU_1D_FM) {
  SetUp(  config_basic_cpu + config_unitcell_sc_2_atom + config_lattice_1D + config_dipole_bruteforce_1000);

  auto h_bruteforce = new DipoleHamiltonianBruteforce(::config->lookup("hamiltonians.[0]"), globals::num_spins);
  auto h_fft = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

  double result_bruteforce =  numeric_prefactor * h_bruteforce->calculate_total_energy() / double(globals::num_spins) ;
  double result_fft =  numeric_prefactor * h_fft->calculate_total_energy() / double(globals::num_spins);

  ASSERT_EQ(std::signbit(result_bruteforce), std::signbit(result_fft));
  ASSERT_NEAR(result_bruteforce/result_fft, 1.0, 1e-6);
}

//---------------------------------------------------------------------

TEST_F(DipoleHamiltonianFFTTest, total_energy_two_atom_CPU_2D_FM_SLOW) {
  SetUp(  config_basic_cpu + config_unitcell_bcc_2_atom + config_lattice_2D_128 + config_dipole_bruteforce_64);

  auto h_bruteforce = new DipoleHamiltonianBruteforce(::config->lookup("hamiltonians.[0]"), globals::num_spins);
  auto h_fft = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

  double result_bruteforce =  numeric_prefactor * h_bruteforce->calculate_total_energy() / double(globals::num_spins) ;
  double result_fft =  numeric_prefactor * h_fft->calculate_total_energy() / double(globals::num_spins);

  ASSERT_EQ(std::signbit(result_bruteforce), std::signbit(result_fft));
  ASSERT_NEAR(result_bruteforce/result_fft, 1.0, 1e-6);
}

//---------------------------------------------------------------------

TEST_F(DipoleHamiltonianFFTTest, total_energy_two_atom_CPU_1D_FM_RAND) {
  SetUp(  config_basic_cpu + config_unitcell_sc_2_atom + config_lattice_1D + config_dipole_bruteforce_1000);

  for (unsigned int i = 0; i < globals::num_spins; ++i) {
    rng->sphere(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
  }

  auto h_bruteforce = new DipoleHamiltonianBruteforce(::config->lookup("hamiltonians.[0]"), globals::num_spins);
  auto h_fft = new DipoleHamiltonianFFT(::config->lookup("hamiltonians.[0]"), globals::num_spins);

  double result_bruteforce =  numeric_prefactor * h_bruteforce->calculate_total_energy() / double(globals::num_spins) ;
  double result_fft =  numeric_prefactor * h_fft->calculate_total_energy() / double(globals::num_spins);

  ASSERT_EQ(std::signbit(result_bruteforce), std::signbit(result_fft));
  ASSERT_NEAR(result_bruteforce/result_fft, 1.0, 1e-6);
}