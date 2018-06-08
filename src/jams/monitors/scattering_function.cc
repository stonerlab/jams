//
// Created by Joe Barker on 2018/05/31.
//

#include <vector>
#include <complex>

#include "jams/monitors/scattering_function.h"

#include <libconfig.h++>
#include <jams/core/solver.h>
#include <jams/core/globals.h>
#include <jams/helpers/fft.h>
#include <jams/helpers/stats.h>
#include <pcg/pcg_random.hpp>
#include "jams/core/lattice.h"

ScatteringFunctionMonitor::ScatteringFunctionMonitor(const libconfig::Setting &settings) : Monitor(settings) {
  std::string name = seedname + "_fk.tsv";
  outfile.open(name.c_str());

  libconfig::Setting& solver_settings = ::config->lookup("solver");

}

void ScatteringFunctionMonitor::update(Solver *solver) {
  using namespace globals;
  using namespace std;

  std::vector<double> sxt(num_spins);
  std::vector<double> syt(num_spins);
  std::vector<double> szt(num_spins);

  for (auto i = 0; i < num_spins; ++i) {
    sxt[i] = s(i, 0);
    syt[i] = s(i, 1);
    szt[i] = s(i, 2);
  }

  sx_.push_back(sxt);
  sy_.push_back(syt);
  sz_.push_back(szt);
}

bool ScatteringFunctionMonitor::is_converged() {
  return false;
}

ScatteringFunctionMonitor::~ScatteringFunctionMonitor() {
  using namespace std;
  using namespace globals;
  using Complex = std::complex<double>;

  std::ofstream cfile(seedname + "_corr.tsv");

  unsigned num_samples = 5000;
  unsigned num_sub_samples = 200;

  const unsigned num_qvec = 8;
  const Vec3 qmax = {0.0, 0.0, 0.5};
  vector<Vec3> qvecs(num_qvec, {0.0, 0.0, 0.0});

  for (auto i = 0; i < num_qvec; ++i){
    qvecs[i] = qmax * (i / double(num_qvec));
  }

  pcg32 rng;
  vector<unsigned> random_spins(20);
  for (auto ii = 0; ii < random_spins.size(); ++ii) {
    random_spins[ii] = rng(num_spins);
  }

  vector<Vec3> r(num_spins);
  for (auto i = 0; i < num_spins; ++i) {
    r[i] = lattice->atom_position(i);
  }
  for (auto t = 0; t < num_sub_samples; ++t) {
    vector<Complex> Sq(num_qvec, 0.0);
    for (auto t0 = 0; t0 < (num_samples-num_sub_samples); t0+=num_sub_samples/2) {
      for (auto ii = 0; ii < random_spins.size(); ++ii) {
        unsigned i = random_spins[ii];
        for (auto j = 0; j < num_spins; ++j) {
          const auto rij = lattice->displacement(r[i], r[j]);
          for (auto n = 0; n < num_qvec; ++n){
            const auto Q = qvecs[n];
            Complex s_plus_i(sx_[t0+t][i], sy_[t0+t][i]);
            Complex s_minus_j(sx_[t0][j], -sy_[t0][j]);

            Sq[n] += (s_plus_i * s_minus_j) * exp(-kImagTwoPi * dot(Q, rij));
          }
        }
      }
    }
    for (auto n = 0; n < num_qvec; ++n){
      const auto Q = qvecs[n];
      cfile << t << " " << Q[2] << " " << Sq[n].real() << " " << Sq[n].imag() << " " << "\n";
    }
    cfile << "\n" << std::endl;
  }
}
