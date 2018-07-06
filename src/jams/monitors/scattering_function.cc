//
// Created by Joe Barker on 2018/05/31.
//

#include <vector>
#include <complex>
#include <chrono>

#include "jams/monitors/scattering_function.h"

#include <libconfig.h++>
#include <jams/core/solver.h>
#include <jams/core/globals.h>
#include <jams/helpers/fft.h>
#include <jams/helpers/stats.h>
#include "jams/helpers/duration.h"
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

std::vector<double> ScatteringFunctionMonitor::time_correlation(unsigned i, unsigned j, unsigned subsample){
  unsigned n_a = sx_.size();
  unsigned n_b = subsample;

  std::vector<double> out(n_a + n_b - 1, 0.0);

  for (auto m = 0; m < n_a + n_b - 1; ++m) {
    double sum = 0.0;
    for (auto n = 0; n < n_b; ++n) {
      sum += ((n < n_a) && (m + n < n_b)) ? sx_[n][i]*sx_[m + n][j] + sy_[n][i]*sy_[m + n][j] : 0.0;
    }
    out[m] = sum;
  }
  return out;
}


ScatteringFunctionMonitor::~ScatteringFunctionMonitor() {
  using namespace std;
  using namespace std::chrono;
  using namespace globals;
  using Complex = std::complex<double>;

  cout << "calculating correlation function" << std::endl;
  auto start_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "start   " << get_date_string(start_time) << "\n\n";
  cout.flush();


  unsigned num_samples = sx_.size();
  unsigned num_sub_samples = 200;

  const unsigned num_qvec = 5;
  const Vec3 qmax = {0.0, 0.0, 0.5};

  vector<Vec3> qvecs(num_qvec, {0.0, 0.0, 0.0});
  for (auto i = 0; i < num_qvec; ++i){
    qvecs[i] = qmax * (i / double(num_qvec));
  }

  vector<double> wpoints(40, 0.0);
  for (auto i = 0; i < wpoints.size(); ++i) {
    wpoints[i] = i * 1.0;
  }

  pcg32 rng;
  vector<unsigned> random_spins(10);
  for (unsigned int &random_spin : random_spins) {
    random_spin = rng(num_spins);
  }

  vector<Vec3> r(num_spins);
  for (auto i = 0; i < num_spins; ++i) {
    r[i] = lattice->atom_position(i);
  }

  vector<Complex> result(num_sub_samples + num_samples - 1, 0.0);

  vector<vector<Complex>> SQw(qvecs.size());
  for (auto &i : SQw) {
    i = std::vector<Complex>(wpoints.size(), {0.0, 0.0});
  }

  const double delta_t = 1e-15 / 1e-12; // ps
  const double lambda = 0.1;

  for (const auto i : random_spins) {
#pragma omp parallel for default(none) shared(SQw, globals::num_spins, lattice, r)
    for (unsigned j = 0; j < globals::num_spins; ++j) {
//      std::cout << i << " " << j << std::endl;

      const auto R = lattice->displacement(r[i], r[j]);
      const auto correlation = time_correlation(i, j, 300);

      // spatial fourier transform
      unsigned nq = 0;
      for (const auto &Q : qvecs) {
        unsigned nw = 0;
        const auto exp_QR = exp(kImagTwoPi * dot(Q, R));
        for (const auto &w : wpoints) {
          // n = 0
          Complex sum = correlation[0] * exp_QR;

          auto exp_wt = exp(kImagTwoPi * w * delta_t);
          for (auto n = 1; n < correlation.size(); ++n) {
             sum += correlation[n] * exp_QR * exp_wt;
             exp_wt *= exp_wt;
          }
#pragma omp critical
          {
            SQw[nq][nw] += -kImagOne * sum;
          };
          nw++;
        }
        nq++;
      }
    }
  }

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "finish  " << get_date_string(end_time) << "\n\n";
  cout << "runtime " << duration_string(end_time - start_time) << "\n";
  cout.flush();

  std::ofstream cfile(seedname + "_corr.tsv");

  unsigned nq = 0;
  for (const auto Q : qvecs) {
    unsigned nw = 0;
    for (const auto w : wpoints) {
        cfile << Q[2] << " " << w << " " << real(SQw[nq][nw]) << " " << imag(SQw[nq][nw]) << std::endl;
      nw++;
    }
    nq++;
  }


}
