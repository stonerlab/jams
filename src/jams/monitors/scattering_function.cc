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
#include "jams/helpers/random.h"
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
  const unsigned n_a = sx_.size();
  const unsigned n_b = subsample;

  std::vector<double> out(n_a + n_b - 1, 0.0);

  for (auto m = 0; m < n_a + n_b - 1; ++m) {
    for (auto n = 0; n < n_b; ++n) {
      if ((n > n_a) || (m + n > n_b)) {
        continue;
      }
      out[m] += sx_[n][i] * sx_[m + n][j] + sy_[n][i] * sy_[m + n][j];
    }
  }
  return out;
}


ScatteringFunctionMonitor::~ScatteringFunctionMonitor() {
  using namespace std;
  using namespace std::chrono;
  using namespace std::placeholders;
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
    qvecs[i] = qmax * (i / double(num_qvec-1));
  }

  vector<double> wpoints(100, 0.0);
  for (auto i = 0; i < wpoints.size(); ++i) {
    wpoints[i] = i * 1.0;
  }

  vector<unsigned> random_spins(512);
  for (unsigned int &random_spin : random_spins) {
    random_spin = jams::random_generator()(num_spins);
  }

  vector<Vec3> r(num_spins);
  for (auto i = 0; i < num_spins; ++i) {
    r[i] = lattice->atom_position(i);
  }

  vector<Complex> result(num_sub_samples + num_samples - 1, 0.0);
  vector<bool> is_vacancy(num_spins, false);
  for (auto i = 0; i < num_spins; ++i) {
    if (s(i, 0) == 0.0 && s(i, 1) == 0.0 && s(i, 2) == 0.0) {
      is_vacancy[i] = true;
    }
  }

  vector<vector<Complex>> SQw(qvecs.size());
  for (auto &i : SQw) {
    i = std::vector<Complex>(wpoints.size(), {0.0, 0.0});
  }

  const double delta_t = 50 * 1e-16 / 1e-12; // ps
  const double lambda = 0.01;

//  for (const auto i : random_spins) {
  for (unsigned i = 0; i < globals::num_spins; ++i) {
#pragma omp parallel for default(none) shared(SQw, globals::num_spins, lattice, r, qvecs, wpoints,i)
    for (unsigned j = 0; j < globals::num_spins; ++j) {
    if (is_vacancy[i]) continue;
    #pragma omp parallel for default(none) shared(SQw, globals::num_spins, lattice, r, qvecs, wpoints,i,is_vacancy)
      if (is_vacancy[j]) continue;
      const auto R = lattice->displacement(r[i], r[j]);
      const auto correlation = time_correlation(i, j, 300);

      for (auto q = 0; q < qvecs.size(); ++q) {
        const auto exp_QR = exp(kImagTwoPi * dot(qvecs[q], R));

        for (auto w = 0; w < wpoints.size(); ++w) {
          const Complex exp_wt = std::exp((kImagTwoPi * wpoints[w] - lambda) * delta_t);
          Complex exp_wt_n = exp_wt;

          Complex sum = correlation[0];
          for (auto n = 1; n < correlation.size(); ++n) {
            sum += correlation[n] * exp_wt_n;
            exp_wt_n *= exp_wt;
          }
#pragma omp critical
          {
            SQw[q][w] += -kImagOne * sum * exp_QR;
          };
        }
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
        cfile << Q[2] << " " << w << " " << real(SQw[nq][nw])/globals::num_spins << " " << imag(SQw[nq][nw])/globals::num_spins << std::endl;
      nw++;
    }
    nq++;
  }


}
