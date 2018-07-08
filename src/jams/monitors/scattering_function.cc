//
// Created by Joe Barker on 2018/05/31.
//

#include <vector>
#include <complex>
#include <chrono>
#include <cmath>

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

namespace {
    template <typename T>
    T polynomial_sum(T x, const std::vector<double> &coeffs) {
      // Horner's method
      auto lambda = [&](T sum, double element){
        return sum * x + element;
      };
      return std::accumulate(coeffs.rbegin(), coeffs.rend(), T{0.0}, lambda);
    }
}


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


  const unsigned num_samples = sx_.size();
  const unsigned num_sub_samples = 200;

  const unsigned num_qvec = 129;
  const Vec3 qmax = {0.0, 0.0, 50.0};

  const unsigned num_w = 100;
  const double wmax = 100.0;

  vector<Vec3> qvecs(num_qvec, {0.0, 0.0, 0.0});
  for (auto i = 0; i < num_qvec; ++i){
    qvecs[i] = qmax * (i / double(num_qvec-1));
  }

  vector<double> wpoints(num_w, 0.0);
  for (auto i = 0; i < num_w; ++i) {
    wpoints[i] = wmax * (i / double(num_w-1));
  }

  vector<Vec3> r(num_spins);
  for (auto i = 0; i < num_spins; ++i) {
    r[i] = lattice->atom_position(i);
  }

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

  const double delta_t = output_step_freq_ * solver->real_time_step() / 1e-12; // ps
  const double lambda = 0.01;

  for (unsigned i = 0; i < globals::num_spins; ++i) {
    if (is_vacancy[i]) continue;

    std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " " << i << std::endl;

    #pragma omp parallel for default(none) shared(SQw, globals::num_spins, lattice, r, qvecs, wpoints,i,is_vacancy,std::cout)
    for (unsigned j = i; j < globals::num_spins; ++j) {
      if (is_vacancy[j]) continue;

      const auto R = lattice->displacement(r[i], r[j]);
      const auto correlation = time_correlation(i, j, num_sub_samples);

      vector<double> qfactors(qvecs.size());
      for (auto q = 0; q < qvecs.size(); ++q) {
        // cosine transform because we use R_ji = -R_ij
        qfactors[q] = 2.0 * cos(kTwoPi * dot(qvecs[q], R));
      }

      // TODO refactor w look and polynomial sum into FFT
      for (auto w = 0; w < wpoints.size(); ++w) {
        const Complex exp_wt = std::exp((kImagTwoPi * wpoints[w] - lambda) * delta_t);
        const auto sum = polynomial_sum(exp_wt, correlation);
        for (auto q = 0; q < qfactors.size(); ++q) {
#pragma omp critical
          {
            SQw[q][w] += -kImagOne * sum * qfactors[q];
          };
        }
      }
    }

    std::ofstream cfile(seedname + "_corr.tsv");
    for (auto q = 0; q < qvecs.size(); ++q) {
      for (auto w = 0; w < wpoints.size(); ++w) {
        cfile << qvecs[q][0] << " " << qvecs[q][1] << " " << qvecs[q][2] << " " << wpoints[w] << " " << real(SQw[q][w])/(i+1) << " " << imag(SQw[q][w])/(i+1) << "\n";
      }
    }
    cfile.flush();
    cfile.close();
  }

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "finish  " << get_date_string(end_time) << "\n\n";
  cout << "runtime " << duration_string(end_time - start_time) << "\n";
  cout.flush();

}
