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

using Complex = std::complex<double>;

namespace {

    inline Complex fast_multiply(const Complex &x, const Complex &y) {
      return {x.real() * y.real() - x.imag() * y.imag(),
        x.real()*y.imag() + x.imag() * y.real()};
    }

    inline Complex fast_multiply(const fftw_complex &x, const fftw_complex &y) {
      return {x[0] * y[0] - x[1] * y[1],
              x[0] * y[1] + x[1] * y[0]};
    }

    std::vector<double> generate_expQR(const std::vector<Vec3> &qvecs, const Vec3& R) {
      std::vector<double> result(qvecs.size());
      for (auto q = 0; q < result.size(); ++q) {
        // cosine transform because we use R_ji = -R_ij
        result[q] = 2.0 * cos(kTwoPi * dot(qvecs[q], R));
      }
      return result;
    }
}



ScatteringFunctionMonitor::ScatteringFunctionMonitor(const libconfig::Setting &settings) : Monitor(settings) {
  using namespace std;

  std::string name = seedname + "_fk.tsv";
  outfile.open(name.c_str());

  libconfig::Setting& solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  double t_sample = output_step_freq_*t_step;
  num_samples_ = ceil(t_run/t_sample);
  double freq_max    = 1.0/(2.0*t_sample);
  freq_delta_  = 1.0/(num_samples_*t_sample);

  cout << "\n";
  cout << "  number of samples " << num_samples_ << "\n";
  cout << "  sampling time (s) " << t_sample << "\n";
  cout << "  acquisition time (s) " << t_sample * num_samples_ << "\n";
  cout << "  frequency resolution (THz) " << freq_delta_/kTHz << "\n";
  cout << "  maximum frequency (THz) " << freq_max/kTHz << "\n";
  cout << "\n";


  num_q_ = jams::config_required<unsigned>(settings, "num_q");
  qmax_      = jams::config_required<Vec3>(settings, "qmax");

  spin_data_.resize(globals::num_spins, num_samples_);
  spin_data_.zero();
}

void ScatteringFunctionMonitor::update(Solver *solver) {
  using namespace globals;
  using namespace std;

  if (time_point_counter_ < num_samples_) {
    for (auto i = 0; i < num_spins; ++i) {
      spin_data_(i, time_point_counter_)[0] = s(i, 0) * fft_window_default(time_point_counter_, num_samples_);
      spin_data_(i, time_point_counter_)[1] = s(i, 1) * fft_window_default(time_point_counter_, num_samples_);
    }
  }

  time_point_counter_++;
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

  cout << "calculating correlation function" << std::endl;
  auto start_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "start   " << get_date_string(start_time) << "\n\n";
  cout.flush();

  //---------------------------------------------------------------------
  int rank            = 1;
  int stride          = 1;
  int dist            = (int) num_samples_; // num_samples
  int num_transforms  = (int) globals::num_spins; // num_spins
  int transform_size[1]  = {(int) num_samples_};

  int * nembed = nullptr;

  std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " planning fft" << std::endl;

  auto plan = fftw_plan_many_dft(
          rank,                    // dimensionality
          transform_size, // array of sizes of each dimension
          num_transforms,          // number of transforms
          spin_data_.data(),        // input: real data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          spin_data_.data(),        // output: complex data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          FFTW_FORWARD,
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

  std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " done" << std::endl;

  std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " executing fft" << std::endl;

  fftw_execute(plan);

  std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " done" << std::endl;


  //---------------------------------------------------------------------

  vector<Vec3> qvecs(num_q_, {0.0, 0.0, 0.0});
  for (auto i = 0; i < num_q_; ++i){
    qvecs[i] = qmax_ * (i / double(num_q_-1));
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

  jblib::Array<Complex, 2> SQw(qvecs.size(), num_samples_ / 2 + 1);
  SQw.zero();

  for (unsigned i = 0; i < globals::num_spins; ++i) {
    if (is_vacancy[i]) continue;
    std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " " << i << std::endl;
    for (unsigned j = i; j < globals::num_spins; ++j) {
      if (is_vacancy[j]) continue;

      const auto qfactors = generate_expQR(qvecs, lattice->displacement(r[i], r[j]));

      #pragma omp parallel for default(none) shared(SQw,i,j)
      for (unsigned w = 0; w < num_samples_ / 2 + 1; ++w) {
        const Complex correlation = fast_multiply(spin_data_(i,w), spin_data_(j,w));
        for (unsigned q = 0; q < qfactors.size(); ++q) {
          SQw(q,w) += -kImagOne * qfactors[q] * correlation;
        }
      }
    }

    if (i%10 == 0) {
      std::ofstream cfile(seedname + "_corr.tsv");
      for (unsigned q = 0; q < qvecs.size(); ++q) {
        for (unsigned w = 0; w < num_samples_ / 2 + 1; ++w) {
          cfile << qvecs[q][0] << " " << qvecs[q][1] << " " << qvecs[q][2] << " " << w * freq_delta_ << " "
                << real(SQw(q, w)) / (i + 1) << " " << imag(SQw(q, w)) / (i + 1) << "\n";
        }
      }
      cfile.flush();
      cfile.close();
    }

  }

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "finish  " << get_date_string(end_time) << "\n\n";
  cout << "runtime " << duration_string(end_time - start_time) << "\n";
  cout.flush();

}
