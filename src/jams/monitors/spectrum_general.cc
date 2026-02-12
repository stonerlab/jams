//
// Created by Joe Barker on 2018/05/31.
//

#include <vector>
#include <complex>
#include <chrono>
#include <cmath>

#include "jams/monitors/spectrum_general.h"

#include <libconfig.h++>
#include <jams/common.h>
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/interface/fft.h"
#include "jams/helpers/stats.h"
#include "jams/helpers/duration.h"
#include "jams/helpers/random.h"
#include "jams/core/lattice.h"
#include "spectrum_general.h"
#include "jams/helpers/output.h"
#include <jams/helpers/mixed_precision.h>

namespace {
    std::vector<jams::ComplexHi> generate_expQR(const std::vector<Vec3> &qvecs, const Vec3& R) {
      std::vector<jams::ComplexHi> result(qvecs.size());
      for (auto q = 0; q < result.size(); ++q) {
        result[q] = exp(kImagTwoPi * jams::dot(qvecs[q], R));
      }
      return result;
    }
}



SpectrumGeneralMonitor::SpectrumGeneralMonitor(const libconfig::Setting &settings)
: Monitor(settings),
outfile(jams::output::full_path_filename("fk.tsv")){
  libconfig::Setting& solver_settings = ::globals::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  double t_sample = output_step_freq_*t_step;
  num_samples_ = ceil(t_run/t_sample);
  padded_size_ = 2*num_samples_ - 1;
  double freq_max    = 1.0/(2.0*t_sample);
  freq_delta_  = 1.0/(num_samples_*t_sample);

  std::cout << "\n";
  std::cout << "  number of samples " << num_samples_ << "\n";
  std::cout << "  sampling time (s) " << t_sample << "\n";
  std::cout << "  acquisition time (s) " << t_sample * num_samples_ << "\n";
  std::cout << "  frequency resolution (THz) " << freq_delta_ << "\n";
  std::cout << "  maximum frequency (THz) " << freq_max << "\n";
  std::cout << "\n";

  num_qvectors_ = jams::config_required<unsigned>(settings, "num_qvectors");
  num_qpoints_ = jams::config_required<unsigned>(settings, "num_qpoints");
  qmax_      = jams::config_required<double>(settings, "qmax");

  spin_data_.resize(globals::num_spins, padded_size_);
  spin_data_.zero();
}

void SpectrumGeneralMonitor::update(Solver& solver) {
  if (time_point_counter_ < num_samples_) {
    for (auto i = 0; i < globals::num_spins; ++i) {
      spin_data_(i, time_point_counter_) = jams::ComplexHi{globals::s(i, 0), globals::s(i, 1)};
    }
  }
  time_point_counter_++;
}

void SpectrumGeneralMonitor::apply_time_fourier_transform() {

  // window the data in time space
  for (auto i = 0; i < spin_data_.size(0); ++i) {
    for (auto n = 0; n < num_samples_; ++n) {
      spin_data_(i, n) *= fft_window_exponential(n, num_samples_);
    }
  }

  // Fourier transform each spin's time series
  // we go from S_i(t) -> S_i(w)
  int rank            = 1;
  int stride          = 1;
  int dist            = (int) padded_size_; // num_samples
  int num_transforms  = (int) spin_data_.size(0); // num_spins
  int transform_size[1]  = {(int) padded_size_};

  int * nembed = nullptr;

  // FFTW_BACKWARD is used so the sign convention is consistent with Alben AIP Conf. Proc. 29 136 (1976)
  auto plan = fftw_plan_many_dft(
          rank,                    // dimensionality
          transform_size, // array of sizes of each dimension
          num_transforms,          // number of transforms
          reinterpret_cast<fftw_complex*>(spin_data_.data()),        // input: real data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          reinterpret_cast<fftw_complex*>(spin_data_.data()),        // output: complex data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          FFTW_BACKWARD,
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

  fftw_execute(plan);

  fftw_destroy_plan(plan);
}

SpectrumGeneralMonitor::~SpectrumGeneralMonitor() {
  using namespace std::chrono;
  using namespace std::placeholders;

  std::cout << "calculating correlation function" << std::endl;
  auto start_time = time_point_cast<milliseconds>(system_clock::now());
  std::cout << "start   " << get_date_string(start_time) << "\n\n";
  std::cout.flush();

  std::cout << duration_string(start_time, system_clock::now()) << " calculating fft time => frequency" << std::endl;

  apply_time_fourier_transform();

  std::cout << duration_string(start_time, system_clock::now()) << " done" << std::endl;


  std::vector<std::vector<Vec3>> qvecs(num_qpoints_);
  for (auto n = 0; n < num_qvectors_; ++n) {
    auto qvec_rand = qmax_ * uniform_random_sphere<double>(jams::instance().random_generator());
    std::vector<Vec3> qpoints(num_qvectors_);
    for (auto i = 0; i < num_qpoints_; ++i){
      qpoints[i] = qvec_rand * (i / double(num_qpoints_-1));
    }
    qvecs.push_back(qpoints);
  }

  // support for lattice vacancies (we will skip these in the spectrum loop)
  std::vector<bool> is_vacancy(globals::num_spins, false);
  for (auto i = 0; i < globals::num_spins; ++i) {
    if (globals::s(i, 0) == 0.0 && globals::s(i, 1) == 0.0 && globals::s(i, 2) == 0.0) {
      is_vacancy[i] = true;
    }
  }

  jams::MultiArray<jams::ComplexHi, 2> SQw(qvecs.size(), padded_size_/2+1);
  SQw.zero();

  // generate spectrum looping over all i,j
  for (unsigned i = 0; i < globals::num_spins; ++i) {
    if (is_vacancy[i]) continue;
    std::cout << duration_string(time_point_cast<milliseconds>(system_clock::now()) - start_time) << " " << i << std::endl;
    for (unsigned j = 0; j < globals::num_spins; ++j) {
      if (is_vacancy[j]) continue;
      for (unsigned n = 0; n < qvecs.size(); ++n) {
        // precalculate the exponential factors for the spatial fourier transform
        const auto qfactors = generate_expQR(qvecs[n], globals::lattice->displacement(j, i));

        for (unsigned w = 0; w < padded_size_ / 2 + 1; ++w) {
          for (unsigned q = 0; q < qfactors.size(); ++q) {
            // the spin_data_ multiplication uses convolution theory to avoid first calculating the time correlation
            // (padded_size_ - w) % padded_size_ gives the -w data
            SQw(q, w) += -kImagOne * qfactors[q] * spin_data_(i, w) * spin_data_(j, (padded_size_ - w) % padded_size_);
          }
        }
      }
    }

    if (i%10 == 0) {
      std::ofstream cfile(jams::output::full_path_filename("corr.tsv"));
      for (unsigned q = 0; q < qvecs.size(); ++q) {
        for (unsigned w = 0; w < padded_size_/2+1; ++w) {
          cfile << qmax_ * (q / double(num_qpoints_-1)) << " " << 0.5*w * freq_delta_ << " " << -SQw(q, w).imag() / (i + 1)/ static_cast<double>(padded_size_) << "\n";
        }
      }
      cfile.flush();
      cfile.close();
    }

  }

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  std::cout << "finish  " << get_date_string(end_time) << "\n\n";
  std::cout << "runtime " << duration_string(start_time, end_time) << "\n";
  std::cout.flush();
}
