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
#include "jams/core/lattice.h"

ScatteringFunctionMonitor::ScatteringFunctionMonitor(const libconfig::Setting &settings) : Monitor(settings) {
  std::string name = seedname + "_fk.tsv";
  outfile.open(name.c_str());

  libconfig::Setting& solver_settings = ::config->lookup("solver");

  num_kpoints_ = 17;

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  t_sample_ = output_step_freq_*t_step;
  num_samples_ = ceil(t_run/t_sample_);
  double freq_max    = 1.0/(2.0*t_sample_);
  double freq_delta  = 1.0/(num_samples_*t_sample_);
}

void ScatteringFunctionMonitor::update(Solver *solver) {
  using namespace globals;
  using namespace std;

  const double kmax = 0.5;
  vector<complex<double>> fx_k(num_kpoints_, {0.0, 0.0});
  vector<complex<double>> fy_k(num_kpoints_, {0.0, 0.0});

  vector<double> kvec(num_kpoints_, 0.0);
  for (auto i = 0; i < num_kpoints_; ++i) {
    kvec[i] = i * kmax/double(num_kpoints_);
  }

  for (auto k = 0; k < num_kpoints_; ++k) {
    Vec3 kz = {0, 0, kvec[k]};

    for (auto j = 0; j < num_spins; ++j) {
      const Vec3 r_j = lattice->atom_position(j);
      const Vec3 s_j = {s(j, 0), s(j, 1), s(j, 2)};

        fx_k[k] += s_j[0] * exp(-kImagTwoPi * dot(kz, r_j));
        fy_k[k] += s_j[1] * exp(-kImagTwoPi * dot(kz, r_j));
      }
  }

  for (auto k = 0; k < num_kpoints_; ++k) {
    fx_k[k] /= sqrt(num_spins);
    fy_k[k] /= sqrt(num_spins);
  }

  fx_t.insert(fx_t.end(), fx_k.begin(), fx_k.end());
  fy_t.insert(fy_t.end(), fy_k.begin(), fy_k.end());

}

bool ScatteringFunctionMonitor::is_converged() {
  return false;
}

ScatteringFunctionMonitor::~ScatteringFunctionMonitor() {
  using namespace globals;

  const int kpoints = num_kpoints_;
  const int time_points = static_cast<int>(fx_t.size() / kpoints);

  for (auto i = 0; i < time_points; ++i) {
    for (auto j = 0; j < kpoints; ++j) {
      fx_t[kpoints * i + j] *= fft_window_default(i, time_points);
      fy_t[kpoints * i + j] *= fft_window_default(i, time_points);
    }
  }
  
  auto fx_w = fx_t;
  auto fy_w = fy_t;

  int rank       = 1;
  int sizeN[]   = {time_points};
  int howmany    = kpoints;
  int inembed[] = {time_points}; int onembed[] = {time_points};
  int istride    = kpoints; int ostride    = kpoints;
  int idist      = 1;            int odist      = 1;


  fftw_plan fft_plan_time_x = fftw_plan_many_dft(rank,sizeN,howmany,
          reinterpret_cast<fftw_complex*>(fx_w.data()),inembed,istride,idist,
          reinterpret_cast<fftw_complex*>(fx_w.data()),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan fft_plan_time_y = fftw_plan_many_dft(rank,sizeN,howmany,
          reinterpret_cast<fftw_complex*>(fy_w.data()),inembed,istride,idist,
          reinterpret_cast<fftw_complex*>(fy_w.data()),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);


  fftw_execute(fft_plan_time_x);
  fftw_execute(fft_plan_time_y);

  std::ofstream freqfile("dsf.tsv");

  for (auto i = 0; i < (time_points / 2) + 1; ++i) {
    for (auto j = 0; j < kpoints; ++j) {
      freqfile << j << " " << i << " " << fx_w[kpoints * i + j].real() << " " << fx_w[kpoints * i + j].imag() << " " << fy_w[kpoints * i + j].real() << " " << fy_w[kpoints * i + j].imag() << "\n";
    }
    freqfile << std::endl;
  }

}
