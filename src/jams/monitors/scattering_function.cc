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

//  std::vector<double> sxt(num_spins);
//  std::vector<double> syt(num_spins);
//  std::vector<double> szt(num_spins);
//
//  for (auto i = 0; i < num_spins; ++i) {
//    sxt[i] = s(i, 0);
//    syt[i] = s(i, 1);
//    szt[i] = s(i, 2);
//  }
//
//  sx_.push_back(sxt);
//  sy_.push_back(syt);
//  sz_.push_back(szt);

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

//  const unsigned num_kpoints = 64;
//  const double rmax = lattice->max_interaction_radius();
//  const double kmax = 4;
//
//
//  vector<double> f_k(num_kpoints, 0.0);
//  vector<double> kvec(num_kpoints, 0.0);
//  for (auto i = 0; i < num_kpoints; ++i) {
//    kvec[i] = i * kmax/double(num_kpoints);
//  }
//
//  for (auto i = 0; i < num_spins; ++i) {
//    const Vec3 s_i = {s(i, 0), s(i, 1), s(i, 2)};
//    const Vec3 r_i = lattice->atom_position(i);
//
//    for (auto j = 0; j < num_spins; ++j) {
//      if (i == j) continue;
//
//      const Vec3 r_j = lattice->atom_position(j);
//      const auto r_ij = lattice->displacement(r_i, r_j);
//      const auto r = abs(r_ij);
//
//      if (r > rmax) continue;
//
//      const Vec3 s_j = {s(j, 0), s(j, 1), s(j, 2)};
//
//      auto xhat = normalize(r_ij);
//      auto yhat = normalize(s_i - xhat * dot(s_i, xhat));
//
//      auto S_ix = dot(s_i, xhat);
//      auto S_iy = dot(s_i, yhat);
//
//      auto S_jx = dot(s_j, xhat);
//      auto S_jy = dot(s_j, yhat);
//
//      auto Sxx = S_ix * S_jx;
//      auto Syy = S_iy * S_jy;
//
//      for (auto k = 0; k < num_kpoints; ++k) {
//        f_k[k] += Syy * (sin(kvec[k] * r ) / r) + (2*Sxx - Syy) * ((sin(kvec[k] * r ) / pow3(kvec[k]*r) ) - (cos(kvec[k] * r ) / pow2(kvec[k] * r)));
//      }
//    }
//  }
//
//  for (auto k = 0; k < num_kpoints; ++k) {
//    outfile << solver->time() << " " << k << " " << kvec[k] << " " << fx_k[k].real() << " " << fy_k[k].real() << std::endl;
//  }
//  outfile << "\n" << std::endl;
}

bool ScatteringFunctionMonitor::is_converged() {
  return false;
}

ScatteringFunctionMonitor::~ScatteringFunctionMonitor() {
  using namespace globals;
//  std::ofstream freqfile("dsf.tsv");
//
//  double freq_delta  = 1.0/(num_samples_*t_sample_);
//
//  double kmax = 0.5;
//  std::vector<double> kvec(num_kpoints_, 0.0);
//  for (auto i = 0; i < num_kpoints_; ++i) {
//    kvec[i] = i * kmax/double(num_kpoints_);
//  }
//
//  unsigned num_wpoints_ = num_samples_;
//
//  for (auto kk = 0; kk < num_kpoints_; ++kk) {
//    for (auto ww = 0; ww < num_samples_ / 2 + 1; ++ww) {
//      std::complex<double> total = {0.0, 0.0};
//      const Vec3 k = {0, 0, kvec[kk]};
//      const double omega = ww * freq_delta;
//      for (auto m = 0; m < num_samples_; ++m) {
//      for (auto n = 0; n < 1; ++n) {
//        for (auto i = 0; i < num_spins; ++i) {
//          for (auto j = 0; j < num_spins; ++j) {
//            const Vec3 r_i = lattice->atom_position(i);
//            const Vec3 r_j = lattice->atom_position(j);
//            double t = (m - n) * t_sample_;
//            total += sx_[m][i] * sx_[num_samples_-1][j] * std::exp(kImagTwoPi * (dot(k, r_i - r_j) + omega * t));
//          }
//        }
//      }
//      }
//      freqfile << kk << " " << omega << " " << total.real() << " " << total.imag() << std::endl;
//    }
//    freqfile << std::endl;
//  }


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
