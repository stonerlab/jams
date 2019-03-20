#include <fstream>
#include "jams/monitors/spin_correlation.h"

#include "jams/core/globals.h"
#include "spin_correlation.h"
#include "jams/core/lattice.h"

using namespace std;

SpinCorrelationMonitor::SpinCorrelationMonitor(const libconfig::Setting &settings) :
Monitor(settings){

  libconfig::Setting& solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  double t_sample = output_step_freq_*t_step;
  num_samples_ = ceil(t_run/t_sample);

  spin_data_.resize(globals::num_spins, num_samples_);
}

void SpinCorrelationMonitor::update(Solver *solver) {
  using namespace globals;

  if (time_point_counter_ < num_samples_) {
    for (auto i = 0; i < num_spins; ++i) {
      spin_data_(i, time_point_counter_) = {float(s(i,0)), float(s(i,1)), float(s(i,2))};
    }
  }

  time_point_counter_++;
}

void SpinCorrelationMonitor::post_process() {
  using namespace globals;

  // calculate average sz of each position in unit cell
  vector<double> avg_sz(::lattice->num_motif_atoms(), 0.0);

  for (auto i = 0; i < num_spins; ++i) {
    auto n = lattice->atom_motif_position(i);
    for (auto t = 0; t < num_samples_; ++t) {
      avg_sz[n] += spin_data_(i, t)[2];
    }
  }

  for (double &n : avg_sz) {
    n /= double(product(lattice->size()) * num_samples_);
  }

  // subtract from the spin data
  for (auto i = 0; i < num_spins; ++i) {
    auto n = lattice->atom_motif_position(i);
    for (auto t = 0; t < num_samples_; ++t) {
      spin_data_(i, t)[2] -= avg_sz[n];
    }
  }

  // calculate correlation function

  using histo_map = std::map<double, histo, float_compare>;
  histo_map out_of_plane_sz_corr_histogram_;
  histo_map in_plane_sz_corr_histogram_;

  const double eps = 1e-5;
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = i + 1; j < globals::num_spins; ++j) {
//      if (i == j) continue;
      const auto r = lattice->displacement(lattice->atom_position(i), lattice->atom_position(j));

      // in the same column
      if (abs(r[0]) < eps && abs(r[1]) < eps) {
        for (auto t = 0; t < num_samples_; ++t) {
          const auto siz_sjz = spin_data_(i, t)[2] * spin_data_(j, t)[2];
          const auto siplus_sj_minus = complex<float>{spin_data_(i, t)[0], spin_data_(i, t)[1]} * complex<float>{spin_data_(j, t)[0], -spin_data_(j, t)[1]};

          out_of_plane_sz_corr_histogram_[abs_sq(r)].count++;
          out_of_plane_sz_corr_histogram_[abs_sq(r)].Szz += siz_sjz;
          out_of_plane_sz_corr_histogram_[abs_sq(r)].Szz_sq += square(siz_sjz);
          out_of_plane_sz_corr_histogram_[abs_sq(r)].S_plus_minus += siplus_sj_minus;

        }
      }

      // in the same plane
      if (abs(r[2]) < eps) {
        for (auto t = 0; t < num_samples_; ++t) {
          const auto siz_sjz = spin_data_(i, t)[2] * spin_data_(j, t)[2];
          const auto siplus_sj_minus = complex<float>{spin_data_(i, t)[0], spin_data_(i, t)[1]} * complex<float>{spin_data_(j, t)[0], -spin_data_(j, t)[1]};

          in_plane_sz_corr_histogram_[abs_sq(r)].count++;
          in_plane_sz_corr_histogram_[abs_sq(r)].Szz += siz_sjz;
          in_plane_sz_corr_histogram_[abs_sq(r)].Szz_sq += square(siz_sjz);
          in_plane_sz_corr_histogram_[abs_sq(r)].S_plus_minus += siplus_sj_minus;
        }
      }
    }
  }

  {
    ofstream of(seedname + "_corr_outplane.tsv");
    for (auto x : out_of_plane_sz_corr_histogram_) {
      auto delta_r = sqrt(x.first);
      auto Czz = (x.second.Szz / double(x.second.count));
      auto C_plus_minus =  (x.second.S_plus_minus / double(x.second.count));
      auto Czz_sq = (x.second.Szz_sq / double(x.second.count));
      auto Czz_stddev = sqrt(Czz_sq - square(Czz));
      auto Czz_stderr = Czz_stddev / sqrt(num_samples_);
      of << delta_r << "\t" << Czz << "\t" << Czz_stderr << "\t" << C_plus_minus.real() << "\t" << C_plus_minus.imag() << "\n";
    }
  }

  {
    ofstream of(seedname + "_corr_inplane.tsv");
    for (auto x : in_plane_sz_corr_histogram_) {
      auto delta_r = sqrt(x.first);
      auto Czz = (x.second.Szz / double(x.second.count));
      auto C_plus_minus =  (x.second.S_plus_minus / double(x.second.count));
      auto Czz_sq = (x.second.Szz_sq / double(x.second.count));
      auto Czz_stddev = sqrt(Czz_sq - square(Czz));
      auto Czz_stderr = Czz_stddev / sqrt(num_samples_);
      of << delta_r << "\t" << Czz << "\t" << Czz_stderr << "\t" << C_plus_minus.real() << "\t" << C_plus_minus.imag() << "\n";
    }
  }
}

