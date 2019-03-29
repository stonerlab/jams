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

  sz_data_.resize(globals::num_spins, num_samples_);
}

void SpinCorrelationMonitor::update(Solver *solver) {
  using namespace globals;

  if (time_point_counter_ < num_samples_) {
    for (auto i = 0; i < num_spins; ++i) {
      sz_data_(i, time_point_counter_) = static_cast<float>(s(i,2));
    }
  }

  time_point_counter_++;
}

#pragma GCC optimize ("Ofast")
void SpinCorrelationMonitor::post_process() {
  using namespace globals;

  // calculate average sz of each position in unit cell
  vector<double> avg_sz(::lattice->num_motif_atoms(), 0.0);

  for (auto i = 0; i < num_spins; ++i) {
    auto n = lattice->atom_motif_position(i);
    for (auto t = 0; t < num_samples_; ++t) {
      avg_sz[n] += sz_data_(i, t);
    }
  }

  for (double &n : avg_sz) {
    n /= double(product(lattice->size()) * num_samples_);
  }

  // subtract from the spin data
  for (auto i = 0; i < num_spins; ++i) {
    auto n = lattice->atom_motif_position(i);
    for (auto t = 0; t < num_samples_; ++t) {
      sz_data_(i, t) -= avg_sz[n];
    }
  }

  // calculate correlation function

  using histo_map = std::map<double, Datum<double>, float_compare>;
  histo_map out_of_plane_sz_corr_histogram_;
  histo_map in_plane_sz_corr_histogram_;

  const double eps = 1e-5;
  for (auto i = 0; i < globals::num_spins; ++i) {

      const auto r_i = lattice->atom_position(i);

    for (auto j = i + 1; j < globals::num_spins; ++j) {
//      if (i == j) continue;
      const auto r_ij = lattice->displacement(r_i, lattice->atom_position(j));

      // in the same column
      if (abs(r_ij[0]) < eps && abs(r_ij[1]) < eps) {
        for (auto t = 0; t < num_samples_; ++t) {
          out_of_plane_sz_corr_histogram_[abs_sq(r_ij)].count++;
          out_of_plane_sz_corr_histogram_[abs_sq(r_ij)].total += sz_data_(i, t) * sz_data_(j, t);
        }
      }

      // in the same plane
      if (abs(r_ij[2]) < eps) {
        for (auto t = 0; t < num_samples_; ++t) {
          in_plane_sz_corr_histogram_[abs_sq(r_ij)].count++;
          in_plane_sz_corr_histogram_[abs_sq(r_ij)].total += sz_data_(i, t) * sz_data_(j, t);
        }
      }
    }
  }

  {
    ofstream of(seedname + "_corr_outplane.tsv");
    of << "delta_r\tCzz\n";
    for (auto x : out_of_plane_sz_corr_histogram_) {
      auto delta_r = sqrt(x.first);
      auto Czz = (x.second.total / double(x.second.count));
      of << delta_r << "\t" << Czz << "\n";
    }
  }

  {
    ofstream of(seedname + "_corr_inplane.tsv");
    of << "delta_r\tCzz\n";
    for (auto x : in_plane_sz_corr_histogram_) {
        auto delta_r = sqrt(x.first);
        auto Czz = (x.second.total / double(x.second.count));
        of << delta_r << "\t" << Czz << "\n";
    }
  }
}

