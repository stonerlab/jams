#include <fstream>
#include "jams/monitors/spin_correlation.h"

#include "jams/core/globals.h"
#include "spin_correlation.h"
#include "jams/core/lattice.h"
#include "jams/interface/openmp.h"
#include "jams/helpers/output.h"

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

  const double eps = jams::defaults::lattice_tolerance; // (lattice constants) this should be quite large for comparing lattice distances

  auto comparison = [eps](const double& a, const double& b) { return definately_less_than(a, b, eps); };
  using histo_map = std::map<double, Datum<double>, decltype(comparison)>;

  histo_map out_of_plane_sz_corr_histogram_(comparison);
  histo_map in_plane_sz_corr_histogram_(comparison);

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = i + 1; j < globals::num_spins; ++j) {
      const auto r_ij = lattice->displacement(i, j);

      const auto do_out_of_plane = (approximately_zero(r_ij[0], eps) && approximately_zero(r_ij[1], eps));
      const auto do_in_plane = (approximately_zero(r_ij[2], eps));

      if (do_in_plane || do_out_of_plane) {

        auto sum = 0.0;
        #if HAS_OMP
        #pragma parallel for reduction(+:sum)
        #endif
        for (auto t = 0; t < num_samples_; ++t) {
          sum += sz_data_(i, t) * sz_data_(j, t);
        }

        const auto r_ij_sq = norm_sq(r_ij);

        if (do_in_plane) {
          in_plane_sz_corr_histogram_[r_ij_sq].count += num_samples_;
          in_plane_sz_corr_histogram_[r_ij_sq].total += sum;
        }

        if (do_out_of_plane) {
          out_of_plane_sz_corr_histogram_[r_ij_sq].count += num_samples_;
          out_of_plane_sz_corr_histogram_[r_ij_sq].total += sum;
        }
      }
    }
  }

  {
    ofstream of = jams::output::open_file(simulation_name + "_corr_outplane.tsv");
    of << "delta_r\tCzz\n";
    for (auto x : out_of_plane_sz_corr_histogram_) {
      auto delta_r = sqrt(x.first);
      auto Czz = (x.second.total / double(x.second.count));
      of << std::fixed << delta_r << "\t" << std::scientific << Czz << "\n";
    }
  }

  {
    ofstream of = jams::output::open_file(simulation_name + "_corr_inplane.tsv");
    of << "delta_r\tCzz\n";
    for (auto x : in_plane_sz_corr_histogram_) {
        auto delta_r = sqrt(x.first);
        auto Czz = (x.second.total / double(x.second.count));
        of << std::fixed << delta_r << "\t" << std::scientific << Czz << "\n";
    }
  }
}

