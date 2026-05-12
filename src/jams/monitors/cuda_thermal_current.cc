//
// Created by Joe Barker on 2018/04/20.
//

#include <jams/helpers/exception.h>
#include <jams/core/interactions.h>
#include <jams/helpers/error.h>
#include <jams/helpers/consts.h>
#include <jams/helpers/maths.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/helpers/output.h"
#include "jams/core/globals.h"
#include "jams/interface/config.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/monitors/cuda_thermal_current.h"
#include "jams/cuda/cuda_common.h"
#include "jams/containers/csr.h"
#include "jams/hamiltonian/exchange.h"
#include "cuda_thermal_current.h"
#include "../core/globals.h"

CudaThermalCurrentMonitor::CudaThermalCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {
  jams_warning("This monitor automatically identifies the FIRST exchange hamiltonian\n"
               "in the config and assumes the exchange interaction is DIAGONAL AND ISOTROPIC");
  jams_warning("This monitor should only be used with collinear systems which are magnetised along the z direction.");

  const auto& exchange_hamiltonian = find_hamiltonian<ExchangeHamiltonian>(::globals::solver->hamiltonians());

  const auto& nbr_list = exchange_hamiltonian.neighbour_list();


  const auto triad_list = generate_three_spin_from_two_spin_interactions(exchange_hamiltonian.neighbour_list());

  std::cout << "    total ijk triads: " << triad_list.size() << std::endl;

  std::cout << "    interaction matrix memory: " << interaction_matrix_.memory() / kBytesToMegaBytes << "MB" << std::endl;

  zero(thermal_current_rx_.resize(globals::num_spins));
  zero(thermal_current_ry_.resize(globals::num_spins));
  zero(thermal_current_rz_.resize(globals::num_spins));

  tsv_.open(jams::output::monitor_filename(name(), "tsv"),
            {{"time", "picoseconds"},
             {"jq_rx", "internal"},
             {"jq_ry", "internal"},
             {"jq_rz", "internal"}});
}

void CudaThermalCurrentMonitor::update(Solver& solver) {
  const auto& spins = globals::s;
  jams::Vec<double, 3> js = execute_cuda_thermal_current_kernel(
          stream, spins, interaction_matrix_, thermal_current_rx_, thermal_current_ry_, thermal_current_rz_);

  tsv_.write_row_values(solver.time(), js[0], js[1], js[2]);
}

CudaThermalCurrentMonitor::~CudaThermalCurrentMonitor() {
}

CudaThermalCurrentMonitor::ThreeSpinList CudaThermalCurrentMonitor::generate_three_spin_from_two_spin_interactions(const jams::InteractionList<jams::Mat<double, 3, 3>, 2>& nbr_list) {
  ThreeSpinList three_spin_list;

  // Jij * Jjk
  for (auto i = 0; i < globals::num_spins; ++i) {
    // ij
    for (auto const &nbr_j: nbr_list.interactions_of(i)) {
      const auto j = nbr_j.first[1];
      const auto Jij = nbr_j.second[0][0];
      // jk
      for (auto const &nbr_k: nbr_list.interactions_of(j)) {
        const int k = nbr_k.first[1];
        const auto Jjk = nbr_k.second[0][0];
        if (i == j || j == k || i == k) continue;
        if (i > j || j > k || i > k) continue;
        three_spin_list.insert({i, j, k}, Jij * Jjk * globals::lattice->displacement(i, k));
      }
    }
  }

  // Jij * Jik
  for (auto i = 0; i < globals::num_spins; ++i) {
    // ij
    for (auto const &nbr_j: nbr_list.interactions_of(i)) {
      const auto j = nbr_j.first[1];
      const auto Jij = nbr_j.second[0][0];
      // ik
      for (auto const &nbr_k: nbr_list.interactions_of(i)) {
        const auto k = nbr_k.first[1];
        const auto Jik = nbr_k.second[0][0];
        if (i == j || j == k || i == k) continue;
        if (i > j || j > k || i > k) continue;
        three_spin_list.insert({i, j, k}, Jij * Jik * globals::lattice->displacement(j, i));
      }
    }
  }

//  // Jik * Jjk
  for (auto i = 0; i < globals::num_spins; ++i) {
    // ik
    for (auto const &nbr_k: nbr_list.interactions_of(i)) {
      const auto  k = nbr_k.first[1];
      const auto Jik = nbr_k.second[0][0];
      // jk
      for (auto const &nbr_j: nbr_list.interactions_of(k)) {
        const auto  j = nbr_j.first[1];
        const auto Jjk = nbr_j.second[0][0];
        if (i == j || j == k || i == k) continue;
        if (i > j || j > k || i > k) continue;
        three_spin_list.insert({i, j, k}, Jik * Jjk * globals::lattice->displacement(k, j));
      }
    }
  }

  interaction_matrix_ = jams::InteractionMatrix<jams::Vec<double, 3>, double>(three_spin_list, globals::num_spins);

  return three_spin_list;
}
