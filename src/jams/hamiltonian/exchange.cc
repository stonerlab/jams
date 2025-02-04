#include <fstream>
#include <set>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/output.h"

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {

  auto sparse_matrix_symmetry_check = read_sparse_matrix_symmetry_check_from_settings(settings, jams::SparseMatrixSymmetryCheck::Symmetric);

  neighbour_list_ = create_neighbour_list_from_settings(settings);

  print_neighbour_list_info(std::cout, neighbour_list_);

  if (debug_is_enabled()) {
    write_neighbour_list(jams::output::full_path_ofstream("DEBUG_exchange_nbr_list.tsv"), neighbour_list_);
  }

  for (auto n = 0; n < neighbour_list_.size(); ++n) {
    auto i = neighbour_list_[n].first[0];
    auto j = neighbour_list_[n].first[1];
    auto Jij = input_energy_unit_conversion_ * neighbour_list_[n].second;
    insert_interaction_tensor(i, j, Jij);
  }

  finalize(sparse_matrix_symmetry_check);
}

const jams::InteractionList<Mat3,2> &ExchangeHamiltonian::neighbour_list() const {
  return neighbour_list_;
}