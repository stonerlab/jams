#include <fstream>
#include <set>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/output.h"
#include "jams/helpers/interaction_list_helpers.h"


ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : SparseInteractionHamiltonian(settings, size) {

  neighbour_list_ = jams::InteractionListFromSettings(settings);

  jams::PrintInteractionListProperties(std::cout, neighbour_list_);

  if (debug_is_enabled()) {
    std::ofstream of(jams::output::full_path_filename(name() + "_hamiltonian_nbr.tsv"));
    jams::PrintInteractionList(of, neighbour_list_);
    of.close();
  }

  for (auto n = 0; n < neighbour_list_.size(); ++n) {
    auto i = neighbour_list_[n].first[0];
    auto j = neighbour_list_[n].first[1];
    auto Jij = neighbour_list_[n].second;
    insert_interaction_tensor(i, j, Jij);
  }

  jams::SparseMatrixSymmetryCheck sparse_matrix_checks = jams::SparseMatrixSymmetryCheck::Symmetric;

  if (settings.exists("check_sparse_matrix_symmetry")) {
    if (bool(settings["check_sparse_matrix_symmetry"]) == false) {
      sparse_matrix_checks = jams::SparseMatrixSymmetryCheck::None;
    }
  }

  finalize(sparse_matrix_checks);
}

const jams::InteractionList<Mat3,2> &ExchangeHamiltonian::neighbour_list() const {
  return neighbour_list_;
}