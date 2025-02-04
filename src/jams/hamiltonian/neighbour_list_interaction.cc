#include <jams/hamiltonian/neighbour_list_interaction.h>
#include <jams/helpers/output.h>
#include <fstream>
#include <jams/core/globals.h>

NeighbourListInteractionHamiltonian::NeighbourListInteractionHamiltonian(const libconfig::Setting &settings, unsigned int size)
: Hamiltonian(settings, size),
sparse_matrix_builder_(size, size)
{

}

void NeighbourListInteractionHamiltonian::insert_interaction_scalar(const int i, const int j, const double &value) {
  assert(!is_finalized_);
  if (value == 0.0) {
    return;
  }
  sparse_matrix_builder_.insert(i, j, value);
}

void NeighbourListInteractionHamiltonian::finalize(jams::SparseMatrixSymmetryCheck symmetry_check) {
  assert(!is_finalized_);

  if (debug_is_enabled()) {
    std::ofstream os(jams::output::full_path_filename("DEBUG_" + name() + "_spm.tsv"));
    sparse_matrix_builder_.output(os);
    os.close();
  }

  switch(symmetry_check) {
    case jams::SparseMatrixSymmetryCheck::None:
      break;
    case jams::SparseMatrixSymmetryCheck::Symmetric:
      if (!sparse_matrix_builder_.is_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not symmetric");
      }
    break;
    case jams::SparseMatrixSymmetryCheck::StructurallySymmetric:
      if (!sparse_matrix_builder_.is_structurally_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not structurally symmetric");
      }
    break;
  }

  interaction_matrix_ = sparse_matrix_builder_
      .set_format(jams::SparseMatrixFormat::CSR)
      .build();
  std::cout << "  " << name() << " sparse matrix memory (CSR): " << memory_in_natural_units(interaction_matrix_.memory()) << "\n";
  sparse_matrix_builder_.clear();
  is_finalized_ = true;
}

void NeighbourListInteractionHamiltonian::calculate_energies(double time) {
  assert(is_finalized_);
  // TODO: Add GPU support

#pragma omp parallel for
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}


double NeighbourListInteractionHamiltonian::calculate_total_energy(double time) {
  using namespace globals;
  assert(is_finalized_);

  calculate_fields(time);
  double total_energy = 0.0;
#if HAS_OMP
#pragma omp parallel for default(none) shared(num_spins, s, field_) reduction(+:total_energy)
#endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
    Vec3 h_i = {field_(i,0), field_(i, 1), field_(i, 2)};
    total_energy += -dot(s_i, 0.5*h_i);
  }
  return 0.5 * total_energy;
}

double NeighbourListInteractionHamiltonian::calculate_energy(int i, double time) {
  using namespace globals;
  assert(is_finalized_);
  Vec3 s_i = {s(i,0), s(i,1), s(i,2)};
  auto field = calculate_field(i, time);
  return -0.5*dot(s_i, field);
}

double NeighbourListInteractionHamiltonian::calculate_energy_difference(int i,
                                                                        const Vec3 &spin_initial,
                                                                        const Vec3 &spin_final,
                                                                        double time) {

  assert(is_finalized_);
  auto field = calculate_field(i, time);
  auto e_initial = -dot(spin_initial, 0.5*field);
  auto e_final = -dot(spin_final, 0.5*field);
  return e_final - e_initial;

}
