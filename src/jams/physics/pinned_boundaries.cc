// pinned_boundaries.cc                                                -*-C++-*-
#include <jams/physics/pinned_boundaries.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/spinops.h>
#include <jams/cuda/cuda_spin_ops.h>
#include <jams/cuda/cuda_array_reduction.h>


PinnedBoundariesPhysics::PinnedBoundariesPhysics(const libconfig::Setting &settings)
: Physics(settings)
, left_pinned_magnetisation_(
    jams::config_required<Vec3>(settings, "left_pinned_magnetisation"))
, right_pinned_magnetisation_(
    jams::config_required<Vec3>(settings, "right_pinned_magnetisation")) {

  // Number of unit cells from 0 along the 'a' direction to pin on the left
  int left_pinned_cells
    = jams::config_optional<int>(settings, "left_pinned_cells", 1);

  // Number of unit cells along the 'a' direction to pin from the right
  int right_pinned_cells
    = jams::config_optional<int>(settings, "right_pinned_cells", 1);

  // Find which spins are in the left and right pinned regions
  std::vector<int> left_region_indices;
  std::vector<int> right_region_indices;

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto cell = ::lattice->cell_offset(i);
    if (cell[0] < left_pinned_cells) {
      left_region_indices.push_back(i);
    }
    if (cell[0] >= lattice->size()[0] - right_pinned_cells) {
      right_region_indices.push_back(i);
    }
  }

  // Store the spin indices in the class
  left_pinned_indices_ = jams::MultiArray<int,1>(
      left_region_indices.begin(), left_region_indices.end());
  right_pinned_indices_ = jams::MultiArray<int,1>(
      right_region_indices.begin(), right_region_indices.end());

  initialized = true;
}

void PinnedBoundariesPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;

  // Rotate the spins in the left region so that their magnetisation points
  // along the pinning direction

  if (::solver->is_cuda_solver()) {
    Vec3 left_mag = jams::vector_field_indexed_scale_and_reduce_cuda(globals::s, globals::mus, left_pinned_indices_);

    auto left_rotation_matrix = rotation_matrix_between_vectors(
        left_mag, left_pinned_magnetisation_);

    jams::rotate_spins_cuda(globals::s, left_rotation_matrix,
                            left_pinned_indices_);

    Vec3 right_mag = jams::vector_field_indexed_scale_and_reduce_cuda(globals::s, globals::mus, right_pinned_indices_);

    auto right_rotation_matrix = rotation_matrix_between_vectors(
        right_mag, right_pinned_magnetisation_);

    jams::rotate_spins_cuda(globals::s, right_rotation_matrix,
                            right_pinned_indices_);

  } else {
    Vec3 left_mag = jams::sum_spins_moments(globals::s, globals::mus, left_pinned_indices_);

    auto left_rotation_matrix = rotation_matrix_between_vectors(
        left_mag, left_pinned_magnetisation_);

    jams::rotate_spins(globals::s, left_rotation_matrix, left_pinned_indices_);

    // Rotate the spins in the right region so that their magnetisation points
    // along the pinning direction

    Vec3 right_mag = jams::sum_spins_moments(globals::s, globals::mus, right_pinned_indices_);

    auto right_rotation_matrix = rotation_matrix_between_vectors(
        right_mag, right_pinned_magnetisation_);

    jams::rotate_spins(globals::s, right_rotation_matrix, right_pinned_indices_);
  }
}

// ----------------------------- END-OF-FILE ----------------------------------
