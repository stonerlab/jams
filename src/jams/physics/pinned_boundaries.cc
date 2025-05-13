// pinned_boundaries.cc                                                -*-C++-*-
#include <jams/physics/pinned_boundaries.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/spinops.h>
#include <jams/cuda/cuda_spin_ops.h>
#include <jams/cuda/cuda_array_reduction.h>


PinnedBoundariesPhysics::PinnedBoundariesPhysics(const libconfig::Setting &settings)
: Physics(settings) {
  for (const auto& [name, comp] : boundary_definitions_) {
    if (!settings.exists(name + "_pinned_magnetisation")) continue;

    PinnedBoundary boundary;
    boundary.magnetisation = jams::config_required<Vec3>(settings, name + "_pinned_magnetisation");

    int num_pinned_cells = jams::config_optional<int>(settings, name + "_pinned_cells", 1);

    std::vector<int> region_indices;
    for (auto i = 0; i < globals::num_spins; ++i) {
      if (comp(num_pinned_cells, globals::lattice->cell_offset(i), globals::lattice->size())) {
        region_indices.push_back(i);
      }
    }
    boundary.indices = jams::MultiArray<int,1>(region_indices.begin(), region_indices.end());

    boundaries.insert({name, boundary});
  }
}

void PinnedBoundariesPhysics::update(const int &iterations, const double &time, const double &dt) {
    for (const auto& [name, boundary] : boundaries) {
      if (globals::solver->is_cuda_solver()) {
        Vec3 mag = jams::vector_field_indexed_scale_and_reduce_cuda(globals::s, globals::mus, boundary.indices);
        auto rotation_matrix = rotation_matrix_between_vectors(mag, boundary.magnetisation);
        jams::rotate_spins_cuda(globals::s, rotation_matrix, boundary.indices);
      } else {
        Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus, boundary.indices);
        auto rotation_matrix = rotation_matrix_between_vectors(mag, boundary.magnetisation);
        jams::rotate_spins(globals::s, rotation_matrix, boundary.indices);
    }
  }
}

// ----------------------------- END-OF-FILE ----------------------------------
