// pinned_boundaries.cc                                                -*-C++-*-
#include <jams/physics/pinned_boundaries.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>

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

  // Calculate the magnetisation in the left region
  Vec3 left_mag = {0.0, 0.0, 0.0};
  for (auto n = 0; n < left_pinned_indices_.elements(); ++n) {
    const auto i = left_pinned_indices_(n);
    for (auto j = 0; j < 3; ++j) {
      left_mag[j] += globals::s(i, j);
    }
  }

  // Rotate the spins in the left region so that their magnetisation points
  // along the pinning direction
  auto left_rotation_matrix = rotation_matrix_between_vectors(
      left_mag, left_pinned_magnetisation_);

  for (auto n = 0; n < left_pinned_indices_.elements(); ++n) {
    auto i = left_pinned_indices_(n);
    Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    spin = left_rotation_matrix * spin;
    for (auto j = 0; j < 3; ++j) {
      s(i, j) = spin[j];
    }
  }

  // Calculate the magnetisation in the right region
  Vec3 right_mag = {0.0, 0.0, 0.0};
  for (auto n = 0; n < right_pinned_indices_.elements(); ++n) {
    auto i = right_pinned_indices_(n);
    for (auto j = 0; j < 3; ++j) {
      right_mag[j] += globals::s(i, j);
    }
  }

  // Rotate the spins in the left region so that their magnetisation points
  // along the pinning direction
  auto right_rotation_matrix = rotation_matrix_between_vectors(
      right_mag, right_pinned_magnetisation_);

  for (auto n = 0; n < right_pinned_indices_.elements(); ++n) {
    auto i = right_pinned_indices_(n);
    Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    spin = right_rotation_matrix * spin;
    for (auto j = 0; j < 3; ++j) {
      s(i, j) = spin[j];
    }
  }
}