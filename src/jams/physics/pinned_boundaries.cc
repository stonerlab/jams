#include <jams/physics/pinned_boundaries.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>

PinnedBoundariesPhysics::PinnedBoundariesPhysics(const libconfig::Setting &settings)
: Physics(settings) {

  left_pinned_magnetisation_ = jams::config_required<Vec3>(settings, "left_pinned_magnetisation");
  right_pinned_magnetisation_ = jams::config_required<Vec3>(settings, "right_pinned_magnetisation");

  // number of unit cells from 0 along the 'a' direction to pin on the left
  int left_pinned_cells = jams::config_optional<int>(settings, "left_pinned_cells", 1);

  // number of unit cells along the 'a' direction to pin from the right
  int right_pinned_cells = jams::config_optional<int>(settings, "right_pinned_cells", 1);

  std::vector<int> left_region;
  std::vector<int> right_region;

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto cell = ::lattice->cell_offset(i);
    if (cell[0] < left_pinned_cells) {
      left_region.push_back(i);
    }
    if (cell[0] >= lattice->size()[0] - right_pinned_cells) {
      right_region.push_back(i);
    }
  }

  left_pinned_spins_ = jams::MultiArray<int,1>(left_region.begin(), left_region.end());
  right_pinned_spins_ = jams::MultiArray<int,1>(right_region.begin(), right_region.end());

  initialized = true;
}

void PinnedBoundariesPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;

  Vec3 left_mag = {0.0, 0.0, 0.0};
  for (auto n = 0; n < left_pinned_spins_.elements(); ++n) {
    const auto i = left_pinned_spins_(n);
    for (auto j = 0; j < 3; ++j) {
      left_mag[j] += globals::s(i, j);
    }
  }

  auto left_rotation_matrix = rotation_matrix_between_vectors(left_mag, left_pinned_magnetisation_);

  for (auto n = 0; n < left_pinned_spins_.elements(); ++n) {
    const auto i = left_pinned_spins_(n);
    Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    spin = left_rotation_matrix * spin;
    for (auto j = 0; j < 3; ++j) {
      s(i, j) = spin[j];
    }
  }

  Vec3 right_mag = {0.0, 0.0, 0.0};
  for (auto n = 0; n < right_pinned_spins_.elements(); ++n) {
    const auto i = right_pinned_spins_(n);
    for (auto j = 0; j < 3; ++j) {
      right_mag[j] += globals::s(i, j);
    }
  }

  auto right_rotation_matrix = rotation_matrix_between_vectors(right_mag, right_pinned_magnetisation_);

  for (auto n = 0; n < right_pinned_spins_.elements(); ++n) {
    const auto i = right_pinned_spins_(n);
    Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    spin = right_rotation_matrix * spin;
    for (auto j = 0; j < 3; ++j) {
      s(i, j) = spin[j];
    }
  }
}