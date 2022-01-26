#ifndef JAMS_PINNED_BOUNDARIES_PHYSICS_H
#define JAMS_PINNED_BOUNDARIES_PHYSICS_H

#include <jams/core/physics.h>
#include <jams/interface/config.h>
#include <jams/containers/multiarray.h>

/// @class PinnedBoundaryPhysics
///
/// Maintains the direction of magnetisation in two edge regions by rotating
/// spins in that region.
///
/// @details
/// @p This class is designed for pinning the direction of the magnetisation at the
/// edges of the simulation cell. The primary reason to do this is to simulate
/// domain walls by pinning the magnetisation of each side of a finite system
/// in opposite directions.
///
/// @p A 'left' and 'right' region are defined as a number of unit cells starting
/// from the left and right edges of the system along the 'a' lattice vector.
/// The update() method calculates the magnetisation direction in those regions
/// and rotates all of the spins so that the magnetisation aligns with preset
/// pinning directions. The pinning directions can be different on the left
/// and right. The length of the magnetisation is not constrained, so the pinned
/// regions can still have a magnetisation length which is the equilibrium
/// value.
///
/// @setting `physics.left_pinned_magnetisation` (required) magnetisation pinned
///           direction on the left side
/// @setting `physics.right_pinned_magnetisation` (required) magnetisation pinned
///           direction on the right side
/// @setting `physics.left_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the left edge
/// @setting `physics.right_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the right edge
///
/// @example Example config:
/// @code{.unparsed}
/// physics: {
///     module = "pinned_boundaries";
///     left_pinned_magnetisation = [0.0, 0.0,-1.0];
///     right_pinned_magnetisation = [0.0, 0.0, 1.0];
///     left_pinned_cells = 5;
///     right_pinned_cells = 5;
///     temperature = 100.0;
/// };
/// @endcode


class PinnedBoundariesPhysics : public Physics {
public:
    PinnedBoundariesPhysics(const libconfig::Setting &settings);
    ~PinnedBoundariesPhysics() = default;
    void update(const int &iterations, const double &time, const double &dt);
private:
    bool initialized;

    Vec3 left_pinned_magnetisation_ = {0.0, 0.0, -1.0};
    Vec3 right_pinned_magnetisation_ = {0.0, 0.0, 1.0};

    jams::MultiArray<int,1> left_pinned_indices_;
    jams::MultiArray<int,1> right_pinned_indices_;
};

#endif  // JAMS_PINNED_BOUNDARIES_PHYSICS_H
