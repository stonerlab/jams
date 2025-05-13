#ifndef JAMS_PINNED_BOUNDARIES_PHYSICS_H
#define JAMS_PINNED_BOUNDARIES_PHYSICS_H

#include <functional>
#include <unordered_map>

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
/// from the left and right edges of the system along the 'a' lattice vector,
/// 'front' and 'back' are along the 'b' lattice vector and 'bottom' and 'top'
/// are along the 'c' lattice vector.
/// The update() method calculates the magnetisation direction in those regions
/// and rotates all of the spins so that the magnetisation aligns with preset
/// pinning directions. The pinning directions can be different on the left
/// and right. The length of the magnetisation is not constrained, so the pinned
/// regions can still have a magnetisation length which is the equilibrium
/// value.
///
/// @setting `physics.left_pinned_magnetisation` (optional) magnetisation pinned
///           direction on the left side
/// @setting `physics.right_pinned_magnetisation` (optional) magnetisation pinned
///           direction on the right side
/// @setting `physics.left_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the left edge
/// @setting `physics.right_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the right edge
/// @setting `physics.front_pinned_magnetisation` (optional) magnetisation pinned
///           direction on the front side
/// @setting `physics.back_pinned_magnetisation` (optional) magnetisation pinned
///           direction on the back side
/// @setting `physics.front_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the front edge
/// @setting `physics.back_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the back edge
/// @setting `physics.bottom_pinned_magnetisation` (optional) magnetisation pinned
///           direction on the bottom side
/// @setting `physics.top_pinned_magnetisation` (optional) magnetisation pinned
///           direction on the top side
/// @setting `physics.bottom_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the bottom edge
/// @setting `physics.top_pinned_cells` (optional, default `1`) number of unit
///           cells to pin from the top edge
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

#include <jams/core/physics.h>
#include <jams/interface/config.h>
#include <jams/containers/multiarray.h>

class PinnedBoundariesPhysics : public Physics {
public:
    explicit PinnedBoundariesPhysics(const libconfig::Setting &settings);
    ~PinnedBoundariesPhysics() override = default;

    void update(const int &iterations, const double &time, const double &dt) override;

private:
    struct PinnedBoundary {
        Vec3 magnetisation;
        jams::MultiArray<int, 1> indices;
    };

    // Create the definitions of each boundary. This is based on the dimension ({0,1,2} -> {a, b, c}) and
    // whether this is the lower boundary (increasing from 0) or the upper boundary (decreasing from
    // supercell size). The boundary is stored as a comparator which can be called like
    // `comp(num_pinned_cells, globals::lattice->cell_offset(i), globals::lattice->size())` where true
    // indicates spin i is included in the boundary boundary.

    using BoundaryPredicate = std::function<bool(int, const Vec3i&, const Vec3i&)>;
    static inline std::unordered_map<std::string, BoundaryPredicate> make_boundary_definitions() {
        auto comp = [](int dim, bool upper) {
            return [dim, upper](int n, const Vec3i& cell, const Vec3i& size) {
                return upper ? cell[dim] >= size[dim] - n : cell[dim] < n;
            };
        };

        return {
            {"left",   comp(0, false)},
            {"right",  comp(0, true)},
            {"front",  comp(1, false)},
            {"back",   comp(1, true)},
            {"bottom", comp(2, false)},
            {"top",    comp(2, true)},
        };
    }

    static inline const auto boundary_definitions_ = make_boundary_definitions();
    std::unordered_map<std::string, PinnedBoundary> boundaries;
};

#endif  // JAMS_PINNED_BOUNDARIES_PHYSICS_H

// ----------------------------- END-OF-FILE ----------------------------------
