//
// Created by Joseph Barker on 2019-05-03.
//

#ifndef JAMS_SOLVERS_CPU_ROTATIONS_H
#define JAMS_SOLVERS_CPU_ROTATIONS_H

#include "jams/containers/multiarray.h"
#include "jams/core/solver.h"

#include <string>
#include <vector>

/// Solver for scanning static spin configurations over spherical rotation angles.
///
/// RotationSolver does not time-integrate the spin equations of motion. Instead,
/// each solver step restores the initial spin state and applies the current
/// spherical angle pair from a configured theta/phi grid. Monitors therefore see
/// one independent angle sample per solver iteration.
///
/// By default, the legacy `num_theta` and `num_phi` settings scan the full
/// sphere, with theta in [0, 180] degrees and phi in [0, 360] degrees. The
/// optional `theta` and `phi` settings can instead define a constant angle, a
/// finite range, or an explicit list of angle values. New angle settings are
/// specified in degrees and stored internally in radians.
///
/// Supported angle specifications:
/// - `theta = 90.0;`
/// - `theta = { value_deg = 90.0; };`
/// - `theta = { start_deg = 0.0; stop_deg = 180.0; count = 37; endpoint = true; };`
/// - `theta = { values_deg = [ 0.0, 45.0, 90.0 ]; };`
///
/// If `rotate_all_spins` is true, each sampled angle applies a common rotation
/// matrix to all spins. If false, the solver scans one basis spin target at a
/// time by assigning that spin to the sampled spherical direction.
///
/// The optional `rotation_targets` setting defines named material or unit-cell
/// position targets with independent theta/phi grids. The solver scans the
/// Cartesian product of all target grids. At each sample, all spins selected by
/// a target are assigned to that target's sampled spherical direction, so
/// multiple materials or positions in one target are co-rotated.
class RotationSolver : public Solver {
public:
    RotationSolver() = default;
    ~RotationSolver() override = default;

    inline explicit RotationSolver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;
    bool is_running() override;
    std::vector<jams::output::ColDef> monitor_coordinate_columns() const override;
    void append_monitor_coordinates(std::vector<double>& values) const override;

    std::string name() const override { return "rotations-cpu"; }


private:
    struct RotationTarget {
      std::string name;
      std::vector<int> spin_indices;
      std::vector<double> theta_values;
      std::vector<double> phi_values;
    };

    /// Lazily capture the initial spin state and apply the first rotation.
    void prepare_rotation_run();

    /// Restore initial spins, then apply the current angle sample.
    void apply_current_rotation();

    /// Read named material/position rotation targets from the solver settings.
    void read_rotation_targets(const libconfig::Setting& settings);

    /// Current basis spin target when `rotate_all_spins` is false.
    [[nodiscard]] unsigned current_spin_index() const;

    /// Current theta-grid index inside the active spin target.
    [[nodiscard]] unsigned current_theta_index() const;

    /// Current phi-grid index inside the active theta row.
    [[nodiscard]] unsigned current_phi_index() const;

    /// Current polar angle in radians.
    [[nodiscard]] double current_theta() const;

    /// Current azimuthal angle in radians.
    [[nodiscard]] double current_phi() const;

    /// Number of angle samples per spin target.
    [[nodiscard]] unsigned rotation_grid_size() const;

    /// True when the named multi-target rotation grid is active.
    [[nodiscard]] bool has_rotation_targets() const;

    /// Number of angle samples for one named target.
    [[nodiscard]] unsigned target_grid_size(std::size_t target_index) const;

    /// Current flat angle-grid index for one named target.
    [[nodiscard]] unsigned current_target_grid_index(std::size_t target_index) const;

    /// Current theta index for one named target.
    [[nodiscard]] unsigned current_target_theta_index(std::size_t target_index) const;

    /// Current phi index for one named target.
    [[nodiscard]] unsigned current_target_phi_index(std::size_t target_index) const;

    /// Current polar angle in radians for one named target.
    [[nodiscard]] double current_target_theta(std::size_t target_index) const;

    /// Current azimuthal angle in radians for one named target.
    [[nodiscard]] double current_target_phi(std::size_t target_index) const;

    bool     rotate_all_spins_ = true;

    /// Legacy full-sphere theta resolution, also used as the default range count.
    unsigned num_theta_ = 36;

    /// Legacy full-sphere phi resolution, also used as the default range count.
    unsigned num_phi_   = 72;

    /// Polar angle samples in radians.
    std::vector<double> theta_values_;

    /// Azimuthal angle samples in radians.
    std::vector<double> phi_values_;

    /// Named material/position targets for independent angle-grid scans.
    std::vector<RotationTarget> rotation_targets_;

    bool prepared_ = false;

    /// Spin state captured before the first rotation is applied.
    jams::MultiArray<double, 2> initial_spins_;
};

#endif //JAMS_SOLVERS_CPU_ROTATIONS_H
