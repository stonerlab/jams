//
// Created by Joseph Barker on 2019-05-03.
//

#ifndef JAMS_SOLVERS_CPU_ROTATIONS_H
#define JAMS_SOLVERS_CPU_ROTATIONS_H

#include "jams/containers/multiarray.h"
#include "jams/core/solver.h"

#include <vector>

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
    void prepare_rotation_run();
    void apply_current_rotation();

    [[nodiscard]] unsigned current_spin_index() const;
    [[nodiscard]] unsigned current_theta_index() const;
    [[nodiscard]] unsigned current_phi_index() const;
    [[nodiscard]] double current_theta() const;
    [[nodiscard]] double current_phi() const;
    [[nodiscard]] unsigned rotation_grid_size() const;

    bool     rotate_all_spins_ = true;
    unsigned num_theta_ = 36;
    unsigned num_phi_   = 72;
    std::vector<double> theta_values_;
    std::vector<double> phi_values_;
    bool prepared_ = false;
    jams::MultiArray<double, 2> initial_spins_;
};

#endif //JAMS_SOLVERS_CPU_ROTATIONS_H
