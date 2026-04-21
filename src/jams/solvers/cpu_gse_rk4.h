#ifndef JAMS_SOLVERS_CPU_GSE_RK4_H
#define JAMS_SOLVERS_CPU_GSE_RK4_H

#include "jams/solvers/cpu_llg_additional_solvers.h"

class CPUGSERK4Solver : public CPULLGSolverBase {
 public:
  CPUGSERK4Solver() = default;
  inline explicit CPUGSERK4Solver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override {
    initialize_llg_solver(settings);
    s_old_.resize(globals::num_spins, 3);
    k1_.resize(globals::num_spins, 3);
    k2_.resize(globals::num_spins, 3);
    k3_.resize(globals::num_spins, 3);
    k4_.resize(globals::num_spins, 3);
  }

  void run() override {
    const double t0 = time_;
    s_old_ = globals::s;
    generate_white_noise(step_size_);

    time_ = t0;
    compute_stage(k1_);

    globals::s = s_old_;
    add_stage_spins(k1_, 0.5 * step_size_);
    time_ = t0 + 0.5 * step_size_;
    compute_stage(k2_);

    globals::s = s_old_;
    add_stage_spins(k2_, 0.5 * step_size_);
    time_ = t0 + 0.5 * step_size_;
    compute_stage(k3_);

    globals::s = s_old_;
    add_stage_spins(k3_, step_size_);
    time_ = t0 + step_size_;
    compute_stage(k4_);

    globals::s = s_old_;
    for (auto i = 0; i < globals::num_spins; ++i) {
      set_spin(i, {
          s_old_(i, 0) + step_size_ * (k1_(i, 0) + 2.0 * k2_(i, 0) + 2.0 * k3_(i, 0) + k4_(i, 0)) / 6.0,
          s_old_(i, 1) + step_size_ * (k1_(i, 1) + 2.0 * k2_(i, 1) + 2.0 * k3_(i, 1) + k4_(i, 1)) / 6.0,
          s_old_(i, 2) + step_size_ * (k1_(i, 2) + 2.0 * k2_(i, 2) + 2.0 * k3_(i, 2) + k4_(i, 2)) / 6.0,
      });
    }
    normalize_spins();

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "gse-rk4-cpu"; }

 private:
  void compute_stage(jams::MultiArray<double, 2>& stage) {
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto current_spin = spin(i);
      const Vec3 deterministic_field = {
          globals::h(i, 0) / globals::mus(i),
          globals::h(i, 1) / globals::mus(i),
          globals::h(i, 2) / globals::mus(i),
      };
      const Vec3 noise = {noise_(i, 0), noise_(i, 1), noise_(i, 2)};
      const auto spin_cross_field = jams::cross(current_spin, deterministic_field);
      const auto rhs = -globals::gyro(i) * (spin_cross_field - globals::alpha(i) * deterministic_field)
          + globals::gyro(i) * noise;
      for (auto n = 0; n < 3; ++n) {
        stage(i, n) = rhs[n];
      }
    }
  }

  void add_stage_spins(const jams::MultiArray<double, 2>& stage, const double scale) {
    for (auto i = 0; i < globals::num_spins; ++i) {
      set_spin(i, {
          s_old_(i, 0) + scale * stage(i, 0),
          s_old_(i, 1) + scale * stage(i, 1),
          s_old_(i, 2) + scale * stage(i, 2),
      });
    }
  }

  jams::MultiArray<double, 2> s_old_;
  jams::MultiArray<double, 2> k1_;
  jams::MultiArray<double, 2> k2_;
  jams::MultiArray<double, 2> k3_;
  jams::MultiArray<double, 2> k4_;
};

#endif  // JAMS_SOLVERS_CPU_GSE_RK4_H
