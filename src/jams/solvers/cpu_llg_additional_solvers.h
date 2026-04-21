#ifndef JAMS_SOLVERS_CPU_LLG_ADDITIONAL_SOLVERS_H
#define JAMS_SOLVERS_CPU_LLG_ADDITIONAL_SOLVERS_H

#include <cmath>
#include <random>
#include <string>

#include <libconfig.h++>
#include <pcg_random.hpp>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/interface/config.h"
#include "jams/solvers/llg_solver_utils.h"
#include "jams/solvers/llg_spin_torque_terms.h"
#include "jams/solvers/solver_descriptor.h"

class CPULLGSolverBase : public Solver {
 protected:
  void initialize_llg_solver(const libconfig::Setting& settings) {
    step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
    const auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
    const auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

    max_steps_ = static_cast<int>(t_max / step_size_);
    min_steps_ = static_cast<int>(t_min / step_size_);

    std::cout << "\ntimestep (ps) " << step_size_ << "\n";
    std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
    std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

    const auto descriptor = jams::solvers::describe_solver_setting(settings, *globals::config);
    const auto torque_field = jams::solvers::build_llg_spin_torque_field(settings, descriptor);
    extra_torque_ = torque_field.torque;
    thermal_prefactor_.resize(globals::num_spins);
    noise_.resize(globals::num_spins, 3);
    noise_.zero();

    const bool use_gilbert_prefactor = jams::config_optional<bool>(settings, "gilbert_prefactor", false);
    for (auto i = 0; i < globals::num_spins; ++i) {
      double denominator = 1.0;
      if (use_gilbert_prefactor) {
        denominator = 1.0 + pow2(globals::alpha(i));
      }
      thermal_prefactor_(i) = std::sqrt(
          (2.0 * kBoltzmannIU * globals::alpha(i)) / (globals::mus(i) * globals::gyro(i) * denominator));
    }
  }

  void generate_white_noise(const double dt) {
    if (physics_module_ == nullptr || physics_module_->temperature() <= 0.0) {
      noise_.zero();
      return;
    }

    const double scale = std::sqrt(physics_module_->temperature() / dt);
    std::normal_distribution<double> normal_distribution;

    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto n = 0; n < 3; ++n) {
        noise_(i, n) = normal_distribution(random_generator_) * thermal_prefactor_(i) * scale;
      }
    }
  }

  Vec3 spin(const int i) const {
    return {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
  }

  void set_spin(const int i, const Vec3& value) {
    for (auto n = 0; n < 3; ++n) {
      globals::s(i, n) = value[n];
    }
  }

  Vec3 torque(const int i) const {
    return {extra_torque_(i, 0), extra_torque_(i, 1), extra_torque_(i, 2)};
  }

  Vec3 effective_field(const int i) const {
    return {
        globals::h(i, 0) / globals::mus(i) + noise_(i, 0),
        globals::h(i, 1) / globals::mus(i) + noise_(i, 1),
        globals::h(i, 2) / globals::mus(i) + noise_(i, 2),
    };
  }

  Vec3 llg_rhs_at_spin(const int i, const Vec3& current_spin) const {
    return jams::solvers::llg_rhs(
        current_spin,
        effective_field(i),
        globals::gyro(i),
        globals::alpha(i),
        torque(i),
        globals::mus(i));
  }

  Vec3 llg_omega_at_spin(const int i, const Vec3& current_spin) const {
    return jams::solvers::llg_omega(
        current_spin,
        effective_field(i),
        globals::gyro(i),
        globals::alpha(i),
        torque(i),
        globals::mus(i));
  }

  void apply_noise_rodrigues(const double dt) {
    if (physics_module_ == nullptr || physics_module_->temperature() <= 0.0) {
      return;
    }

    generate_white_noise(dt);
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto current_spin = spin(i);
      const Vec3 thermal_field = {noise_(i, 0), noise_(i, 1), noise_(i, 2)};
      const auto omega = jams::solvers::llg_omega(
          current_spin, thermal_field, globals::gyro(i), globals::alpha(i));
      set_spin(i, jams::solvers::rodrigues_rotate(dt * omega, current_spin));
    }
  }

  void apply_noise_cayley(const double dt) {
    if (physics_module_ == nullptr || physics_module_->temperature() <= 0.0) {
      return;
    }

    generate_white_noise(dt);
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto current_spin = spin(i);
      const Vec3 thermal_field = {noise_(i, 0), noise_(i, 1), noise_(i, 2)};
      const auto omega = jams::solvers::llg_omega(
          current_spin, thermal_field, globals::gyro(i), globals::alpha(i));
      set_spin(i, jams::solvers::cayley_rotate(dt * omega, current_spin));
    }
  }

  void normalize_spins() {
    for (auto i = 0; i < globals::num_spins; ++i) {
      set_spin(i, jams::unit_vector(spin(i)));
    }
  }

  pcg32_k1024 random_generator_ = pcg_extras::seed_seq_from<pcg32>(jams::instance().random_generator()());
  jams::MultiArray<double, 2> extra_torque_;
  jams::MultiArray<double, 2> noise_;
  jams::MultiArray<double, 1> thermal_prefactor_;
};

class CPULLGRK4Solver : public CPULLGSolverBase {
 public:
  CPULLGRK4Solver() = default;
  inline explicit CPULLGRK4Solver(const libconfig::Setting& settings) {
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
      const Vec3 spin_new = {
          s_old_(i, 0) + step_size_ * (k1_(i, 0) + 2.0 * k2_(i, 0) + 2.0 * k3_(i, 0) + k4_(i, 0)) / 6.0,
          s_old_(i, 1) + step_size_ * (k1_(i, 1) + 2.0 * k2_(i, 1) + 2.0 * k3_(i, 1) + k4_(i, 1)) / 6.0,
          s_old_(i, 2) + step_size_ * (k1_(i, 2) + 2.0 * k2_(i, 2) + 2.0 * k3_(i, 2) + k4_(i, 2)) / 6.0,
      };
      set_spin(i, spin_new);
    }
    normalize_spins();

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "llg-rk4-cpu"; }

 private:
  void compute_stage(jams::MultiArray<double, 2>& stage) {
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto rhs = llg_rhs_at_spin(i, spin(i));
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

class CPULLGRKMK2Solver : public CPULLGSolverBase {
 public:
  CPULLGRKMK2Solver() = default;
  inline explicit CPULLGRKMK2Solver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override {
    initialize_llg_solver(settings);
    s_init_.resize(globals::num_spins, 3);
    phi_.resize(globals::num_spins, 3);
  }

  void run() override {
    const double t0 = time_;
    const double half_dt = 0.5 * step_size_;

    apply_noise_rodrigues(half_dt);
    s_init_ = globals::s;

    time_ = t0;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      const Vec3 phi = 0.5 * step_size_ * omega;
      for (auto n = 0; n < 3; ++n) {
        phi_(i, n) = phi[n];
      }
      set_spin(i, jams::solvers::rodrigues_rotate(phi, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    time_ = t0 + half_dt;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      const Vec3 v2 = step_size_ * omega;
      const Vec3 phi = {phi_(i, 0), phi_(i, 1), phi_(i, 2)};
      const auto k = jams::solvers::dexp_inv_so3(phi, v2);
      set_spin(i, jams::solvers::rodrigues_rotate(k, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    apply_noise_rodrigues(half_dt);

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "llg-rkmk2-cpu"; }

 private:
  jams::MultiArray<double, 2> s_init_;
  jams::MultiArray<double, 2> phi_;
};

class CPULLGRKMK4Solver : public CPULLGSolverBase {
 public:
  CPULLGRKMK4Solver() = default;
  inline explicit CPULLGRKMK4Solver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override {
    initialize_llg_solver(settings);
    s_init_.resize(globals::num_spins, 3);
    k1_.resize(globals::num_spins, 3);
    k2_.resize(globals::num_spins, 3);
    k3_.resize(globals::num_spins, 3);
  }

  void run() override {
    const double t0 = time_;
    const double half_dt = 0.5 * step_size_;

    apply_noise_rodrigues(half_dt);
    s_init_ = globals::s;

    time_ = t0;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      const Vec3 k1 = step_size_ * omega;
      for (auto n = 0; n < 3; ++n) {
        k1_(i, n) = k1[n];
      }
      set_spin(i, jams::solvers::rodrigues_rotate(0.5 * k1, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    time_ = t0 + half_dt;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      const Vec3 k2 = jams::solvers::dexp_inv_so3(
          {0.5 * k1_(i, 0), 0.5 * k1_(i, 1), 0.5 * k1_(i, 2)},
          step_size_ * omega);
      for (auto n = 0; n < 3; ++n) {
        k2_(i, n) = k2[n];
      }
      set_spin(i, jams::solvers::rodrigues_rotate(0.5 * k2, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    time_ = t0 + half_dt;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      const Vec3 k3 = jams::solvers::dexp_inv_so3(
          {0.5 * k2_(i, 0), 0.5 * k2_(i, 1), 0.5 * k2_(i, 2)},
          step_size_ * omega);
      for (auto n = 0; n < 3; ++n) {
        k3_(i, n) = k3[n];
      }
      set_spin(i, jams::solvers::rodrigues_rotate(k3, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    time_ = t0 + step_size_;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      const Vec3 k4 = jams::solvers::dexp_inv_so3(
          {k3_(i, 0), k3_(i, 1), k3_(i, 2)},
          step_size_ * omega);
      const Vec3 k = {
          (k1_(i, 0) + 2.0 * k2_(i, 0) + 2.0 * k3_(i, 0) + k4[0]) / 6.0,
          (k1_(i, 1) + 2.0 * k2_(i, 1) + 2.0 * k3_(i, 1) + k4[1]) / 6.0,
          (k1_(i, 2) + 2.0 * k2_(i, 2) + 2.0 * k3_(i, 2) + k4[2]) / 6.0,
      };
      set_spin(i, jams::solvers::rodrigues_rotate(k, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    apply_noise_rodrigues(half_dt);

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "llg-rkmk4-cpu"; }

 private:
  jams::MultiArray<double, 2> s_init_;
  jams::MultiArray<double, 2> k1_;
  jams::MultiArray<double, 2> k2_;
  jams::MultiArray<double, 2> k3_;
};

class CPULLGSemiImplicitSolver : public CPULLGSolverBase {
 public:
  CPULLGSemiImplicitSolver() = default;
  inline explicit CPULLGSemiImplicitSolver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override {
    initialize_llg_solver(settings);
    s_init_.resize(globals::num_spins, 3);
  }

  void run() override {
    const double t0 = time_;
    const double half_dt = 0.5 * step_size_;

    apply_noise_cayley(half_dt);
    s_init_ = globals::s;

    time_ = t0;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto current_spin = spin(i);
      auto omega = step_size_ * llg_omega_at_spin(i, current_spin);
      omega = jams::solvers::tangent_projection(omega, current_spin);
      const auto predictor = jams::solvers::cayley_rotate(omega, current_spin);
      set_spin(i, 0.5 * (predictor + current_spin));
    }

    time_ = t0 + half_dt;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto midpoint = spin(i);
      auto omega = step_size_ * llg_omega_at_spin(i, midpoint);
      omega = jams::solvers::tangent_projection(omega, midpoint);
      set_spin(i, jams::solvers::cayley_rotate(omega, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    apply_noise_cayley(half_dt);

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "llg-simp-cpu"; }

 private:
  jams::MultiArray<double, 2> s_init_;
};

class CPULLGDMSolver : public CPULLGSolverBase {
 public:
  CPULLGDMSolver() = default;
  inline explicit CPULLGDMSolver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override {
    initialize_llg_solver(settings);
    s_init_.resize(globals::num_spins, 3);
    s_pred_.resize(globals::num_spins, 3);
    omega1_.resize(globals::num_spins, 3);
  }

  void run() override {
    const double t0 = time_;
    const double half_dt = 0.5 * step_size_;

    apply_noise_rodrigues(half_dt);
    s_init_ = globals::s;

    time_ = t0;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega = llg_omega_at_spin(i, spin(i));
      for (auto n = 0; n < 3; ++n) {
        omega1_(i, n) = omega[n];
      }
      const auto predicted = jams::solvers::rodrigues_rotate(step_size_ * omega, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      });
      for (auto n = 0; n < 3; ++n) {
        s_pred_(i, n) = predicted[n];
        globals::s(i, n) = predicted[n];
      }
    }

    time_ = t0 + step_size_;
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto omega2 = llg_omega_at_spin(i, spin(i));
      const Vec3 omega_bar = {
          0.5 * (omega1_(i, 0) + omega2[0]),
          0.5 * (omega1_(i, 1) + omega2[1]),
          0.5 * (omega1_(i, 2) + omega2[2]),
      };
      set_spin(i, jams::solvers::rodrigues_rotate(step_size_ * omega_bar, {
        s_init_(i, 0), s_init_(i, 1), s_init_(i, 2),
      }));
    }

    apply_noise_rodrigues(half_dt);

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "llg-dm-cpu"; }

 private:
  jams::MultiArray<double, 2> s_init_;
  jams::MultiArray<double, 2> s_pred_;
  jams::MultiArray<double, 2> omega1_;
};

#endif  // JAMS_SOLVERS_CPU_LLG_ADDITIONAL_SOLVERS_H
