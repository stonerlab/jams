#ifndef JAMS_SOLVERS_CPU_LL_LORENTZIAN_RK4_H
#define JAMS_SOLVERS_CPU_LL_LORENTZIAN_RK4_H

#include "jams/solvers/cpu_llg_additional_solvers.h"

class CPULLLorentzianRK4Solver : public CPULLGSolverBase {
 public:
  CPULLLorentzianRK4Solver() = default;
  inline explicit CPULLLorentzianRK4Solver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override {
    if (globals::lattice->num_materials() > 1) {
      throw std::runtime_error("CPULLLorentzianRK4Solver is only implemented for single material cells");
    }

    initialize_llg_solver(settings);

    lorentzian_gamma_ = kTwoPi * jams::config_required<double>(globals::config->lookup("thermostat"), "lorentzian_gamma");
    lorentzian_omega_ = kTwoPi * jams::config_required<double>(globals::config->lookup("thermostat"), "lorentzian_omega0");

    const double eta_g = globals::alpha(0) / (globals::mus(0) * globals::gyro(0));
    lorentzian_a_ = eta_g * pow4(lorentzian_omega_) / lorentzian_gamma_;

    s_old_.resize(globals::num_spins, 3);
    s_k1_.resize(globals::num_spins, 3);
    s_k2_.resize(globals::num_spins, 3);
    s_k3_.resize(globals::num_spins, 3);
    s_k4_.resize(globals::num_spins, 3);

    w_memory_.resize(globals::num_spins, 3);
    w_memory_old_.resize(globals::num_spins, 3);
    w_k1_.resize(globals::num_spins, 3);
    w_k2_.resize(globals::num_spins, 3);
    w_k3_.resize(globals::num_spins, 3);
    w_k4_.resize(globals::num_spins, 3);
    zero(w_memory_);
    zero(w_memory_old_);

    v_memory_.resize(globals::num_spins, 3);
    v_memory_old_.resize(globals::num_spins, 3);
    v_k1_.resize(globals::num_spins, 3);
    v_k2_.resize(globals::num_spins, 3);
    v_k3_.resize(globals::num_spins, 3);
    v_k4_.resize(globals::num_spins, 3);
    zero(v_memory_);
    zero(v_memory_old_);
  }

  void run() override {
    const double t0 = time_;

    s_old_ = globals::s;
    w_memory_old_ = w_memory_;
    v_memory_old_ = v_memory_;
    update_thermostat();

    time_ = t0;
    compute_stage(s_k1_, w_k1_, v_k1_);

    globals::s = s_old_;
    w_memory_ = w_memory_old_;
    v_memory_ = v_memory_old_;
    add_stage(0.5 * step_size_, s_k1_, w_k1_, v_k1_);
    time_ = t0 + 0.5 * step_size_;
    compute_stage(s_k2_, w_k2_, v_k2_);

    globals::s = s_old_;
    w_memory_ = w_memory_old_;
    v_memory_ = v_memory_old_;
    add_stage(0.5 * step_size_, s_k2_, w_k2_, v_k2_);
    time_ = t0 + 0.5 * step_size_;
    compute_stage(s_k3_, w_k3_, v_k3_);

    globals::s = s_old_;
    w_memory_ = w_memory_old_;
    v_memory_ = v_memory_old_;
    add_stage(step_size_, s_k3_, w_k3_, v_k3_);
    time_ = t0 + step_size_;
    compute_stage(s_k4_, w_k4_, v_k4_);

    globals::s = s_old_;
    w_memory_ = w_memory_old_;
    v_memory_ = v_memory_old_;
    combine_state(step_size_, s_k1_, s_k2_, s_k3_, s_k4_, globals::s, true);
    combine_state(step_size_, w_k1_, w_k2_, w_k3_, w_k4_, w_memory_, false);
    combine_state(step_size_, v_k1_, v_k2_, v_k3_, v_k4_, v_memory_, false);

    iteration_++;
    time_ = iteration_ * step_size_;
  }

  std::string name() const override { return "ll-lorentzian-rk4-cpu"; }

 private:
  void compute_stage(jams::MultiArray<double, 2>& s_stage,
                     jams::MultiArray<double, 2>& w_stage,
                     jams::MultiArray<double, 2>& v_stage) {
    compute_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto current_spin = spin(i);
      const Vec3 v = {v_memory_(i, 0), v_memory_(i, 1), v_memory_(i, 2)};
      const Vec3 w = {w_memory_(i, 0), w_memory_(i, 1), w_memory_(i, 2)};
      const auto thermal = thermal_field(i);
      const Vec3 field = {
          globals::h(i, 0) / globals::mus(i) + thermal[0] + v[0],
          globals::h(i, 1) / globals::mus(i) + thermal[1] + v[1],
          globals::h(i, 2) / globals::mus(i) + thermal[2] + v[2],
      };

      const auto s_rhs = -globals::gyro(i) * jams::cross(current_spin, field);
      const Vec3 w_rhs = -pow2(lorentzian_omega_) * v - lorentzian_gamma_ * w
          + lorentzian_a_ * globals::mus(i) * current_spin;

      for (auto n = 0; n < 3; ++n) {
        s_stage(i, n) = s_rhs[n];
        w_stage(i, n) = w_rhs[n];
        v_stage(i, n) = w[n];
      }
    }
  }

  void add_stage(const double scale,
                 const jams::MultiArray<double, 2>& s_stage,
                 const jams::MultiArray<double, 2>& w_stage,
                 const jams::MultiArray<double, 2>& v_stage) {
    for (auto i = 0; i < globals::num_spins; ++i) {
      set_spin(i, {
          s_old_(i, 0) + scale * s_stage(i, 0),
          s_old_(i, 1) + scale * s_stage(i, 1),
          s_old_(i, 2) + scale * s_stage(i, 2),
      });
      for (auto n = 0; n < 3; ++n) {
        w_memory_(i, n) = w_memory_old_(i, n) + scale * w_stage(i, n);
        v_memory_(i, n) = v_memory_old_(i, n) + scale * v_stage(i, n);
      }
    }
  }

  void combine_state(const double dt,
                     const jams::MultiArray<double, 2>& k1,
                     const jams::MultiArray<double, 2>& k2,
                     const jams::MultiArray<double, 2>& k3,
                     const jams::MultiArray<double, 2>& k4,
                     jams::MultiArray<double, 2>& state,
                     const bool normalize) {
    for (auto i = 0; i < globals::num_spins; ++i) {
      Vec3 combined = {
          state(i, 0) + dt * (k1(i, 0) + 2.0 * k2(i, 0) + 2.0 * k3(i, 0) + k4(i, 0)) / 6.0,
          state(i, 1) + dt * (k1(i, 1) + 2.0 * k2(i, 1) + 2.0 * k3(i, 1) + k4(i, 1)) / 6.0,
          state(i, 2) + dt * (k1(i, 2) + 2.0 * k2(i, 2) + 2.0 * k3(i, 2) + k4(i, 2)) / 6.0,
      };
      if (normalize) {
        combined = jams::unit_vector(combined);
      }
      for (auto n = 0; n < 3; ++n) {
        state(i, n) = combined[n];
      }
    }
  }

  double lorentzian_omega_ = 0.0;
  double lorentzian_gamma_ = 0.0;
  double lorentzian_a_ = 0.0;

  jams::MultiArray<double, 2> s_old_;
  jams::MultiArray<double, 2> s_k1_;
  jams::MultiArray<double, 2> s_k2_;
  jams::MultiArray<double, 2> s_k3_;
  jams::MultiArray<double, 2> s_k4_;

  jams::MultiArray<double, 2> w_memory_;
  jams::MultiArray<double, 2> w_memory_old_;
  jams::MultiArray<double, 2> w_k1_;
  jams::MultiArray<double, 2> w_k2_;
  jams::MultiArray<double, 2> w_k3_;
  jams::MultiArray<double, 2> w_k4_;

  jams::MultiArray<double, 2> v_memory_;
  jams::MultiArray<double, 2> v_memory_old_;
  jams::MultiArray<double, 2> v_k1_;
  jams::MultiArray<double, 2> v_k2_;
  jams::MultiArray<double, 2> v_k3_;
  jams::MultiArray<double, 2> v_k4_;
};

#endif  // JAMS_SOLVERS_CPU_LL_LORENTZIAN_RK4_H
