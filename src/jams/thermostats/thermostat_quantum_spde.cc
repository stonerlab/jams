// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/thermostat_quantum_spde.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/utils.h"

#if HAS_OMP
#include <omp.h>
#endif

namespace {

int omp_thread_count() {
#if HAS_OMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // namespace

namespace jams {

QuantumSpdeNoiseGenerator::QuantumSpdeNoiseGenerator(
    const int process_count, const double delta_tau, const double omega_max,
    const bool zero_point)
    : process_count_(process_count),
      delta_tau_(delta_tau),
      omega_max_(omega_max),
      zero_point_(zero_point),
      random_generator_(pcg_extras::seed_seq_from<pcg32>(
          jams::instance().random_generator()())) {
  ensure_random_generators(1);

  zeta5_.resize(process_count_).zero();
  zeta5p_.resize(process_count_).zero();
  zeta6_.resize(process_count_).zero();
  zeta6p_.resize(process_count_).zero();
  eta1_.resize(2 * process_count_).zero();
  eta_stationary_.resize(4 * process_count_).zero();

  if (zero_point_) {
    zero_point_coefficients_ = quantum_spde_zero_point_update_coefficients(delta_tau_, omega_max_);
    zeta0_.resize(4 * process_count_).zero();
    eta0_.resize(4 * process_count_).zero();
  }
}

void QuantumSpdeNoiseGenerator::ensure_random_generators(const int count) {
  if (random_generators_.empty() && count > 0) {
    random_generators_.push_back(random_generator_);
  }
  while (static_cast<int>(random_generators_.size()) < count) {
    random_generators_.emplace_back(
        pcg_extras::seed_seq_from<pcg32>(random_generator_()));
  }
}

void QuantumSpdeNoiseGenerator::fill_standard_normal(
    jams::MultiArray<jams::Real, 1>& values) {
  ensure_random_generators(omp_thread_count());

  auto values_view = values.mutable_host_view();
  auto* data = values_view.data();
  const auto count = static_cast<int>(values_view.size());
#if HAS_OMP
#pragma omp parallel
  {
    const auto thread_id = omp_get_thread_num();
    auto& generator = random_generators_[thread_id];
    std::normal_distribution<> distribution;
#pragma omp for schedule(static)
    for (auto i = 0; i < count; ++i) {
      data[i] = static_cast<jams::Real>(distribution(generator));
    }
  }
#else
  std::normal_distribution<> distribution;
  for (auto i = 0; i < count; ++i) {
    data[i] = static_cast<jams::Real>(distribution(random_generator_));
  }
#endif
}

void QuantumSpdeNoiseGenerator::prepare_fixed_temperature_coefficients(
    const jams::Real temperature) {
  const double reduced_delta_tau = delta_tau_ * static_cast<double>(temperature);
  if (reduced_delta_tau <= 0.0) {
    fast_coefficients_valid_ = false;
    fast_temperature_ = temperature;
    return;
  }

  fast_factor5_ = quantum_spde_bose_update_coefficients(
      kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
  fast_factor6_ = quantum_spde_bose_update_coefficients(
      kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
  fast_temperature_ = temperature;
  fast_coefficients_valid_ = true;
}

void QuantumSpdeNoiseGenerator::prepare_temperature_profile_coefficients(
    const jams::MultiArray<jams::Real, 1>& temperature) {
  if (static_cast<int>(temperature.size()) != process_count_) {
    throw std::runtime_error("quantum SPDE temperature profile size does not match process count");
  }

  profile_factor5_.resize(process_count_);
  profile_factor6_.resize(process_count_);
  profile_stationary_factor5_.resize(process_count_);
  profile_stationary_factor6_.resize(process_count_);

  const auto temperature_view = temperature.host_view();
  auto factor5 = profile_factor5_.mutable_host_view();
  auto factor6 = profile_factor6_.mutable_host_view();
  auto stationary5 = profile_stationary_factor5_.mutable_host_view();
  auto stationary6 = profile_stationary_factor6_.mutable_host_view();
  const auto* temperature_values = temperature_view.data();
  auto* factor5_values = factor5.data();
  auto* factor6_values = factor6.data();
  auto* stationary5_values = stationary5.data();
  auto* stationary6_values = stationary6.data();

  int has_positive_temperature = 0;
#if HAS_OMP
#pragma omp parallel for schedule(static) reduction(max : has_positive_temperature)
#endif
  for (auto x = 0; x < process_count_; ++x) {
    const double reduced_delta_tau =
        delta_tau_ * static_cast<double>(temperature_values[x]);
    if (reduced_delta_tau <= 0.0) {
      factor5_values[x] = {};
      factor6_values[x] = {};
      stationary5_values[x] = {};
      stationary6_values[x] = {};
      continue;
    }

    has_positive_temperature = 1;
    factor5_values[x] = quantum_spde_bose_update_coefficients(
        kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
    factor6_values[x] = quantum_spde_bose_update_coefficients(
        kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
    stationary5_values[x] = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
    stationary6_values[x] = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
  }
  profile_has_positive_temperature_ = has_positive_temperature != 0;
}

void QuantumSpdeNoiseGenerator::zero_noise(jams::Real* noise) const {
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (auto x = 0; x < process_count_; ++x) {
    noise[x] = jams::Real{0.0};
  }
}

void QuantumSpdeNoiseGenerator::zero_state() {
  zeta5_.zero();
  zeta5p_.zero();
  zeta6_.zero();
  zeta6p_.zero();
  if (zero_point_) {
    zeta0_.zero();
  }
}

void QuantumSpdeNoiseGenerator::initialize(const Initialization initialization,
                                           const jams::Real temperature) {
  if (initialization == Initialization::Stationary) {
    initialize_stationary(temperature);
    return;
  }

  zero_state();
  prepare_fixed_temperature_coefficients(temperature);
}

void QuantumSpdeNoiseGenerator::initialize(
    const Initialization initialization,
    const jams::MultiArray<jams::Real, 1>& temperature) {
  prepare_temperature_profile_coefficients(temperature);
  if (initialization == Initialization::Stationary) {
    initialize_stationary(temperature);
    return;
  }

  zero_state();
}

void QuantumSpdeNoiseGenerator::initialize_stationary(const jams::Real temperature) {
  zero_state();

  const double reduced_delta_tau = delta_tau_ * static_cast<double>(temperature);
  if (reduced_delta_tau > 0.0) {
    fill_standard_normal(eta_stationary_);

    const auto factor5 = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
    const auto factor6 = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);

    const auto eta_view = eta_stationary_.host_view();
    auto zeta5 = zeta5_.mutable_host_view();
    auto zeta5p = zeta5p_.mutable_host_view();
    auto zeta6 = zeta6_.mutable_host_view();
    auto zeta6p = zeta6p_.mutable_host_view();
    const auto* eta = eta_view.data();
    auto* z5 = zeta5.data();
    auto* z5p = zeta5p.data();
    auto* z6 = zeta6.data();
    auto* z6p = zeta6p.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto x = 0; x < process_count_; ++x) {
      const double g5_0 = static_cast<double>(eta[x]);
      const double g5_1 = static_cast<double>(eta[process_count_ + x]);
      z5[x] = factor5.l00 * g5_0;
      z5p[x] = factor5.l10 * g5_0 + factor5.l11 * g5_1;

      const double g6_0 = static_cast<double>(eta[2 * process_count_ + x]);
      const double g6_1 = static_cast<double>(eta[3 * process_count_ + x]);
      z6[x] = factor6.l00 * g6_0;
      z6p[x] = factor6.l10 * g6_0 + factor6.l11 * g6_1;
    }
  }

  if (zero_point_) {
    fill_standard_normal(eta0_);

    const double h_omega_max = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
    const auto eta_view = eta0_.host_view();
    auto zeta0 = zeta0_.mutable_host_view();
    const auto* eta = eta_view.data();
    auto* z0 = zeta0.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto x = 0; x < process_count_; ++x) {
      for (auto i = 0; i < 4; ++i) {
        const auto index = 4 * x + i;
        const double variance =
            quantum_spde_zero_point_stationary_variance(h_omega_max, i);
        z0[index] = static_cast<double>(eta[index]) * std::sqrt(variance);
      }
    }
  }

  prepare_fixed_temperature_coefficients(temperature);
}

void QuantumSpdeNoiseGenerator::initialize_stationary(
    const jams::MultiArray<jams::Real, 1>& temperature) {
  prepare_temperature_profile_coefficients(temperature);
  zero_state();

  if (profile_has_positive_temperature_) {
    fill_standard_normal(eta_stationary_);

    const auto eta_view = eta_stationary_.host_view();
    const auto factor5 = profile_stationary_factor5_.host_view();
    const auto factor6 = profile_stationary_factor6_.host_view();
    auto zeta5 = zeta5_.mutable_host_view();
    auto zeta5p = zeta5p_.mutable_host_view();
    auto zeta6 = zeta6_.mutable_host_view();
    auto zeta6p = zeta6p_.mutable_host_view();
    const auto* eta = eta_view.data();
    const auto* f5 = factor5.data();
    const auto* f6 = factor6.data();
    auto* z5 = zeta5.data();
    auto* z5p = zeta5p.data();
    auto* z6 = zeta6.data();
    auto* z6p = zeta6p.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto x = 0; x < process_count_; ++x) {
      if (f5[x].l00 == 0.0 && f6[x].l00 == 0.0) {
        z5[x] = 0.0;
        z5p[x] = 0.0;
        z6[x] = 0.0;
        z6p[x] = 0.0;
        continue;
      }

      const double g5_0 = static_cast<double>(eta[x]);
      const double g5_1 = static_cast<double>(eta[process_count_ + x]);
      z5[x] = f5[x].l00 * g5_0;
      z5p[x] = f5[x].l10 * g5_0 + f5[x].l11 * g5_1;

      const double g6_0 = static_cast<double>(eta[2 * process_count_ + x]);
      const double g6_1 = static_cast<double>(eta[3 * process_count_ + x]);
      z6[x] = f6[x].l00 * g6_0;
      z6p[x] = f6[x].l10 * g6_0 + f6[x].l11 * g6_1;
    }
  }

  if (zero_point_) {
    fill_standard_normal(eta0_);

    const double h_omega_max = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
    const auto eta_view = eta0_.host_view();
    auto zeta0 = zeta0_.mutable_host_view();
    const auto* eta = eta_view.data();
    auto* z0 = zeta0.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto x = 0; x < process_count_; ++x) {
      for (auto i = 0; i < 4; ++i) {
        const auto index = 4 * x + i;
        const double variance =
            quantum_spde_zero_point_stationary_variance(h_omega_max, i);
        z0[index] = static_cast<double>(eta[index]) * std::sqrt(variance);
      }
    }
  }
}

void QuantumSpdeNoiseGenerator::warmup(const unsigned steps,
                                       const jams::Real temperature,
                                       jams::Real* noise,
                                       const jams::Real* sigma) {
  for (auto i = 0u; i < steps; ++i) {
    update(noise, sigma, temperature);
  }
}

void QuantumSpdeNoiseGenerator::warmup(
    const unsigned steps,
    const jams::MultiArray<jams::Real, 1>& temperature,
    jams::Real* noise,
    const jams::Real* sigma) {
  for (auto i = 0u; i < steps; ++i) {
    update(noise, sigma, temperature);
  }
}

void QuantumSpdeNoiseGenerator::update_zero_point(jams::Real* noise,
                                                  const jams::Real* sigma) {
  const auto eta_view = eta0_.host_view();
  auto zeta0 = zeta0_.mutable_host_view();
  const auto* eta = eta_view.data();
  auto* z0 = zeta0.data();
  const auto coeffs = zero_point_coefficients_;
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (auto x = 0; x < process_count_; ++x) {
    double s0 = 0.0;

    for (auto i = 0; i < 4; ++i) {
      const auto index = 4 * x + i;
      const double z_old = z0[index];
      const double e = static_cast<double>(eta[index] * coeffs.eta_scale[i]);
      const double z_new = coeffs.decay[i] * z_old + (1.0 - coeffs.decay[i]) * e;
      z0[index] = z_new;
      s0 += static_cast<double>(coeffs.weight[i]) * (e - z_new);
    }

    noise[x] += sigma[x] * coeffs.zero_point_scale * static_cast<jams::Real>(s0);
  }
}

void QuantumSpdeNoiseGenerator::update(jams::Real* noise,
                                       const jams::Real* sigma,
                                       const jams::Real temperature) {
  if (temperature <= jams::Real{0.0}) {
    zero_noise(noise);

    if (zero_point_) {
      fill_standard_normal(eta0_);
      update_zero_point(noise, sigma);
    }

    return;
  }

  if (!fast_coefficients_valid_ || temperature != fast_temperature_) {
    prepare_fixed_temperature_coefficients(temperature);
  }

  fill_standard_normal(eta1_);

  const auto eta_view = eta1_.host_view();
  auto zeta5 = zeta5_.mutable_host_view();
  auto zeta5p = zeta5p_.mutable_host_view();
  auto zeta6 = zeta6_.mutable_host_view();
  auto zeta6p = zeta6p_.mutable_host_view();
  const auto* eta = eta_view.data();
  auto* z5 = zeta5.data();
  auto* z5p = zeta5p.data();
  auto* z6 = zeta6.data();
  auto* z6p = zeta6p.data();
  const auto factor5 = fast_factor5_;
  const auto factor6 = fast_factor6_;
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (auto x = 0; x < process_count_; ++x) {
    const double force5 =
        static_cast<double>(eta[x] * factor5.eta_scale) * factor5.inv_omega2;
    const double old_z5 = z5[x];
    const double old_z5p = z5p[x];
    const double z5_new =
        factor5.m00 * old_z5 + factor5.m01 * old_z5p + factor5.force0 * force5;
    const double z5p_new =
        factor5.m10 * old_z5 + factor5.m11 * old_z5p + factor5.force1 * force5;
    z5[x] = z5_new;
    z5p[x] = z5p_new;

    const double force6 =
        static_cast<double>(eta[process_count_ + x] * factor6.eta_scale) * factor6.inv_omega2;
    const double old_z6 = z6[x];
    const double old_z6p = z6p[x];
    const double z6_new =
        factor6.m00 * old_z6 + factor6.m01 * old_z6p + factor6.force0 * force6;
    const double z6p_new =
        factor6.m10 * old_z6 + factor6.m11 * old_z6p + factor6.force1 * force6;
    z6[x] = z6_new;
    z6p[x] = z6p_new;

    const double s1 = kQuantumSpdeWeight5 * z5_new + kQuantumSpdeWeight6 * z6_new;
    noise[x] = temperature * sigma[x] * static_cast<jams::Real>(s1);
  }

  if (zero_point_) {
    fill_standard_normal(eta0_);
    update_zero_point(noise, sigma);
  }
}

void QuantumSpdeNoiseGenerator::update(
    jams::Real* noise,
    const jams::Real* sigma,
    const jams::MultiArray<jams::Real, 1>& temperature) {
  if (!profile_has_positive_temperature_) {
    zero_noise(noise);
  } else {
    fill_standard_normal(eta1_);

    const auto eta_view = eta1_.host_view();
    const auto temperature_view = temperature.host_view();
    const auto factor5 = profile_factor5_.host_view();
    const auto factor6 = profile_factor6_.host_view();
    auto zeta5 = zeta5_.mutable_host_view();
    auto zeta5p = zeta5p_.mutable_host_view();
    auto zeta6 = zeta6_.mutable_host_view();
    auto zeta6p = zeta6p_.mutable_host_view();
    const auto* eta = eta_view.data();
    const auto* temperature_values = temperature_view.data();
    const auto* f5 = factor5.data();
    const auto* f6 = factor6.data();
    auto* z5 = zeta5.data();
    auto* z5p = zeta5p.data();
    auto* z6 = zeta6.data();
    auto* z6p = zeta6p.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto x = 0; x < process_count_; ++x) {
      const jams::Real t = temperature_values[x];
      if (t <= jams::Real{0.0}) {
        noise[x] = jams::Real{0.0};
        continue;
      }

      const auto factor5_x = f5[x];
      const auto factor6_x = f6[x];

      const double force5 =
          static_cast<double>(eta[x] * factor5_x.eta_scale) * factor5_x.inv_omega2;
      const double old_z5 = z5[x];
      const double old_z5p = z5p[x];
      const double z5_new = factor5_x.m00 * old_z5 + factor5_x.m01 * old_z5p
          + factor5_x.force0 * force5;
      const double z5p_new = factor5_x.m10 * old_z5 + factor5_x.m11 * old_z5p
          + factor5_x.force1 * force5;
      z5[x] = z5_new;
      z5p[x] = z5p_new;

      const double force6 =
          static_cast<double>(eta[process_count_ + x] * factor6_x.eta_scale)
          * factor6_x.inv_omega2;
      const double old_z6 = z6[x];
      const double old_z6p = z6p[x];
      const double z6_new = factor6_x.m00 * old_z6 + factor6_x.m01 * old_z6p
          + factor6_x.force0 * force6;
      const double z6p_new = factor6_x.m10 * old_z6 + factor6_x.m11 * old_z6p
          + factor6_x.force1 * force6;
      z6[x] = z6_new;
      z6p[x] = z6p_new;

      const double s1 = kQuantumSpdeWeight5 * z5_new + kQuantumSpdeWeight6 * z6_new;
      noise[x] = t * sigma[x] * static_cast<jams::Real>(s1);
    }
  }

  if (zero_point_) {
    fill_standard_normal(eta0_);
    update_zero_point(noise, sigma);
  }
}

}  // namespace jams

ThermostatQuantumSpde::ThermostatQuantumSpde(const jams::Real& temperature,
                                             const jams::Real& sigma,
                                             const jams::Real timestep,
                                             const int num_spins)
    : Thermostat(temperature, sigma, timestep, num_spins) {
  std::cout << "\n  initialising quantum-spde-cpu thermostat\n";

  const libconfig::Setting* thermostat_settings = nullptr;
  if (globals::config->exists("thermostat")) {
    thermostat_settings = &globals::config->lookup("thermostat");
  }

  const bool do_zero_point = thermostat_settings != nullptr
      && jams::config_optional<bool>(*thermostat_settings, "zero_point", false);

  double t_warmup = 1e-10;  // 0.1 ns
  if (thermostat_settings != nullptr) {
    t_warmup = jams::config_optional<double>(*thermostat_settings, "warmup_time", t_warmup);
  }
  t_warmup = t_warmup / 1e-12;  // convert to ps
  const bool do_warmup = thermostat_settings != nullptr
      && jams::config_optional<bool>(*thermostat_settings, "warmup", false);
  const auto initialization = lowercase(thermostat_settings != nullptr
      ? jams::config_optional<std::string>(*thermostat_settings, "initialization", "stationary")
      : std::string("stationary"));
  if (initialization != "stationary" && initialization != "zero") {
    throw jams::ConfigException(
        *thermostat_settings, "initialization must be either 'stationary' or 'zero'");
  }

  double omega_max = 25.0 * kTwoPi;
  if (thermostat_settings != nullptr) {
    omega_max = jams::config_optional<double>(*thermostat_settings, "w_max", omega_max);
  }

  const double dt_thermostat = timestep;
  const double delta_tau = (dt_thermostat * kBoltzmannIU) / kHBarIU;

  std::cout << "    omega_max (THz) " << omega_max / (kTwoPi) << "\n";
  std::cout << "    hbar*w/kB " << (kHBarIU * omega_max) / (kBoltzmannIU) << "\n";
  std::cout << "    t_step " << dt_thermostat << "\n";
  std::cout << "    delta tau " << delta_tau << "\n";
  std::cout << "    initialization " << initialization << "\n";
  std::cout << "    warmup " << std::boolalpha << do_warmup << "\n";
  std::cout << "    zero_point " << do_zero_point << "\n";

  auto sigma_view = sigma_.mutable_host_view();
  auto alpha_view = globals::alpha.host_view();
  auto mu_view = globals::mus.host_view();
  auto gyro_view = globals::gyro.host_view();
  auto* sigma_values = sigma_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* mu_values = mu_view.data();
  const auto* gyro_values = gyro_view.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (auto i = 0; i < num_spins; ++i) {
    const auto value = static_cast<jams::Real>(
        kBoltzmannIU * std::sqrt((2.0 * alpha_values[i])
        / (kHBarIU * gyro_values[i] * mu_values[i])));
    const auto offset = 3 * i;
    sigma_values[offset] = value;
    sigma_values[offset + 1] = value;
    sigma_values[offset + 2] = value;
  }

  noise_generator_ = std::make_unique<jams::QuantumSpdeNoiseGenerator>(
      num_spins * 3, delta_tau, omega_max, do_zero_point);

  if (has_per_spin_temperature()) {
    process_temperature_.resize(num_spins * 3);
    const auto& spin_temperature = temperature_profile().temperature();
    const auto spin_temperature_view = spin_temperature.host_view();
    auto process_temperature_view = process_temperature_.mutable_host_view();
    const auto* spin_temperature_values = spin_temperature_view.data();
    auto* process_temperature_values = process_temperature_view.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto i = 0; i < num_spins; ++i) {
      const auto offset = 3 * i;
      process_temperature_values[offset] = spin_temperature_values[i];
      process_temperature_values[offset + 1] = spin_temperature_values[i];
      process_temperature_values[offset + 2] = spin_temperature_values[i];
    }
  }

  if (initialization == "stationary") {
    if (has_per_spin_temperature()) {
      std::cout << "initialising thermostat from per-spin stationary distribution" << std::endl;
      noise_generator_->initialize_stationary(process_temperature_);
    } else {
      std::cout << "initialising thermostat from stationary distribution @ ";
      std::cout << this->temperature() << "K" << std::endl;
      noise_generator_->initialize_stationary(this->temperature());
    }
  } else {
    if (has_per_spin_temperature()) {
      noise_generator_->initialize(jams::QuantumSpdeNoiseGenerator::Initialization::Zero,
                                   process_temperature_);
    } else {
      noise_generator_->initialize(jams::QuantumSpdeNoiseGenerator::Initialization::Zero,
                                   this->temperature());
    }
  }

  auto num_warm_up_steps = static_cast<unsigned>(t_warmup / dt_thermostat);
  if (do_warmup && num_warm_up_steps > 0) {
    auto noise_view = noise_.mutable_host_view();
    const auto sigma_values_view = sigma_.host_view();
    if (has_per_spin_temperature()) {
      std::cout << "warming up thermostat " << num_warm_up_steps
                << " steps with per-spin temperatures" << std::endl;
      noise_generator_->warmup(num_warm_up_steps, process_temperature_,
                               noise_view.data(), sigma_values_view.data());
    } else {
      std::cout << "warming up thermostat " << num_warm_up_steps << " steps @ ";
      std::cout << this->temperature() << "K" << std::endl;
      noise_generator_->warmup(num_warm_up_steps, this->temperature(),
                               noise_view.data(), sigma_values_view.data());
    }
  }
}

void ThermostatQuantumSpde::update() {
  auto noise_view = noise_.mutable_host_view();
  const auto sigma_view = sigma_.host_view();
  if (has_per_spin_temperature()) {
    noise_generator_->update(noise_view.data(), sigma_view.data(), process_temperature_);
    return;
  }

  noise_generator_->update(noise_view.data(), sigma_view.data(), this->temperature());
}
