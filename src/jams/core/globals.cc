// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/globals.h"

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace globals {

namespace {

constexpr std::array<double, 3> kDefaultVacancySpin = {0.0, 0.0, 1.0};

template <typename Spins>
void ensure_vacancy_spin_is_unit(const int spin, Spins& spins) {
  const double x = spins(spin, 0);
  const double y = spins(spin, 1);
  const double z = spins(spin, 2);
  const double length = std::sqrt(x * x + y * y + z * z);

  if (!std::isfinite(length) || length <= std::numeric_limits<double>::min()) {
    for (auto component = 0; component < 3; ++component) {
      spins(spin, component) = kDefaultVacancySpin[component];
    }
    return;
  }

  spins(spin, 0) = x / length;
  spins(spin, 1) = y / length;
  spins(spin, 2) = z / length;
}

}  // namespace

void sync_magnetic_moment_data() {
  if (num_spins < 0) {
    throw std::runtime_error("global spin count must be non-negative");
  }

  if (static_cast<int>(mus.elements()) != num_spins) {
    std::ostringstream message;
    message << "global magnetic moment array has " << mus.elements()
            << " entries but num_spins is " << num_spins;
    throw std::runtime_error(message.str());
  }

  inv_mus.resize(num_spins);
  num_magnetic_spins = 0;

  const auto moments = mus.host_view();
  auto inverse_moments = inv_mus.mutable_host_view();

  const bool can_repair_spins =
      static_cast<int>(s.extent(0)) == num_spins && s.extent(1) == 3;
  auto spins = can_repair_spins ? s.mutable_host_view() : decltype(s.mutable_host_view()){};

  for (int spin = 0; spin < num_spins; ++spin) {
    const double moment = static_cast<double>(moments(spin));
    if (!std::isfinite(moment)) {
      std::ostringstream message;
      message << "magnetic moment for spin " << spin << " is not finite";
      throw std::runtime_error(message.str());
    }
    if (moment < 0.0) {
      std::ostringstream message;
      message << "magnetic moment for spin " << spin << " is negative";
      throw std::runtime_error(message.str());
    }

    if (moment > 0.0) {
      inverse_moments(spin) = static_cast<jams::Real>(1.0 / moment);
      ++num_magnetic_spins;
      continue;
    }

    inverse_moments(spin) = jams::Real{0.0};
    if (can_repair_spins) {
      ensure_vacancy_spin_is_unit(spin, spins);
    }
  }
}

int first_magnetic_spin() {
  if (static_cast<int>(inv_mus.elements()) != num_spins) {
    sync_magnetic_moment_data();
  }

  const auto inverse_moments = inv_mus.host_view();
  for (int spin = 0; spin < num_spins; ++spin) {
    if (inverse_moments(spin) > jams::Real{0.0}) {
      return spin;
    }
  }

  throw std::runtime_error("operation requires at least one magnetic spin");
}

}  // namespace globals
