#include <jams/initializer/init_spin_perturbation.h>

#include <cmath>
#include <limits>

#include <jams/core/globals.h>
#include <jams/helpers/finite.h>
#include <jams/helpers/random.h>

void jams::InitSpinPerturbation::execute(const libconfig::Setting& settings) {
  const double amplitude = read_numeric_setting<double>(settings["amplitude"], "amplitude");
  if (!is_finite(amplitude) || amplitude < 0.0) {
    throw ConfigException(settings, "amplitude must be finite and non-negative");
  }
  const int seed = read_integer_in_range(settings["seed"], "seed", 0,
                                        std::numeric_limits<int>::max());
  if (amplitude == 0.0) return;

  // A local generator leaves the simulation/thermostat RNG state untouched.
  pcg32 rng(static_cast<std::uint64_t>(seed));
  const auto moments = globals::mus.host_view();
  auto spins = globals::s.mutable_host_view();
  for (int i = 0; i < globals::num_spins; ++i) {
    if (moments(i) == 0.0) continue;
    const Vec<double, 3> spin = {spins(i, 0), spins(i, 1), spins(i, 2)};
    const double length = std::hypot(spin[0], spin[1], spin[2]);
    if (!is_finite(spin[0]) || !is_finite(spin[1]) || !is_finite(spin[2])
        || !is_finite(length) || length == 0.0) {
      throw ConfigException(settings, "spin ", i, " must have finite, non-zero length");
    }
    const auto direction = spin / length;
    const auto random = uniform_random_sphere<double>(rng);
    const auto transverse = random - dot(random, direction) * direction;
    const auto perturbed = direction + amplitude * transverse;
    // Normalisation implements (n + epsilon p) / sqrt(1 + epsilon^2 |p|^2).
    const double norm = std::hypot(perturbed[0], perturbed[1], perturbed[2]);
    if (!is_finite(norm) || norm == 0.0) {
      throw ConfigException(settings, "non-finite perturbation for spin ", i);
    }
    for (int j = 0; j < 3; ++j) {
      spins(i, j) = length * (perturbed[j] / norm);
    }
  }
}
