#include <jams/metadynamics/cvars/cvar_reduced_mz.h>
#include <jams/core/globals.h>

std::string jams::CVarReducedMz::name() {
  return name_;
}


jams::CVarReducedMz::CVarReducedMz(const libconfig::Setting &settings) {

}


double jams::CVarReducedMz::value() {
  if (!cache_initialised_) {
    cached_magnetisation_ = calculate_magnetisation();
    cache_initialised_ = true;
  }

  return cached_magnetisation_[2] / norm(cached_magnetisation_);
}


double
jams::CVarReducedMz::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                           const Vec3 &spin_trial) {

  const Vec3 trial_magnetisation = cached_magnetisation_ - spin_initial + spin_trial;

  cached_i_ = i;
  cached_spin_initial_ = spin_initial;
  cached_spin_trial_ = spin_trial;
  cached_trial_magnetisation_ = trial_magnetisation;

  return trial_magnetisation[2] / norm(trial_magnetisation);
}


void jams::CVarReducedMz::spin_move_accepted(int i, const Vec3 &spin_initial,
                                             const Vec3 &spin_trial) {
  if (cache_initialised_ && cached_spin_initial_ == spin_initial
  && cached_spin_trial_ == spin_trial && cached_i_ == i) {
    cached_magnetisation_ = cached_trial_magnetisation_;
  } else {
    cached_magnetisation_ = calculate_magnetisation();
    cache_initialised_ = true;
  }
}


Vec3 jams::CVarReducedMz::calculate_magnetisation() {
  Vec3 magnetisation = {0, 0, 0};
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      magnetisation[j] += globals::s(i, j);
    }
  }
  return magnetisation;
}
