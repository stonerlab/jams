// cvar_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/core/globals.h>


double jams::CVarMagnetisation::value() {
  return cached_value() / globals::num_spins;
}


double jams::CVarMagnetisation::calculate_expensive_value() {
  double magnetisation = 0.0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    magnetisation += globals::s(i, magnetisation_component_);
  }
  return magnetisation;
}


double jams::CVarMagnetisation::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_trial) {

  const double trial_value = cached_value() - spin_initial[magnetisation_component_] + spin_trial[magnetisation_component_];

  set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);

  return trial_value / globals::num_spins;
}


std::string jams::CVarMagnetisation::name() {
  return name_;
}


jams::CVarMagnetisation::CVarMagnetisation(const libconfig::Setting &settings) {
  auto component = config_required<std::string>(settings, "component");

  if (lowercase(component) == "x") {
    magnetisation_component_ = 0;
    name_ = "magnetisation_x";
  } else if (lowercase(component) == "y") {
    magnetisation_component_ = 1;
    name_ = "magnetisation_y";
  } else if (lowercase(component) == "z") {
    magnetisation_component_ = 2;
    name_ = "magnetisation_z";
  } else {
    throw std::runtime_error("'component' setting in magnetisation collective variable must be x, y or z");
  }
}




