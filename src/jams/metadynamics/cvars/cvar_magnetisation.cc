// cvar_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

double jams::CVarMagnetisation::value() {
  return cached_value() / num_mat_spins_;
}


double jams::CVarMagnetisation::calculate_expensive_value() {
  double magnetisation = 0.0;

  for (auto i =0; i < globals::num_spins; ++i) {
    // if the spin matches the specified material id, or we are considering
    // all spins in the system, then add that spin to the total magnetisation.
    if (lattice->atom_material_id(i) || material_==-1) {
      magnetisation += globals::s(i, magnetisation_component_);
    }
  }

  // TODO: Check if we should normalise this by the number of spins of relevant material
  // or not. We do this in spin_move_trial_value()...
  return magnetisation;
}


double
jams::CVarMagnetisation::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_trial) {
  if (lattice->atom_material_id(i) != material_) {
      set_cache_values(i, spin_initial, spin_trial, cached_value(), cached_value());
  }
  const double trial_value = cached_value() - spin_initial[magnetisation_component_] + spin_trial[magnetisation_component_];

  set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);

  return trial_value / num_mat_spins_;
}


std::string jams::CVarMagnetisation::name() {
  return name_;
}


jams::CVarMagnetisation::CVarMagnetisation(const libconfig::Setting &settings) {
  std::string def = "all";
  std::string material = def;

  if (settings.exists("material")) {
      material = config_optional<std::string>(settings, "material", def);
      if (material != "all" && !lattice->material_exists(material)) {
          throw std::runtime_error("Invalid material specified in magnetisation collective variable");
      }
  }
  material_ = (material==def) ? -1 : lattice->material_id(material);

  // Count the number of spins of each type.
  int count = 0;
  if (material_==-1) {
      count = globals::num_spins;
  } else {
      for (auto i=0; i < globals::num_spins; ++i) {
          if (lattice->atom_material_id(i)==material_) {
              count += 1;
          }
      }
  }

  num_mat_spins_ = count;

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




