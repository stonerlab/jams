// cvar_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

double jams::CVarMagnetisation::value() {
  return cached_value() / num_mat_spins_;
}


double jams::CVarMagnetisation::calculate_expensive_value() {
  double magnetisation = 0.0;

  if (material_ == "All") {
      for (auto i = 0; i < globals::num_spins; ++i) {
          magnetisation += globals::s(i, magnetisation_component_);
      }
  } else {
      for (auto i =0; i < globals::num_spins; ++i) {
          if (lattice->atom_material_name(i) == material_) {
              magnetisation += globals::s(i, magnetisation_component_);
          }
      }
  }

  return magnetisation;
}


double
jams::CVarMagnetisation::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_trial) {

  const double trial_value = cached_value() - spin_initial[magnetisation_component_] + spin_trial[magnetisation_component_];

  set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);

  return trial_value / num_mat_spins_;
}


std::string jams::CVarMagnetisation::name() {
  return name_;
}


jams::CVarMagnetisation::CVarMagnetisation(const libconfig::Setting &settings) {
  std::string material;
  // New behaviour: if we only want to do metadynamics for one type of material in the lattice,
  // the material parameter needs to be specified. Probably worth adding a type of safety check.
  // If more than two materials are present, we give the option to choose which material metadynamics
  // is applied to. Of course, more than one collective variable can be specified, which allows us to run metadynamics
  // on up to two materials
  if (lattice->num_materials() >= 2) {
      std::string def = "All";
      material = config_optional<std::string>(settings, "material",def);
      if (material != "All" && !lattice->material_exists(material)) {
          throw std::runtime_error("Invalid material specified in magnetisation collective variable");
      }
  } else {
      material = lattice->material_name(0);
  }
  // Count the number of spins of each type.
  int count = 0;
  if (material=="All") {
      count = globals::num_spins;
  } else {
      for (auto i=0; i < globals::num_spins; ++i) {
          if (lattice->atom_material_name(i) == material) {
              count += 1;
          }
      }
  }

  auto component = config_required<std::string>(settings, "component");

  if (lowercase(component) == "x") {
    magnetisation_component_ = 0;
    name_ = "magnetisation_x";
    material_ = material;
    num_mat_spins_ = count;
  } else if (lowercase(component) == "y") {
    magnetisation_component_ = 1;
    name_ = "magnetisation_y";
    material_ = material;
    num_mat_spins_ = count;
  } else if (lowercase(component) == "z") {
    magnetisation_component_ = 2;
    name_ = "magnetisation_z";
    material_ = material;
    num_mat_spins_ = count;
  } else {
    throw std::runtime_error("'component' setting in magnetisation collective variable must be x, y or z");
  }
}




