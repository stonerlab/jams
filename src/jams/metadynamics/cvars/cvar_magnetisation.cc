// cvar_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

double jams::CVarMagnetisation::value() {
  return cached_value() / num_selected_spins_;
}


double jams::CVarMagnetisation::calculate_expensive_value() {
  double magnetisation = 0.0;

  // No specific material is selected so use all spins
  if (selected_material_id_ == -1) {
    for (auto i =0; i < globals::num_spins; ++i) {
        magnetisation += globals::s(i, magnetisation_component_);
    }
    return magnetisation;
  }

  assert(selected_material_id_ >= 0 && selected_material_id_ < lattice->num_materials());

  // A specific material has been selected in the config so only include
  // that material in the sum
  for (auto i =0; i < globals::num_spins; ++i) {
    if (lattice->atom_material_id(i) == selected_material_id_) {
      magnetisation += globals::s(i, magnetisation_component_);
    }
  }

  return magnetisation;
}


double
jams::CVarMagnetisation::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_trial) {

  // Spin 'i' is of chosen material, or all materials are selected
  if (selected_material_id_ == -1 || lattice->atom_material_id(i) == selected_material_id_) {
    const double trial_value = cached_value() - spin_initial[magnetisation_component_] + spin_trial[magnetisation_component_];
    set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);
    return trial_value / num_selected_spins_;
  }

  // Spin 'i' is not of a selected material so the CV is not changed by this
  // trial move
  set_cache_values(i, spin_initial, spin_trial, cached_value(), cached_value());
  return cached_value() / num_selected_spins_;
}


std::string jams::CVarMagnetisation::name() {
  return name_;
}


jams::CVarMagnetisation::CVarMagnetisation(const libconfig::Setting &settings) {

  // The optional setting 'material' can be used to restrict the magnetisation
  // calculation to a single material. If the setting is omitted then all spins
  // are used to calculate the magnetisation.
  //
  // We stored the selected material by its id in `selected_material_id_`. If
  // all materials should be used then `selected_material_id_ = -1`.

  if (settings.exists("material")) {
    std::string material = config_required<std::string>(settings, "material");
    if (!lattice->material_exists(material)) {
        throw std::runtime_error("Invalid material specified in magnetisation collective variable");
    }
    selected_material_id_ = lattice->material_id(material);

    // record the number of spins of this material type
    num_selected_spins_ = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
      if (lattice->atom_material_id(i) == selected_material_id_) {
        num_selected_spins_++;
      }
    }
  } else {
    selected_material_id_ = -1;
    num_selected_spins_ = globals::num_spins;
  }

  assert((selected_material_id_ >= 0 && selected_material_id_ < lattice->num_materials()) || selected_material_id_ == -1);
  assert(num_selected_spins_ > 0);

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




