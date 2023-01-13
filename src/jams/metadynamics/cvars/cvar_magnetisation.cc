// cvar_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

double jams::CVarMagnetisation::value() {
  auto magnetisation = cached_value();

  if (normalize_) {
    magnetisation /= total_selected_magnetization_;
  }

  if (magnetisation_component_ == -1) {
    return sqrt(magnetisation[0]*magnetisation[0] + magnetisation[1]*magnetisation[1]);
  }

  if (magnetisation_component_ == 3) {
    return norm(magnetisation);
  }

  return magnetisation[magnetisation_component_];
}


Vec3 jams::CVarMagnetisation::calculate_expensive_cache_value() {
  Vec3 magnetisation = {0.0, 0.0, 0.0};

  for (auto i = 0; i < globals::num_spins; ++i) {
    if (selected_material_id_ == -1 || globals::lattice->atom_material_id(i) == selected_material_id_) {
      for (auto n = 0; n < 3; ++n) {
        magnetisation[n] += (globals::mus(i)/kBohrMagnetonIU) * globals::s(i, n);
      }
    }
  }

  return magnetisation;
}


double
jams::CVarMagnetisation::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_trial) {

  Vec3 magnetisation = cached_value();
  // Spin 'i' is of chosen material, or all materials are selected then we
  // adjust the magnetisation with the difference in the spin. Otherwise the
  // magnetisation remains unchanged.
  if (selected_material_id_ == -1 || globals::lattice->atom_material_id(i) == selected_material_id_) {
    magnetisation = magnetisation + (globals::mus(i)/kBohrMagnetonIU)*(spin_trial - spin_initial);
    set_cache_values(i, spin_initial, spin_trial, cached_value(), magnetisation);
  }

  if (normalize_) {
    magnetisation /= total_selected_magnetization_;
  }

  if (magnetisation_component_ == -1) {
    return sqrt(magnetisation[0]*magnetisation[0] + magnetisation[1]*magnetisation[1]);
  }

  if (magnetisation_component_ == 3) {
    return norm(magnetisation);
  }


  return magnetisation[magnetisation_component_];

}


std::string jams::CVarMagnetisation::name() {
  return name_;
}


jams::CVarMagnetisation::CVarMagnetisation(const libconfig::Setting &settings) {

  normalize_ = config_optional<bool>(settings, "normalize", true);


  // The optional setting 'material' can be used to restrict the magnetisation
  // calculation to a single material. If the setting is omitted then all spins
  // are used to calculate the magnetisation.
  //
  // We stored the selected material by its id in `selected_material_id_`. If
  // all materials should be used then `selected_material_id_ = -1`.

  if (settings.exists("material")) {
    std::string material = config_required<std::string>(settings, "material");
    if (!globals::lattice->material_exists(material)) {
        throw std::runtime_error("Invalid material specified in magnetisation collective variable");
    }
    selected_material_id_ = globals::lattice->material_id(material);

    // record the number of spins of this material type
    total_selected_magnetization_ = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
      if (globals::lattice->atom_material_id(i) == selected_material_id_) {
        total_selected_magnetization_ += (globals::mus(i)/kBohrMagnetonIU);
      }
    }
  } else {
    selected_material_id_ = -1;
    for (auto i = 0; i < globals::num_spins; ++i) {
      total_selected_magnetization_ += (globals::mus(i)/kBohrMagnetonIU);
    }
  }

  assert((selected_material_id_ >= 0 && selected_material_id_ < globals::lattice->num_materials()) || selected_material_id_ == -1);
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
  } else if (lowercase(component) == "magnitude") {
    magnetisation_component_ = 3;
    name_ = "magnetisation_z";
  } else if (lowercase(component) == "transverse") {
    magnetisation_component_ = -1;
    name_ = "magnetisation_transverse";
  } else {
    throw std::runtime_error("'component' setting in magnetisation collective variable must be x, y, z, transverse or magnitude");
  }
}




