// cvar_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_moment.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

double jams::CVarMoment::value() {
  return cached_value() / total_selected_moments_;
}


double jams::CVarMoment::calculate_expensive_cache_value() {
  double moment = 0.0;

  for (auto i = 0; i < globals::num_spins; ++i) {
    if (selected_material_id_ == -1 || globals::lattice->atom_material_id(i) == selected_material_id_) {
        moment += (globals::mus(i)/kBohrMagnetonIU)
            * sqrt(globals::s(i, 0)*globals::s(i, 0) + globals::s(i, 1)*globals::s(i, 1) + globals::s(i, 2)*globals::s(i, 2) );
    }
  }

  return moment;
}


double
jams::CVarMoment::spin_move_trial_value(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_trial) {

  double moment = cached_value();
  // Spin 'i' is of chosen material, or all materials are selected then we
  // adjust the magnetisation with the difference in the spin. Otherwise the
  // magnetisation remains unchanged.
  if (selected_material_id_ == -1 || globals::lattice->atom_material_id(i) == selected_material_id_) {
    moment = moment + (globals::mus(i)/kBohrMagnetonIU)*(norm(spin_trial) - norm(spin_initial));
    set_cache_values(i, spin_initial, spin_trial, cached_value(), moment);
  }

  return moment / total_selected_moments_;

}


std::string jams::CVarMoment::name() {
  return name_;
}


jams::CVarMoment::CVarMoment(const libconfig::Setting &settings) {

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
    total_selected_moments_ = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
      if (globals::lattice->atom_material_id(i) == selected_material_id_) {
        total_selected_moments_ += (globals::mus(i)/kBohrMagnetonIU);
      }
    }
  } else {
    selected_material_id_ = -1;
    for (auto i = 0; i < globals::num_spins; ++i) {
      total_selected_moments_ += (globals::mus(i)/kBohrMagnetonIU);
    }
  }

  assert((selected_material_id_ >= 0 && selected_material_id_ < globals::lattice->num_materials()) || selected_material_id_ == -1);
  assert(total_selected_moments_ > 0);
}




