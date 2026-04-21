#ifndef JAMS_SOLVERS_LLG_SPIN_TORQUE_TERMS_H
#define JAMS_SOLVERS_LLG_SPIN_TORQUE_TERMS_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <libconfig.h++>

#include "jams/containers/cell.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"
#include "jams/solvers/solver_descriptor.h"

namespace jams::solvers {

struct LLGSpinTorqueField {
  jams::MultiArray<double, 2> torque;
  int term_count = 0;

  [[nodiscard]] bool enabled() const {
    return term_count > 0;
  }
};

namespace detail {

inline std::vector<std::string> read_string_list(const libconfig::Setting& settings,
                                                 const std::string& name) {
  std::vector<std::string> values;
  if (!settings.exists(name)) {
    return values;
  }

  const auto& list = settings.lookup(name);
  values.reserve(list.getLength());
  for (auto i = 0; i < list.getLength(); ++i) {
    values.emplace_back(list[i].c_str());
  }
  return values;
}

inline std::vector<int> read_int_list(const libconfig::Setting& settings,
                                      const std::string& name) {
  std::vector<int> values;
  if (!settings.exists(name)) {
    return values;
  }

  const auto& list = settings.lookup(name);
  values.reserve(list.getLength());
  for (auto i = 0; i < list.getLength(); ++i) {
    values.emplace_back(static_cast<int>(list[i]));
  }
  return values;
}

inline bool value_in_list(const int value, const std::vector<int>& values) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

inline bool value_in_list(const std::string& value, const std::vector<std::string>& values) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

inline bool spin_matches_selector(const int spin_index, const libconfig::Setting* selector) {
  if (selector == nullptr) {
    return true;
  }

  if (selector->exists("material")) {
    const std::string material = (*selector)["material"].c_str();
    if (globals::lattice->lattice_site_material_name(spin_index) != material) {
      return false;
    }
  }

  const auto materials = read_string_list(*selector, "materials");
  if (!materials.empty()
      && !value_in_list(globals::lattice->lattice_site_material_name(spin_index), materials)) {
    return false;
  }

  if (selector->exists("basis_site")) {
    if (globals::lattice->lattice_site_basis_index(spin_index)
        != static_cast<unsigned>((*selector)["basis_site"])) {
      return false;
    }
  }

  const auto basis_sites = read_int_list(*selector, "basis_sites");
  if (!basis_sites.empty()
      && !value_in_list(static_cast<int>(globals::lattice->lattice_site_basis_index(spin_index)), basis_sites)) {
    return false;
  }

  const auto sites = read_int_list(*selector, "sites");
  if (!sites.empty() && !value_in_list(spin_index, sites)) {
    return false;
  }

  if (selector->exists("surface_layers")) {
    const int surface_layers = std::max(0, static_cast<int>((*selector)["surface_layers"]));
    bool is_surface = false;
    const auto offset = globals::lattice->cell_offset(spin_index);
    const auto size = globals::lattice->size();

    for (auto dim = 0; dim < 3; ++dim) {
      if (globals::lattice->is_periodic(dim)) {
        continue;
      }

      const int lower = surface_layers;
      const int upper = size[dim] - surface_layers;
      if (offset[dim] < lower || offset[dim] >= upper) {
        is_surface = true;
        break;
      }
    }

    if (!is_surface) {
      return false;
    }
  }

  return true;
}

inline double spin_torque_coefficient_from_sot_parameters(const double spin_hall_angle,
                                                          const double charge_current_density_si) {
  auto charge_current_density = charge_current_density_si;
  charge_current_density = charge_current_density
      / (kMeterToNanometer * kMeterToNanometer * kSecondToPicosecond);

  const double volume_per_atom = pow3(globals::lattice->parameter() * kMeterToNanometer)
      * volume(globals::lattice->get_unitcell()) / double(globals::lattice->num_basis_sites());

  return kHBarIU * spin_hall_angle * charge_current_density
      * std::pow(volume_per_atom, 2.0 / 3.0) / (2.0 * kElementaryCharge);
}

inline Vec3 spin_torque_vector_from_term(const libconfig::Setting& term) {
  const auto module_name = lowercase(static_cast<const char*>(term["module"]));
  const auto spin_polarisation = jams::config_required<Vec3>(term, "spin_polarisation");

  double coefficient = 0.0;
  if (term.exists("coefficient")) {
    coefficient = jams::config_required<double>(term, "coefficient");
  } else if (module_name == "sot") {
    coefficient = spin_torque_coefficient_from_sot_parameters(
        jams::config_required<double>(term, "spin_hall_angle"),
        jams::config_required<double>(term, "charge_current_density"));
  } else {
    throw std::runtime_error("dynamics term '" + module_name
                             + "' requires either coefficient or SOT parameters");
  }

  return coefficient * spin_polarisation;
}

inline Vec3 legacy_sot_vector_from_solver_settings(const libconfig::Setting& settings) {
  const auto spin_polarisation = jams::config_required<Vec3>(settings, "spin_polarisation");
  const auto coefficient = spin_torque_coefficient_from_sot_parameters(
      jams::config_required<double>(settings, "spin_hall_angle"),
      jams::config_required<double>(settings, "charge_current_density"));
  return coefficient * spin_polarisation;
}

inline void accumulate_spin_torque_term(jams::MultiArray<double, 2>& torque,
                                        const Vec3& term_vector,
                                        const libconfig::Setting* selector) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    if (!spin_matches_selector(i, selector)) {
      continue;
    }

    for (auto n = 0; n < 3; ++n) {
      torque(i, n) += term_vector[n];
    }
  }
}

}  // namespace detail

inline LLGSpinTorqueField build_llg_spin_torque_field(const libconfig::Setting& solver_settings,
                                                      const SolverDescriptor& descriptor) {
  LLGSpinTorqueField field;
  field.torque.resize(globals::num_spins, 3);
  field.torque.zero();

  if (descriptor.equation != EquationKind::LLG) {
    return field;
  }

  if (descriptor.legacy_sot_alias) {
    const libconfig::Setting* selector = solver_settings.exists("selector")
        ? &solver_settings.lookup("selector") : nullptr;
    detail::accumulate_spin_torque_term(
        field.torque, detail::legacy_sot_vector_from_solver_settings(solver_settings), selector);
    ++field.term_count;
  }

  if (!globals::config->exists("dynamics")) {
    return field;
  }

  const auto& dynamics = globals::config->lookup("dynamics");
  if (!dynamics.exists("terms")) {
    return field;
  }

  const auto& terms = dynamics.lookup("terms");
  for (auto i = 0; i < terms.getLength(); ++i) {
    const auto& term = terms[i];
    const auto module_name = lowercase(static_cast<const char*>(term["module"]));
    if (module_name != "stt" && module_name != "slonczewski" && module_name != "sot") {
      throw std::runtime_error("unsupported LLG dynamics term '" + module_name + "'");
    }

    const libconfig::Setting* selector = term.exists("selector")
        ? &term.lookup("selector") : nullptr;
    detail::accumulate_spin_torque_term(field.torque, detail::spin_torque_vector_from_term(term), selector);
    ++field.term_count;
  }

  return field;
}

}  // namespace jams::solvers

#endif  // JAMS_SOLVERS_LLG_SPIN_TORQUE_TERMS_H
