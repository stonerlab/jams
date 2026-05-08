#include <jams/hamiltonian/crystal_field.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/hamiltonian/tesseral_polynomial_evaluator.h>
#include <jams/helpers/exception.h>
#include <jams/maths/tesseral_harmonics.h>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

namespace {
constexpr int parity_sign(const int m) {
  return (m % 2 == 0) ? 1 : -1;
}
}

CrystalFieldHamiltonian::CrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size)
    : Hamiltonian(settings, size),
      energy_cutoff_(jams::config_required<double>(settings, "energy_cutoff")),
      crystal_field_spin_type_(CrystalFieldSpinType::kSpinUp) {
  std::cout << "energy_cutoff: " << energy_cutoff_ << "\n";

  auto spin_type_string = lowercase(settings["crystal_field_spin_type"]);

  if (spin_type_string == "up") {
    crystal_field_spin_type_ = CrystalFieldSpinType::kSpinUp;
    std::cout << "crystal_field_spin_type: up\n";
  } else if (spin_type_string == "down") {
    crystal_field_spin_type_ = CrystalFieldSpinType::kSpinDown;
    std::cout << "crystal_field_spin_type: down\n";
  } else {
    throw jams::ConfigException(settings["crystal_field_spin_type"], "must be 'up' or 'down'");
  }

  std::vector<std::vector<std::pair<int, double>>> spin_terms(size);
  std::vector<bool> spin_has_crystal_field(size, false);

  for (auto n = 0; n < settings["crystal_field_coefficients"].getLength(); ++n) {

    const auto &cf_params = settings["crystal_field_coefficients"][n];

    // validate settings
    if (cf_params[0].isNumber()) {
      if (int(cf_params[0]) < 1 || int(cf_params[0]) > globals::lattice->num_basis_sites()) {
        throw jams::ConfigException(cf_params[0],
                                    "unit cell index must be between 1 and ",
                                    globals::lattice->num_basis_sites());
      }
    } else if (cf_params[0].isString()) {
      if (!globals::lattice->material_exists(cf_params[0])) {
        throw jams::ConfigException(cf_params[0], "material ", cf_params[0].c_str(), " does not exist in config file");
      }
    } else {
      throw jams::ConfigException(cf_params[0], "must be a unit cell index or material name");
    }

    double J = cf_params[1];
    double alphaJ = cf_params[2];
    double betaJ = cf_params[3];
    double gammaJ = cf_params[4];
    auto cf_coefficient_filename = cf_params[5].c_str();

    std::map<int, double> stevens_prefactor;
    stevens_prefactor.insert({2, J * (J - 0.5) * alphaJ});
    stevens_prefactor.insert({4, J * (J - 0.5) * (J - 1) * (J - 1.5) * betaJ});
    stevens_prefactor.insert({6, J * (J - 0.5) * (J - 1) * (J - 1.5) * (J - 2) * (J - 2.5) * gammaJ});

    auto tesseral_coefficients = convert_spherical_to_tesseral(
        read_crystal_field_coefficients_from_file(cf_coefficient_filename), energy_cutoff_);

    if (debug_is_enabled()) {
      std::cout << "tesseral harmonic coefficients " << n << ":" << std::endl;
      for (auto const &[key, val] : tesseral_coefficients) {
        std::cout << "  " << key.first << " " << key.second << " " << val << std::endl;
      }
    }

    std::vector<std::pair<int, double>> crystal_field_terms;
    for (auto const &[lm, B_lm] : tesseral_coefficients) {
      const auto &[l, m] = lm;
      // We don't use the 0,0 coefficients because they are constant energy offsets.
      if (l == 0 && m == 0) {
        continue;
      }

      const auto coefficient = input_energy_unit_conversion_ * stevens_prefactor[l] * B_lm
          * jams::tesseral_racah_normalisation_scale_lookup<double>(l, m);
      if (coefficient == 0.0) {
        continue;
      }

      crystal_field_terms.emplace_back(jams::tesseral_key(l, m), coefficient);
    }

    for (auto i = 0u; i < size; i++) {
      if (cf_params[0].isNumber()) {
        if (globals::lattice->lattice_site_basis_index(i) != int(cf_params[0]) - 1) {
          continue;
        }
      }

      if (cf_params[0].isString()) {
        if (globals::lattice->lattice_site_material_name(i) != std::string(cf_params[0])) {
          continue;
        }
      }

      if (spin_has_crystal_field[i]) {
        throw std::runtime_error("crystal field is specified more than once for atom " + std::to_string(i));
      }

      spin_terms[i] = crystal_field_terms;
      spin_has_crystal_field[i] = true;
    }
  }

  zero(spin_pointer_.resize(size + 1));

  auto total_terms = 0;
  for (auto i = 0u; i < size; ++i) {
    spin_pointer_(i) = total_terms;
    total_terms += int(spin_terms[i].size());
  }
  spin_pointer_(size) = total_terms;

  tesseral_keys_.resize(total_terms);
  tesseral_coefficients_.resize(total_terms);

  auto term_index = 0;
  for (const auto& terms : spin_terms) {
    for (const auto& [key, coefficient] : terms) {
      tesseral_keys_(term_index) = key;
      tesseral_coefficients_(term_index) = coefficient;
      ++term_index;
    }
  }
}

CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap CrystalFieldHamiltonian::read_crystal_field_coefficients_from_file(
    const std::string& filename) {
  std::ifstream fs(filename);
  if (fs.fail()) {
    throw jams::FileException(filename, "failed to open file");
  }

  SphericalHarmonicCoefficientMap coefficients;

  // We first populate an empty coefficient map. This ensures that the map is always full and ordered
  // so that it is always a consistent length.
  for (auto l : {0, 2, 4, 6}) {
    for (auto m = -l; m <= l; ++m) {
      coefficients.insert({{l, m}, {0, 0}});
    }
  }

  int line_number = 0;
  for (std::string line; getline(fs, line);) {
    if (string_is_comment(line)) {
      continue;
    }

    std::stringstream is(line);

    int l, m;
    double Re_Blm_up, Im_Blm_up, Re_Blm_down, Im_Blm_down;

    is >> l >> m >> Re_Blm_up >> Im_Blm_up >> Re_Blm_down >> Im_Blm_down;

    if (!jams::util::is_in_list(l, {0, 2, 4, 6})) {
      throw jams::FileException(filename, "line ", line_number, ": 'l' must be 0, 2, 4 or 6");
    }

    if (m < -l || m > l) {
      throw jams::FileException(filename, "line ", line_number, ": 'm' must be -l <= m <= l");
    }

    // We use '.at()' to check if the element already exists. It should do because we populated a list of zeros
    // above and only allowed values of l and m should exist.
    // 'up' and 'down' should be the same, but there will be slight differences due to numerical precision. In
    // principle we could average the values, but this then causes difficulties if we convert to tesseral
    // harmonics which are purely real. The averaging means that it becomes harder to check that the real part is
    // purely real.

    const auto &[Re_Blm, Im_Blm] = (crystal_field_spin_type_ == CrystalFieldSpinType::kSpinUp)
                                   ? std::tie(Re_Blm_up, Im_Blm_up) : std::tie(Re_Blm_down, Im_Blm_down);

    coefficients.at({l, m}) = {Re_Blm, Im_Blm};

    line_number++;
  }

  return coefficients;
}

CrystalFieldHamiltonian::TesseralHarmonicCoefficientMap CrystalFieldHamiltonian::convert_spherical_to_tesseral(
    const CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap &spherical_coefficients, const double zero_epsilon) {

  TesseralHarmonicCoefficientMap tesseral_coefficients;

  for (auto l : {0, 2, 4, 6}) {
    for (auto m = -l; m <= l; ++m) {

      std::complex<double> C_lm = {0.0, 0.0};
      if (m == 0) {
        C_lm = spherical_coefficients.at({l, m});
      } else if (m > 0) {
        const auto phase = static_cast<double>(parity_sign(m));
        C_lm = kSqrtOne_Two * (spherical_coefficients.at({l, -m}) + phase * spherical_coefficients.at({l, m}));
      } else if (m < 0) {
        const auto phase = static_cast<double>(parity_sign(m));
        C_lm = kImagOne * kSqrtOne_Two
            * (phase * spherical_coefficients.at({l, -m}) - spherical_coefficients.at({l, m}));
      }

      // We need to check that the imaginary parts are very close to zero. However, this depends on the units
      // we are in, so the zero_epsilon we pass in must be meaningful in the current units.
      if (approximately_zero(C_lm.imag(), zero_epsilon)) {
        C_lm.imag(0.0);
      } else {
        throw std::domain_error("conversion from spherical to tesseral harmonics did not produce purely real values");
      }

      if (approximately_zero(C_lm.real(), zero_epsilon)) {
        C_lm.real(0.0);
      }

      tesseral_coefficients[{l, m}] = C_lm.real();
    }
  }

  return tesseral_coefficients;
}

jams::Vec<jams::Real, 3> CrystalFieldHamiltonian::calculate_field(int i, jams::Real time) {
  if (spin_pointer_(i) == spin_pointer_(i + 1)) {
    return {0.0, 0.0, 0.0};
  }

  const auto spins = std::as_const(globals::s).host_view();
  double h[3];
  jams::tesseral_polynomial::negative_gradient_from_local_terms(
      spin_pointer_(i),
      spin_pointer_(i + 1),
      std::as_const(tesseral_keys_).host_data(),
      std::as_const(tesseral_coefficients_).host_data(),
      double(spins(i, 0)),
      double(spins(i, 1)),
      double(spins(i, 2)),
      h);

  return {jams::Real(h[0]), jams::Real(h[1]), jams::Real(h[2])};
}

jams::Real CrystalFieldHamiltonian::calculate_energy(int i, jams::Real time) {
  const auto spins = std::as_const(globals::s).host_view();
  return crystal_field_energy(i, {double(spins(i, 0)), double(spins(i, 1)), double(spins(i, 2))});
}

jams::Real CrystalFieldHamiltonian::calculate_energy_difference(int i, const jams::Vec<double, 3> &spin_initial, const jams::Vec<double, 3> &spin_final,
                                                            jams::Real time) {
  return crystal_field_energy(i, spin_final) - crystal_field_energy(i, spin_initial);
}
jams::Real CrystalFieldHamiltonian::crystal_field_energy(int i, const jams::Vec<double, 3> &s) {
  if (spin_pointer_(i) == spin_pointer_(i + 1)) {
    return 0.0;
  }

  return static_cast<jams::Real>(
      jams::tesseral_polynomial::energy_from_local_terms(
          spin_pointer_(i),
          spin_pointer_(i + 1),
          std::as_const(tesseral_keys_).host_data(),
          std::as_const(tesseral_coefficients_).host_data(),
          s[0],
          s[1],
          s[2]));
}
