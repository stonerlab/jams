#include <jams/hamiltonian/crystal_field.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/helpers/exception.h>
#include <jams/hamiltonian/tesseral_polynomial_evaluator.h>
#include <jams/interface/config.h>
#include <jams/maths/tesseral_harmonics.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace {
constexpr int parity_sign(const int m) {
  return (m % 2 == 0) ? 1 : -1;
}

bool terms_are_axial(const AnisotropyPolynomialHamiltonian::TesseralKeyCoefficientMap& terms)
{
  return std::all_of(terms.begin(), terms.end(), [](const auto& term) {
    return term.second == jams::Real{0}
        || jams::tesseral_polynomial::axial_coefficient_index_from_key(term.first) >= 0;
  });
}

bool is_valid_crystal_field_l(const int l)
{
  return jams::util::is_in_list(l, {0, 2, 4, 6});
}

CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap zero_spherical_coefficients()
{
  CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap coefficients;
  for (auto l : {0, 2, 4, 6}) {
    for (auto m = -l; m <= l; ++m) {
      coefficients.insert({{l, m}, {0.0, 0.0}});
    }
  }
  return coefficients;
}

CrystalFieldHamiltonian::CrystalFieldSpinType read_crystal_field_spin_type_setting(
    const libconfig::Setting& setting)
{
  if (!jams::is_string_setting(setting)) {
    throw jams::ConfigException(setting, "crystal_field_spin_type must be 'up' or 'down'");
  }

  auto spin_type_string = lowercase(jams::read_string_setting(setting, "crystal_field_spin_type"));

  if (spin_type_string == "up") {
    return CrystalFieldHamiltonian::CrystalFieldSpinType::kSpinUp;
  }
  if (spin_type_string == "down") {
    return CrystalFieldHamiltonian::CrystalFieldSpinType::kSpinDown;
  }

  throw jams::ConfigException(setting, "crystal_field_spin_type must be 'up' or 'down'");
}

std::optional<CrystalFieldHamiltonian::CrystalFieldSpinType> read_optional_crystal_field_spin_type(
    const libconfig::Setting& settings)
{
  if (!settings.exists("crystal_field_spin_type")) {
    return std::nullopt;
  }

  return read_crystal_field_spin_type_setting(settings["crystal_field_spin_type"]);
}

const char* crystal_field_spin_type_name(const CrystalFieldHamiltonian::CrystalFieldSpinType spin_type)
{
  switch (spin_type) {
    case CrystalFieldHamiltonian::CrystalFieldSpinType::kSpinUp:
      return "up";
    case CrystalFieldHamiltonian::CrystalFieldSpinType::kSpinDown:
      return "down";
  }

  throw std::invalid_argument("invalid crystal field spin type");
}

std::map<int, double> stevens_prefactors(
    const double J,
    const double alphaJ,
    const double betaJ,
    const double gammaJ)
{
  return {
      {2, J * (J - 0.5) * alphaJ},
      {4, J * (J - 0.5) * (J - 1) * (J - 1.5) * betaJ},
      {6, J * (J - 0.5) * (J - 1) * (J - 1.5) * (J - 2) * (J - 2.5) * gammaJ}
  };
}
}

CrystalFieldHamiltonian::CrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size)
    : AnisotropyPolynomialHamiltonian(settings, size, EmptyStorageTag{}),
      energy_cutoff_(jams::config_required<double>(settings, "energy_cutoff")) {
  std::cout << "energy_cutoff: " << energy_cutoff_ << "\n";

  const auto crystal_field_spin_type = read_optional_crystal_field_spin_type(settings);
  if (crystal_field_spin_type.has_value()) {
    std::cout << "crystal_field_spin_type: " << crystal_field_spin_type_name(*crystal_field_spin_type) << "\n";
  }

  if (!settings.exists("crystal_field_coefficients")) {
    throw jams::ConfigException(settings, "missing crystal_field_coefficients");
  }

  const auto& crystal_field_settings = settings["crystal_field_coefficients"];
  if (!crystal_field_settings.isList()) {
    throw jams::ConfigException(crystal_field_settings, "crystal_field_coefficients must be a list");
  }
  if (crystal_field_settings.getLength() == 0) {
    throw jams::ConfigException(crystal_field_settings, "crystal_field_coefficients must contain at least one entry");
  }

  std::vector<TesseralKeyCoefficientMap> spin_terms(size);
  std::vector<bool> spin_has_crystal_field(size, false);

  for (auto n = 0; n < crystal_field_settings.getLength(); ++n) {

    const auto &cf_params = crystal_field_settings[n];
    if (!cf_params.isList()) {
      throw jams::ConfigException(cf_params, "crystal field coefficients entry must be a list");
    }
    if (cf_params.getLength() == 0) {
      throw jams::ConfigException(cf_params, "crystal field coefficients entry must not be empty");
    }

    int motif_position = -1;
    int material_id = -1;
    if (jams::is_integer_setting(cf_params[0])) {
      motif_position = jams::read_integer_setting(cf_params[0], "unit cell index") - 1;
      if (motif_position < 0 || motif_position >= globals::lattice->num_basis_sites()) {
        throw jams::ConfigException(cf_params[0],
                                    "unit cell index must be between 1 and ",
                                    globals::lattice->num_basis_sites());
      }
    } else if (cf_params[0].isNumber()) {
      throw jams::ConfigException(cf_params[0], "unit cell index must be an integer");
    } else if (jams::is_string_setting(cf_params[0])) {
      const auto material = jams::read_string_setting(cf_params[0], "material");
      if (!globals::lattice->material_exists(material)) {
        throw jams::ConfigException(cf_params[0], "material ", material, " does not exist in config file");
      }
      material_id = globals::lattice->material_index(material);
    } else {
      throw jams::ConfigException(cf_params[0], "must be a unit cell index or material name");
    }

    int parameter_start = 1;
    const auto local_axes = read_optional_local_axes(cf_params, 1, "crystal field", parameter_start);
    if (cf_params.getLength() < parameter_start + 5) {
      throw jams::ConfigException(
          cf_params,
          "crystal field coefficients entry must have format "
          "(target, [u, v, w], J, alphaJ, betaJ, gammaJ, cf_param_filename) or "
          "(target, [u, v, w], J, alphaJ, betaJ, gammaJ, (l, m, real, imaginary), ...)");
    }

    const double J = jams::read_numeric_setting<double>(cf_params[parameter_start], "J");
    const double alphaJ = jams::read_numeric_setting<double>(cf_params[parameter_start + 1], "alphaJ");
    const double betaJ = jams::read_numeric_setting<double>(cf_params[parameter_start + 2], "betaJ");
    const double gammaJ = jams::read_numeric_setting<double>(cf_params[parameter_start + 3], "gammaJ");
    const auto stevens_prefactor = stevens_prefactors(J, alphaJ, betaJ, gammaJ);

    SphericalHarmonicCoefficientMap spherical_coefficients;
    const auto coefficient_start = parameter_start + 4;
    if (jams::is_string_setting(cf_params[coefficient_start])) {
      if (cf_params.getLength() != coefficient_start + 1) {
        throw jams::ConfigException(
            cf_params,
            "file-based crystal field coefficients entry must have format "
            "(target, [u, v, w], J, alphaJ, betaJ, gammaJ, cf_param_filename)");
      }
      if (!crystal_field_spin_type.has_value()) {
        throw jams::ConfigException(
            settings,
            "crystal_field_spin_type is required when crystal field coefficients are read from a file");
      }

      const auto cf_coefficient_filename = jams::read_string_setting(
          cf_params[coefficient_start], "crystal field coefficient filename");
      spherical_coefficients = read_crystal_field_coefficients_from_file(
          cf_coefficient_filename, *crystal_field_spin_type);
    } else {
      spherical_coefficients = read_crystal_field_coefficients_from_config(cf_params, coefficient_start);
    }

    auto tesseral_coefficients = convert_spherical_to_tesseral(spherical_coefficients, energy_cutoff_);

    if (debug_is_enabled()) {
      std::cout << "tesseral harmonic coefficients " << n << ":" << std::endl;
      for (auto const &[key, val] : tesseral_coefficients) {
        std::cout << "  " << key.first << " " << key.second << " " << val << std::endl;
      }
    }

    TesseralKeyCoefficientMap crystal_field_terms;
    for (auto const &[lm, B_lm] : tesseral_coefficients) {
      const auto &[l, m] = lm;
      // We don't use the 0,0 coefficients because they are constant energy offsets.
      if (l == 0 && m == 0) {
        continue;
      }

      const auto coefficient = input_energy_unit_conversion_ * stevens_prefactor.at(l) * B_lm
          * jams::tesseral_racah_normalisation_scale_lookup<double>(l, m);
      if (coefficient == 0.0) {
        continue;
      }

      crystal_field_terms[jams::tesseral_key(l, m)] = jams::Real(coefficient);
    }
    if (local_axes.has_axes && !local_axes.has_full_axes && !terms_are_axial(crystal_field_terms)) {
      throw jams::ConfigException(
          cf_params,
          "a single crystal-field axis can only be used when all non-zero tesseral terms have m = 0");
    }

    for (auto i = 0u; i < size; i++) {
      if (motif_position >= 0) {
        if (int(globals::lattice->lattice_site_basis_index(i)) != motif_position) {
          continue;
        }
      } else {
        if (globals::lattice->lattice_site_material_id(i) != material_id) {
          continue;
        }
      }

      if (spin_has_crystal_field[i]) {
        throw std::runtime_error("crystal field is specified more than once for atom " + std::to_string(i));
      }

      write_local_axes_for_spin(int(i), local_axes);
      spin_terms[i] = crystal_field_terms;
      spin_has_crystal_field[i] = true;
    }
  }

  set_tesseral_terms(spin_terms);
}

CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap CrystalFieldHamiltonian::read_crystal_field_coefficients_from_file(
    const std::string& filename,
    const CrystalFieldSpinType spin_type) {
  std::ifstream fs(filename);
  if (fs.fail()) {
    throw jams::FileException(filename, "failed to open file");
  }

  auto coefficients = zero_spherical_coefficients();

  int line_number = 0;
  for (std::string line; getline(fs, line);) {
    ++line_number;
    if (string_is_comment(line)) {
      continue;
    }

    std::stringstream is(line);

    int l, m;
    double Re_Blm_up, Im_Blm_up, Re_Blm_down, Im_Blm_down;

    if (!(is >> l >> m >> Re_Blm_up >> Im_Blm_up >> Re_Blm_down >> Im_Blm_down)) {
      throw jams::FileException(filename, "line ", line_number, ": expected columns l m upRe upIm dnRe dnIm");
    }

    if (!is_valid_crystal_field_l(l)) {
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

    const auto &[Re_Blm, Im_Blm] = (spin_type == CrystalFieldSpinType::kSpinUp)
                                   ? std::tie(Re_Blm_up, Im_Blm_up) : std::tie(Re_Blm_down, Im_Blm_down);

    coefficients.at({l, m}) = {Re_Blm, Im_Blm};

  }

  return coefficients;
}

CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap CrystalFieldHamiltonian::read_crystal_field_coefficients_from_config(
    const libconfig::Setting& cf_params,
    const int coefficient_start_index)
{
  auto coefficients = zero_spherical_coefficients();
  std::set<std::pair<int, int>> specified_coefficients;

  for (auto i = coefficient_start_index; i < cf_params.getLength(); ++i) {
    const auto& coefficient_setting = cf_params[i];
    if (!coefficient_setting.isList() || coefficient_setting.getLength() != 4) {
      throw jams::ConfigException(
          coefficient_setting,
          "crystal field coefficient must have format (l, m, real, imaginary)");
    }

    const int l = jams::read_integer_setting(coefficient_setting[0], "l");
    const int m = jams::read_integer_setting(coefficient_setting[1], "m");
    const double real = jams::read_numeric_setting<double>(coefficient_setting[2], "real");
    const double imaginary = jams::read_numeric_setting<double>(coefficient_setting[3], "imaginary");

    if (!is_valid_crystal_field_l(l)) {
      throw jams::ConfigException(coefficient_setting[0], "l must be 0, 2, 4 or 6");
    }

    if (m < -l || m > l) {
      throw jams::ConfigException(coefficient_setting[1], "m must be -l <= m <= l");
    }

    const auto lm = std::make_pair(l, m);
    if (!specified_coefficients.insert(lm).second) {
      throw jams::ConfigException(coefficient_setting, "crystal field coefficient is specified more than once");
    }

    coefficients.at(lm) = {real, imaginary};
  }

  for (auto l : {2, 4, 6}) {
    for (auto m = 1; m <= l; ++m) {
      const auto positive_m = std::make_pair(l, m);
      const auto negative_m = std::make_pair(l, -m);
      const auto has_positive_m = specified_coefficients.count(positive_m) > 0;
      const auto has_negative_m = specified_coefficients.count(negative_m) > 0;

      if (has_positive_m == has_negative_m) {
        continue;
      }

      if (has_positive_m) {
        coefficients.at(negative_m) = static_cast<double>(parity_sign(m)) * std::conj(coefficients.at(positive_m));
      } else {
        coefficients.at(positive_m) = static_cast<double>(parity_sign(m)) * std::conj(coefficients.at(negative_m));
      }
    }
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

jams::Real CrystalFieldHamiltonian::crystal_field_energy(int i, const jams::Vec<double, 3> &s) {
  return calculate_energy_for_spin(i, s, 0.0);
}
