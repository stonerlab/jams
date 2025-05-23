#include <jams/hamiltonian/crystal_field.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/helpers/exception.h>
#include <fstream>
#include <iostream>

CrystalFieldHamiltonian::CrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size)
    : Hamiltonian(settings, size),
      energy_cutoff_(jams::config_required<double>(settings, "energy_cutoff")),
      crystal_field_spin_type_(CrystalFieldSpinType::kSpinUp),
      spin_has_crystal_field_(false, size),
      crystal_field_tesseral_coeff_(0.0, kCrystalFieldNumCoeff_, size) {
  std::cout << "energy_cutoff: " << energy_cutoff_ << "\n";

  auto spin_type_string = lowercase(settings["crystal_field_spin_type"]);

  if (spin_type_string == "up") {
    crystal_field_spin_type_ = CrystalFieldSpinType::kSpinUp;
    std::cout << "crystal_field_spin_type: up\n";
  } else if (spin_type_string == "down") {
    crystal_field_spin_type_ = CrystalFieldSpinType::kSpinUp;
    std::cout << "crystal_field_spin_type: down\n";
  } else {
    throw jams::ConfigException(settings["crystal_field_spin_type"], "must be 'up' or 'down'");
  }

  for (auto n = 0; n < settings["crystal_field_coefficients"].getLength(); ++n) {

    auto &cf_params = settings["crystal_field_coefficients"][n];

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

    for (auto i = 0; i < globals::num_spins; i++) {
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

      if (spin_has_crystal_field_(i)) {
        throw std::runtime_error("crystal field is specified more than once for atom " + std::to_string(i));
      }

      auto cf_counter = 0;
      for (auto const &[lm, B_lm] : tesseral_coefficients) {
        const auto &[l, m] = lm;
        // We don't use the 0,0 coefficients because they are constant energy offsets
        if (l == 0 && m == 0) {
          continue;
        }

        crystal_field_tesseral_coeff_(cf_counter++, i) = input_energy_unit_conversion_ * stevens_prefactor[l] * B_lm;
      }

      spin_has_crystal_field_(i) = true;
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
        C_lm =
            kSqrtOne_Two * (spherical_coefficients.at({l, -m}) + std::pow(-1, m) * spherical_coefficients.at({l, m}));
      } else if (m < 0) {
        C_lm = kImagOne * kSqrtOne_Two
            * (spherical_coefficients.at({l, m}) - std::pow(-1, m) * spherical_coefficients.at({l, -m}));
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

double CrystalFieldHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;
  calculate_energies(time);
  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void CrystalFieldHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

void CrystalFieldHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto local_field = calculate_field(i, time);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = local_field[j];
    }
  }
}

Vec3 CrystalFieldHamiltonian::calculate_field(int i, double time) {
  const double sx = globals::s(i, 0);
  const double sy = globals::s(i, 1);
  const double sz = globals::s(i, 2);

  Vec3 h = {0.0, 0.0, 0.0};

// -C_{2,-2} dZ_{2,-2}/dS
  double C2_2 = crystal_field_tesseral_coeff_(0, i);
  h[0] += C2_2*( -1.7320508075688772*sx*(2.*(sy*sy) + sz*sz) );
  h[1] += C2_2*( -1.7320508075688772*sy*(-2. + 2.*(sy*sy) + sz*sz) );
  h[2] += C2_2*( -1.7320508075688772*sz*(-1. + 2.*(sy*sy) + sz*sz) );

// -C_{2,-1} dZ_{2,-1}/dS
  double C2_1 = crystal_field_tesseral_coeff_(1, i);
  h[0] += C2_1*( 1.7320508075688772*sz*(-1. + 2.*(sy*sy) + 2.*(sz*sz)) );
  h[1] += C2_1*( -3.4641016151377544*sx*sy*sz );
  h[2] += C2_1*( -1.7320508075688772*sx*(-1. + 2.*(sz*sz)) );

// -C_{2,0} dZ_{2,0}/dS
  double C20 = crystal_field_tesseral_coeff_(2, i);
  h[0] += C20*( 3.*sx*(sz*sz) );
  h[1] += C20*( 3.*sy*(sz*sz) );
  h[2] += C20*( 3.*sz*(-1. + sz*sz) );

// -C_{2,1} dZ_{2,1}/dS
  double C21 = crystal_field_tesseral_coeff_(3, i);
  h[0] += C21*( 1.7320508075688772*sz*(-1. + 2.*(sy*sy) + 2.*(sz*sz)) );
  h[1] += C21*( -3.4641016151377544*sx*sy*sz );
  h[2] += C21*( -1.7320508075688772*sx*(-1. + 2.*(sz*sz)) );

// -C_{2,2} dZ_{2,2}/dS
  double C22 = crystal_field_tesseral_coeff_(4, i);
  h[0] += C22*( -1.7320508075688772*sx*(2.*(sy*sy) + sz*sz) );
  h[1] += C22*( -1.7320508075688772*sy*(-2. + 2.*(sy*sy) + sz*sz) );
  h[2] += C22*( -1.7320508075688772*sz*(-1. + 2.*(sy*sy) + sz*sz) );

// -C_{4,-4} dZ_{4,-4}/dS
  double C4_4 = crystal_field_tesseral_coeff_(5, i);
  h[0] += C4_4*( 2.958039891549808*sx*(8.*(sy*sy*sy*sy) - 1.*(sz*sz) + sz*sz*sz*sz + sy*sy*(-4. + 8.*(sz*sz))) );
  h[1] += C4_4*( 2.958039891549808*sy*(4. + 8.*(sy*sy*sy*sy) - 5.*(sz*sz) + sz*sz*sz*sz + 4.*(sy*sy)*(-3. + 2.*(sz*sz))) );
  h[2] += C4_4*( 2.958039891549808*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz );

// -C_{4,-3} dZ_{4,-3}/dS
  double C4_3 = crystal_field_tesseral_coeff_(6, i);
  h[0] += C4_3*( -2.091650066335189*sz*(1. + 16.*(sy*sy*sy*sy) - 5.*(sz*sz) + 4.*(sz*sz*sz*sz) + 2.*(sy*sy)*(-7. + 10.*(sz*sz))) );
  h[1] += C4_3*( 4.183300132670378*sx*sy*sz*(-5. + 8.*(sy*sy) + 2.*(sz*sz)) );
  h[2] += C4_3*( 2.091650066335189*sx*(-1. + 4.*(sy*sy) + sz*sz)*(-1. + 4.*(sz*sz)) );

// -C_{4,-2} dZ_{4,-2}/dS
  double C4_2 = crystal_field_tesseral_coeff_(7, i);
  h[0] += C4_2*( -2.23606797749979*sx*(-1.*(sy*sy) + 2.*(-2. + 7.*(sy*sy))*(sz*sz) + 7.*(sz*sz*sz*sz)) );
  h[1] += C4_2*( -2.23606797749979*sy*(1. - 11.*(sz*sz) + 7.*(sz*sz*sz*sz) + sy*sy*(-1. + 14.*(sz*sz))) );
  h[2] += C4_2*( -2.23606797749979*sz*(-1. + 2.*(sy*sy) + sz*sz)*(-4. + 7.*(sz*sz)) );

// -C_{4,-1} dZ_{4,-1}/dS
  double C4_1 = crystal_field_tesseral_coeff_(8, i);
  h[0] += C4_1*( 0.7905694150420949*sz*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz) + sy*sy*(-6. + 28.*(sz*sz))) );
  h[1] += C4_1*( -1.5811388300841898*sx*sy*sz*(-3. + 14.*(sz*sz)) );
  h[2] += C4_1*( -0.7905694150420949*sx*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz)) );

// -C_{4,0} dZ_{4,0}/dS
  double C40 = crystal_field_tesseral_coeff_(9, i);
  h[0] += C40*( -2.5*sx*(sz*sz)*(3. - 7.*(sz*sz)) );
  h[1] += C40*( -2.5*sy*(sz*sz)*(3. - 7.*(sz*sz)) );
  h[2] += C40*( 2.5*sz*(3. - 10.*(sz*sz) + 7.*(sz*sz*sz*sz)) );

// -C_{4,1} dZ_{4,1}/dS
  double C41 = crystal_field_tesseral_coeff_(10, i);
  h[0] += C41*( 0.7905694150420949*sz*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz) + sy*sy*(-6. + 28.*(sz*sz))) );
  h[1] += C41*( -1.5811388300841898*sx*sy*sz*(-3. + 14.*(sz*sz)) );
  h[2] += C41*( -0.7905694150420949*sx*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz)) );

// -C_{4,2} dZ_{4,2}/dS
  double C42 = crystal_field_tesseral_coeff_(11, i);
  h[0] += C42*( -2.23606797749979*sx*(-1.*(sy*sy) + 2.*(-2. + 7.*(sy*sy))*(sz*sz) + 7.*(sz*sz*sz*sz)) );
  h[1] += C42*( -2.23606797749979*sy*(1. - 11.*(sz*sz) + 7.*(sz*sz*sz*sz) + sy*sy*(-1. + 14.*(sz*sz))) );
  h[2] += C42*( -2.23606797749979*sz*(-1. + 2.*(sy*sy) + sz*sz)*(-4. + 7.*(sz*sz)) );

// -C_{4,3} dZ_{4,3}/dS
  double C43 = crystal_field_tesseral_coeff_(12, i);
  h[0] += C43*( -2.091650066335189*sz*(1. + 16.*(sy*sy*sy*sy) - 5.*(sz*sz) + 4.*(sz*sz*sz*sz) + 2.*(sy*sy)*(-7. + 10.*(sz*sz))) );
  h[1] += C43*( 4.183300132670378*sx*sy*sz*(-5. + 8.*(sy*sy) + 2.*(sz*sz)) );
  h[2] += C43*( 2.091650066335189*sx*(-1. + 4.*(sy*sy) + sz*sz)*(-1. + 4.*(sz*sz)) );

// -C_{4,4} dZ_{4,4}/dS
  double C44 = crystal_field_tesseral_coeff_(13, i);
  h[0] += C44*( 2.958039891549808*sx*(8.*(sy*sy*sy*sy) - 1.*(sz*sz) + sz*sz*sz*sz + sy*sy*(-4. + 8.*(sz*sz))) );
  h[1] += C44*( 2.958039891549808*sy*(4. + 8.*(sy*sy*sy*sy) - 5.*(sz*sz) + sz*sz*sz*sz + 4.*(sy*sy)*(-3. + 2.*(sz*sz))) );
  h[2] += C44*( 2.958039891549808*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz );

// -C_{6,-6} dZ_{6,-6}/dS
  double C6_6 = crystal_field_tesseral_coeff_(14, i);
  h[0] += C6_6*( -4.030159736288377*sx*(6.*(sy*sy*sy*sy*sy*sy) + 5.*(sy*sy*sy*sy)*(sz*sz) - 10.*(sx*sx)*(sy*sy)*(2.*(sy*sy) + sz*sz) + sx*sx*sx*sx*(6.*(sy*sy) + sz*sz)) );
  h[1] += C6_6*( 4.030159736288377*sy*(6.*(sx*sx*sx*sx*sx*sx) - 20.*(sx*sx*sx*sx)*(sy*sy) + 6.*(sx*sx)*(sy*sy*sy*sy) + (5.*(sx*sx*sx*sx) - 10.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
  h[2] += C6_6*( 4.030159736288377*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy))*sz );

// -C_{6,-5} dZ_{6,-5}/dS
  double C6_5 = crystal_field_tesseral_coeff_(15, i);
  h[0] += C6_5*( 2.3268138086232857*sz*(-1.*(sx*sx*sx*sx*sx*sx) + 35.*(sx*sx*sx*sx)*(sy*sy) - 55.*(sx*sx)*(sy*sy*sy*sy) + 5.*(sy*sy*sy*sy*sy*sy) + 5.*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
  h[1] += C6_5*( -4.653627617246571*sx*sy*sz*(13. - 56.*(sy*sy) + 48.*(sy*sy*sy*sy) + 4.*(-4. + 9.*(sy*sy))*(sz*sz) + 3.*(sz*sz*sz*sz)) );
  h[2] += C6_5*( -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*(-1. + 6.*(sz*sz)) );

// -C_{6,-4} dZ_{6,-4}/dS
  double C6_4 = crystal_field_tesseral_coeff_(16, i);
  h[0] += C6_4*( 0.9921567416492215*sx*(13.*(sz*sz) - 46.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(1. - 24.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
  h[1] += C6_4*( 0.9921567416492215*sy*(-8. + 109.*(sz*sz) - 134.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(3. - 46.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
  h[2] += C6_4*( -0.9921567416492215*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz*(13. - 33.*(sz*sz)) );

// -C_{6,-3} dZ_{6,-3}/dS
  double C6_3 = crystal_field_tesseral_coeff_(17, i);
  h[0] += C6_3*( -2.7171331399105196*sz*(-1. + 16.*(sz*sz) - 37.*(sz*sz*sz*sz) + 22.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 11.*(sz*sz)) + 2.*(sy*sy)*(7. - 54.*(sz*sz) + 55.*(sz*sz*sz*sz))) );
  h[1] += C6_3*( 5.434266279821039*sx*sy*sz*(5. - 24.*(sz*sz) + 11.*(sz*sz*sz*sz) + sy*sy*(-8. + 44.*(sz*sz))) );
  h[2] += C6_3*( 2.7171331399105196*sx*(-1. + 4.*(sy*sy) + sz*sz)*(1. - 15.*(sz*sz) + 22.*(sz*sz*sz*sz)) );

// -C_{6,-2} dZ_{6,-2}/dS
  double C6_2 = crystal_field_tesseral_coeff_(18, i);
  h[0] += C6_2*( -0.9057110466368399*sx*(2.*(sy*sy) + (19. - 72.*(sy*sy))*(sz*sz) + 6.*(-17. + 33.*(sy*sy))*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz)) );
  h[1] += C6_2*( -0.9057110466368399*sy*(-2. + 55.*(sz*sz) - 168.*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz) + 2.*(sy*sy)*(1. - 36.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
  h[2] += C6_2*( -0.9057110466368399*sz*(-1. + 2.*(sy*sy) + sz*sz)*(19. - 102.*(sz*sz) + 99.*(sz*sz*sz*sz)) );

// -C_{6,-1} dZ_{6,-1}/dS
  double C6_1 = crystal_field_tesseral_coeff_(19, i);
  h[0] += C6_1*( -0.57282196186948*sz*(5. - 100.*(sz*sz) + 285.*(sz*sz*sz*sz) - 198.*(sz*sz*sz*sz*sz*sz) - 2.*(sy*sy)*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
  h[1] += C6_1*( -1.14564392373896*sx*sy*sz*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz)) );
  h[2] += C6_1*( -0.57282196186948*sx*(-5. + 100.*(sz*sz) - 285.*(sz*sz*sz*sz) + 198.*(sz*sz*sz*sz*sz*sz)) );

// -C_{6,0} dZ_{6,0}/dS
  double C60 = crystal_field_tesseral_coeff_(20, i);
  h[0] += C60*( 2.625*sx*(sz*sz)*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz)) );
  h[1] += C60*( 2.625*sy*(sz*sz)*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz)) );
  h[2] += C60*( 2.625*sz*(-5. + 35.*(sz*sz) - 63.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz)) );

// -C_{6,1} dZ_{6,1}/dS
  double C61 = crystal_field_tesseral_coeff_(21, i);
  h[0] += C61*( -0.57282196186948*sz*(5. - 100.*(sz*sz) + 285.*(sz*sz*sz*sz) - 198.*(sz*sz*sz*sz*sz*sz) - 2.*(sy*sy)*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
  h[1] += C61*( -1.14564392373896*sx*sy*sz*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz)) );
  h[2] += C61*( -0.57282196186948*sx*(-5. + 100.*(sz*sz) - 285.*(sz*sz*sz*sz) + 198.*(sz*sz*sz*sz*sz*sz)) );

// -C_{6,2} dZ_{6,2}/dS
  double C62 = crystal_field_tesseral_coeff_(22, i);
  h[0] += C62*( -0.9057110466368399*sx*(2.*(sy*sy) + (19. - 72.*(sy*sy))*(sz*sz) + 6.*(-17. + 33.*(sy*sy))*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz)) );
  h[1] += C62*( -0.9057110466368399*sy*(-2. + 55.*(sz*sz) - 168.*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz) + 2.*(sy*sy)*(1. - 36.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
  h[2] += C62*( -0.9057110466368399*sz*(-1. + 2.*(sy*sy) + sz*sz)*(19. - 102.*(sz*sz) + 99.*(sz*sz*sz*sz)) );

// -C_{6,3} dZ_{6,3}/dS
  double C63 = crystal_field_tesseral_coeff_(23, i);
  h[0] += C63*( -2.7171331399105196*sz*(-1. + 16.*(sz*sz) - 37.*(sz*sz*sz*sz) + 22.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 11.*(sz*sz)) + 2.*(sy*sy)*(7. - 54.*(sz*sz) + 55.*(sz*sz*sz*sz))) );
  h[1] += C63*( 5.434266279821039*sx*sy*sz*(5. - 24.*(sz*sz) + 11.*(sz*sz*sz*sz) + sy*sy*(-8. + 44.*(sz*sz))) );
  h[2] += C63*( 2.7171331399105196*sx*(-1. + 4.*(sy*sy) + sz*sz)*(1. - 15.*(sz*sz) + 22.*(sz*sz*sz*sz)) );

// -C_{6,4} dZ_{6,4}/dS
  double C64 = crystal_field_tesseral_coeff_(24, i);
  h[0] += C64*( 0.9921567416492215*sx*(13.*(sz*sz) - 46.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(1. - 24.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
  h[1] += C64*( 0.9921567416492215*sy*(-8. + 109.*(sz*sz) - 134.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(3. - 46.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
  h[2] += C64*( -0.9921567416492215*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz*(13. - 33.*(sz*sz)) );

// -C_{6,5} dZ_{6,5}/dS
  double C65 = crystal_field_tesseral_coeff_(25, i);
  h[0] += C65*( 2.3268138086232857*sz*(-1.*(sx*sx*sx*sx*sx*sx) + 35.*(sx*sx*sx*sx)*(sy*sy) - 55.*(sx*sx)*(sy*sy*sy*sy) + 5.*(sy*sy*sy*sy*sy*sy) + 5.*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
  h[1] += C65*( -4.653627617246571*sx*sy*sz*(13. - 56.*(sy*sy) + 48.*(sy*sy*sy*sy) + 4.*(-4. + 9.*(sy*sy))*(sz*sz) + 3.*(sz*sz*sz*sz)) );
  h[2] += C65*( -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*(-1. + 6.*(sz*sz)) );

// -C_{6,6} dZ_{6,6}/dS
  double C66 = crystal_field_tesseral_coeff_(26, i);
  h[0] += C66*( -4.030159736288377*sx*(6.*(sy*sy*sy*sy*sy*sy) + 5.*(sy*sy*sy*sy)*(sz*sz) - 10.*(sx*sx)*(sy*sy)*(2.*(sy*sy) + sz*sz) + sx*sx*sx*sx*(6.*(sy*sy) + sz*sz)) );
  h[1] += C66*( 4.030159736288377*sy*(6.*(sx*sx*sx*sx*sx*sx) - 20.*(sx*sx*sx*sx)*(sy*sy) + 6.*(sx*sx)*(sy*sy*sy*sy) + (5.*(sx*sx*sx*sx) - 10.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
  h[2] += C66*( 4.030159736288377*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy))*sz );

  return h;
}

double CrystalFieldHamiltonian::calculate_energy(int i, double time) {
  return crystal_field_energy(i, {globals::s(i,0), globals::s(i,1), globals::s(i,2)});
}

double CrystalFieldHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final,
                                                            double time) {
  return crystal_field_energy(i, spin_final) - crystal_field_energy(i, spin_initial);
}
double CrystalFieldHamiltonian::crystal_field_energy(int i, const Vec3 &s) {
  if (!spin_has_crystal_field_(i)) {
    return 0.0;
  }

  const double sx = s[0];
  const double sy = s[1];
  const double sz = s[2];

  double energy = 0.0;

// C_{2,-2} Z_{2,-2}
  energy += crystal_field_tesseral_coeff_(0, i) * 0.8660254037844386*(sx - 1.*sy)*(sx + sy);

// C_{2,-1} Z_{2,-1}
  energy += crystal_field_tesseral_coeff_(1, i) * -1.7320508075688772*sx*sz;

// C_{2,0} Z_{2,0}
  energy += crystal_field_tesseral_coeff_(2, i) * 0.5*(-1. + 3.*(sz*sz));

// C_{2,1} Z_{2,1}
  energy += crystal_field_tesseral_coeff_(3, i) * -1.7320508075688772*sx*sz;

// C_{2,2} Z_{2,2}
  energy += crystal_field_tesseral_coeff_(4, i) * 0.8660254037844386*(sx - 1.*sy)*(sx + sy);

// C_{4,-4} Z_{4,-4}
  energy += crystal_field_tesseral_coeff_(5, i) * 0.739509972887452*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy);

// C_{4,-3} Z_{4,-3}
  energy += crystal_field_tesseral_coeff_(6, i) * 2.091650066335189*sx*sz*(-1. + 4.*(sy*sy) + sz*sz);

// C_{4,-2} Z_{4,-2}
  energy += crystal_field_tesseral_coeff_(7, i) * -0.5590169943749475*(-1. + 2.*(sy*sy) + sz*sz)*(-1. + 7.*(sz*sz));

// C_{4,-1} Z_{4,-1}
  energy += crystal_field_tesseral_coeff_(8, i) * 0.7905694150420949*sx*sz*(3. - 7.*(sz*sz));

// C_{4,0} Z_{4,0}
  energy += crystal_field_tesseral_coeff_(9, i) * 0.125*(3. - 30.*(sz*sz) + 35.*(sz*sz*sz*sz));

// C_{4,1} Z_{4,1}
  energy += crystal_field_tesseral_coeff_(10, i) * 0.7905694150420949*sx*sz*(3. - 7.*(sz*sz));

// C_{4,2} Z_{4,2}
  energy += crystal_field_tesseral_coeff_(11, i) * -0.5590169943749475*(-1. + 2.*(sy*sy) + sz*sz)*(-1. + 7.*(sz*sz));

// C_{4,3} Z_{4,3}
  energy += crystal_field_tesseral_coeff_(12, i) * 2.091650066335189*sx*sz*(-1. + 4.*(sy*sy) + sz*sz);

// C_{4,4} Z_{4,4}
  energy += crystal_field_tesseral_coeff_(13, i) * 0.739509972887452*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy);

// C_{6,-6} Z_{6,-6}
  energy += crystal_field_tesseral_coeff_(14, i) * 0.6716932893813962*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy));

// C_{6,-5} Z_{6,-5}
  energy += crystal_field_tesseral_coeff_(15, i) * -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*sz;

// C_{6,-4} Z_{6,-4}
  energy += crystal_field_tesseral_coeff_(16, i) * 0.49607837082461076*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(-1. + 11.*(sz*sz));

// C_{6,-3} Z_{6,-3}
  energy += crystal_field_tesseral_coeff_(17, i) * 0.9057110466368399*sx*sz*(-1. + 4.*(sy*sy) + sz*sz)*(-3. + 11.*(sz*sz));

// C_{6,-2} Z_{6,-2}
  energy += crystal_field_tesseral_coeff_(18, i) * -0.45285552331841994*(-1. + 2.*(sy*sy) + sz*sz)*(1. - 18.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,-1} Z_{6,-1}
  energy += crystal_field_tesseral_coeff_(19, i) * -0.57282196186948*sx*sz*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,0} Z_{6,0}
  energy += crystal_field_tesseral_coeff_(20, i) * 0.0625*(-5. + 21.*(sz*sz)*(5. - 15.*(sz*sz) + 11.*(sz*sz*sz*sz)));

// C_{6,1} Z_{6,1}
  energy += crystal_field_tesseral_coeff_(21, i) * -0.57282196186948*sx*sz*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,2} Z_{6,2}
  energy += crystal_field_tesseral_coeff_(22, i) * -0.45285552331841994*(-1. + 2.*(sy*sy) + sz*sz)*(1. - 18.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,3} Z_{6,3}
  energy += crystal_field_tesseral_coeff_(23, i) * 0.9057110466368399*sx*sz*(-1. + 4.*(sy*sy) + sz*sz)*(-3. + 11.*(sz*sz));

// C_{6,4} Z_{6,4}
  energy += crystal_field_tesseral_coeff_(24, i) * 0.49607837082461076*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(-1. + 11.*(sz*sz));

// C_{6,5} Z_{6,5}
  energy += crystal_field_tesseral_coeff_(25, i) * -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*sz;

// C_{6,6} Z_{6,6}
  energy += crystal_field_tesseral_coeff_(26, i) * 0.6716932893813962*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy));
  return energy;
}
