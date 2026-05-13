#ifndef JAMS_TEST_TESSERAL_HARMONICS_H
#define JAMS_TEST_TESSERAL_HARMONICS_H

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "jams/maths/tesseral_harmonics.h"

namespace {

struct SphericalPolynomialTerm {
  double coefficient;
  int x_power;
  int y_power;
  int z_power;
};

struct SphericalDerivatives {
  double dr = 0.0;
  double dtheta = 0.0;
  double dphi = 0.0;
};

double fraction(const int numerator, const int denominator)
{
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

double pow_int(const double x, const int power)
{
  double result = 1.0;
  for (auto i = 0; i < power; ++i) {
    result *= x;
  }
  return result;
}

std::vector<SphericalPolynomialTerm> spherical_polynomial_terms(const int l, const int m)
{
  switch (jams::tesseral_key(l, m)) {
    case jams::tesseral_key(2, -2): return {{1.0, 1, 1, 0}};
    case jams::tesseral_key(2, -1): return {{1.0, 0, 1, 1}};
    case jams::tesseral_key(2,  0): return {{fraction(-1, 3), 0, 0, 0}, {1.0, 0, 0, 2}};
    case jams::tesseral_key(2,  1): return {{1.0, 1, 0, 1}};
    case jams::tesseral_key(2,  2): return {{1.0, 2, 0, 0}, {-1.0, 0, 2, 0}};

    case jams::tesseral_key(4, -4): return {{fraction(-1, 2), 1, 1, 0}, {1.0, 1, 3, 0}, {fraction(1, 2), 1, 1, 2}};
    case jams::tesseral_key(4, -3): return {{1.0, 2, 1, 1}, {fraction(-1, 3), 0, 3, 1}};
    case jams::tesseral_key(4, -2): return {{fraction(-1, 7), 1, 1, 0}, {1.0, 1, 1, 2}};
    case jams::tesseral_key(4, -1): return {{fraction(-3, 7), 0, 1, 1}, {1.0, 0, 1, 3}};
    case jams::tesseral_key(4,  0): return {{fraction(3, 35), 0, 0, 0}, {fraction(-6, 7), 0, 0, 2}, {1.0, 0, 0, 4}};
    case jams::tesseral_key(4,  1): return {{fraction(-3, 7), 1, 0, 1}, {1.0, 1, 0, 3}};
    case jams::tesseral_key(4,  2): return {{fraction(1, 14), 0, 0, 0}, {fraction(-1, 7), 0, 2, 0}, {fraction(-4, 7), 0, 0, 2}, {1.0, 0, 2, 2}, {fraction(1, 2), 0, 0, 4}};
    case jams::tesseral_key(4,  3): return {{fraction(-1, 4), 1, 0, 1}, {1.0, 1, 2, 1}, {fraction(1, 4), 1, 0, 3}};
    case jams::tesseral_key(4,  4): return {{1.0, 4, 0, 0}, {-6.0, 2, 2, 0}, {1.0, 0, 4, 0}};

    case jams::tesseral_key(6, -6): return {{1.0, 5, 1, 0}, {fraction(-10, 3), 3, 3, 0}, {1.0, 1, 5, 0}};
    case jams::tesseral_key(6, -5): return {{1.0, 4, 1, 1}, {-2.0, 2, 3, 1}, {fraction(1, 5), 0, 5, 1}};
    case jams::tesseral_key(6, -4): return {{fraction(1, 22), 1, 1, 0}, {fraction(-1, 11), 1, 3, 0}, {fraction(-6, 11), 1, 1, 2}, {1.0, 1, 3, 2}, {fraction(1, 2), 1, 1, 4}};
    case jams::tesseral_key(6, -3): return {{fraction(9, 44), 0, 1, 1}, {fraction(-3, 11), 0, 3, 1}, {fraction(-21, 22), 0, 1, 3}, {1.0, 0, 3, 3}, {fraction(3, 4), 0, 1, 5}};
    case jams::tesseral_key(6, -2): return {{fraction(1, 33), 1, 1, 0}, {fraction(-6, 11), 1, 1, 2}, {1.0, 1, 1, 4}};
    case jams::tesseral_key(6, -1): return {{fraction(5, 33), 0, 1, 1}, {fraction(-10, 11), 0, 1, 3}, {1.0, 0, 1, 5}};
    case jams::tesseral_key(6,  0): return {{fraction(-5, 231), 0, 0, 0}, {fraction(5, 11), 0, 0, 2}, {fraction(-15, 11), 0, 0, 4}, {1.0, 0, 0, 6}};
    case jams::tesseral_key(6,  1): return {{fraction(5, 33), 1, 0, 1}, {fraction(-10, 11), 1, 0, 3}, {1.0, 1, 0, 5}};
    case jams::tesseral_key(6,  2): return {{fraction(-1, 66), 0, 0, 0}, {fraction(1, 33), 0, 2, 0}, {fraction(19, 66), 0, 0, 2}, {fraction(-6, 11), 0, 2, 2}, {fraction(-17, 22), 0, 0, 4}, {1.0, 0, 2, 4}, {fraction(1, 2), 0, 0, 6}};
    case jams::tesseral_key(6,  3): return {{fraction(3, 44), 1, 0, 1}, {fraction(-3, 11), 1, 2, 1}, {fraction(-7, 22), 1, 0, 3}, {1.0, 1, 2, 3}, {fraction(1, 4), 1, 0, 5}};
    case jams::tesseral_key(6,  4): return {{fraction(-1, 11), 4, 0, 0}, {fraction(6, 11), 2, 2, 0}, {fraction(-1, 11), 0, 4, 0}, {1.0, 4, 0, 2}, {-6.0, 2, 2, 2}, {1.0, 0, 4, 2}};
    case jams::tesseral_key(6,  5): return {{1.0, 5, 0, 1}, {-10.0, 3, 2, 1}, {5.0, 1, 4, 1}};
    case jams::tesseral_key(6,  6): return {{1.0, 6, 0, 0}, {-15.0, 4, 2, 0}, {15.0, 2, 4, 0}, {-1.0, 0, 6, 0}};
    default: throw std::invalid_argument("unsupported tesseral harmonic");
  }
}

double spherical_term_value(const SphericalPolynomialTerm& term,
                            const double r,
                            const double theta,
                            const double phi)
{
  const int radial_power = term.x_power + term.y_power + term.z_power;
  const int sin_theta_power = term.x_power + term.y_power;
  const int cos_theta_power = term.z_power;
  const int cos_phi_power = term.x_power;
  const int sin_phi_power = term.y_power;

  return term.coefficient
      * pow_int(r, radial_power)
      * pow_int(std::sin(theta), sin_theta_power)
      * pow_int(std::cos(theta), cos_theta_power)
      * pow_int(std::cos(phi), cos_phi_power)
      * pow_int(std::sin(phi), sin_phi_power);
}

double tesseral_monic_polynomial_spherical(const int l, const int m, const double r, const double theta, const double phi)
{
  double value = 0.0;
  for (const auto& term : spherical_polynomial_terms(l, m)) {
    value += spherical_term_value(term, r, theta, phi);
  }
  return value;
}

SphericalDerivatives spherical_term_derivatives(const SphericalPolynomialTerm& term,
                                                const double r,
                                                const double theta,
                                                const double phi)
{
  const int radial_power = term.x_power + term.y_power + term.z_power;
  const int sin_theta_power = term.x_power + term.y_power;
  const int cos_theta_power = term.z_power;
  const int cos_phi_power = term.x_power;
  const int sin_phi_power = term.y_power;

  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double sin_phi = std::sin(phi);
  const double cos_phi = std::cos(phi);

  SphericalDerivatives derivatives;
  if (radial_power > 0) {
    derivatives.dr = term.coefficient
        * radial_power
        * pow_int(r, radial_power - 1)
        * pow_int(sin_theta, sin_theta_power)
        * pow_int(cos_theta, cos_theta_power)
        * pow_int(cos_phi, cos_phi_power)
        * pow_int(sin_phi, sin_phi_power);
  }

  const double phi_factor = pow_int(cos_phi, cos_phi_power) * pow_int(sin_phi, sin_phi_power);
  if (sin_theta_power > 0) {
    derivatives.dtheta += term.coefficient
        * pow_int(r, radial_power)
        * sin_theta_power
        * pow_int(sin_theta, sin_theta_power - 1)
        * cos_theta
        * pow_int(cos_theta, cos_theta_power)
        * phi_factor;
  }
  if (cos_theta_power > 0) {
    derivatives.dtheta += term.coefficient
        * pow_int(r, radial_power)
        * pow_int(sin_theta, sin_theta_power)
        * cos_theta_power
        * pow_int(cos_theta, cos_theta_power - 1)
        * (-sin_theta)
        * phi_factor;
  }

  const double radial_theta_factor = pow_int(r, radial_power)
      * pow_int(sin_theta, sin_theta_power)
      * pow_int(cos_theta, cos_theta_power);
  if (cos_phi_power > 0) {
    derivatives.dphi += term.coefficient
        * radial_theta_factor
        * cos_phi_power
        * pow_int(cos_phi, cos_phi_power - 1)
        * (-sin_phi)
        * pow_int(sin_phi, sin_phi_power);
  }
  if (sin_phi_power > 0) {
    derivatives.dphi += term.coefficient
        * radial_theta_factor
        * pow_int(cos_phi, cos_phi_power)
        * sin_phi_power
        * pow_int(sin_phi, sin_phi_power - 1)
        * cos_phi;
  }

  return derivatives;
}

SphericalDerivatives tesseral_monic_polynomial_spherical_derivatives(
    const int l, const int m, const double r, const double theta, const double phi)
{
  SphericalDerivatives result;
  for (const auto& term : spherical_polynomial_terms(l, m)) {
    const auto derivatives = spherical_term_derivatives(term, r, theta, phi);
    result.dr += derivatives.dr;
    result.dtheta += derivatives.dtheta;
    result.dphi += derivatives.dphi;
  }
  return result;
}

std::array<double, 3> spherical_gradient_to_cartesian(const SphericalDerivatives& derivatives,
                                                      const double r,
                                                      const double theta,
                                                      const double phi)
{
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double sin_phi = std::sin(phi);
  const double cos_phi = std::cos(phi);

  const std::array<double, 3> e_r = {
      sin_theta * cos_phi,
      sin_theta * sin_phi,
      cos_theta
  };
  const std::array<double, 3> e_theta = {
      cos_theta * cos_phi,
      cos_theta * sin_phi,
      -sin_theta
  };
  const std::array<double, 3> e_phi = {
      -sin_phi,
      cos_phi,
      0.0
  };

  std::array<double, 3> gradient = {};
  for (auto i = 0; i < 3; ++i) {
    gradient[i] = derivatives.dr * e_r[i]
        + (derivatives.dtheta / r) * e_theta[i]
        + (derivatives.dphi / (r * sin_theta)) * e_phi[i];
  }

  return gradient;
}

std::vector<std::pair<int, int>> supported_tesseral_harmonics()
{
  std::vector<std::pair<int, int>> lm;
  for (const auto l : {2, 4, 6}) {
    for (auto m = -l; m <= l; ++m) {
      lm.emplace_back(l, m);
    }
  }
  return lm;
}

double condon_shortley_scale_squared_times_pi(const int l, const int m)
{
  switch (jams::tesseral_key(l, m)) {
    case jams::tesseral_key(2, -2): return fraction(15, 4);
    case jams::tesseral_key(2, -1): return fraction(15, 4);
    case jams::tesseral_key(2,  0): return fraction(45, 16);
    case jams::tesseral_key(2,  1): return fraction(15, 4);
    case jams::tesseral_key(2,  2): return fraction(15, 16);

    case jams::tesseral_key(4, -4): return fraction(315, 4);
    case jams::tesseral_key(4, -3): return fraction(2835, 32);
    case jams::tesseral_key(4, -2): return fraction(2205, 16);
    case jams::tesseral_key(4, -1): return fraction(2205, 32);
    case jams::tesseral_key(4,  0): return fraction(11025, 256);
    case jams::tesseral_key(4,  1): return fraction(2205, 32);
    case jams::tesseral_key(4,  2): return fraction(2205, 16);
    case jams::tesseral_key(4,  3): return fraction(315, 2);
    case jams::tesseral_key(4,  4): return fraction(315, 256);

    case jams::tesseral_key(6, -6): return fraction(27027, 512);
    case jams::tesseral_key(6, -5): return fraction(225225, 512);
    case jams::tesseral_key(6, -4): return fraction(99099, 16);
    case jams::tesseral_key(6, -3): return fraction(165165, 32);
    case jams::tesseral_key(6, -2): return fraction(1486485, 512);
    case jams::tesseral_key(6, -1): return fraction(297297, 256);
    case jams::tesseral_key(6,  0): return fraction(693693, 1024);
    case jams::tesseral_key(6,  1): return fraction(297297, 256);
    case jams::tesseral_key(6,  2): return fraction(1486485, 512);
    case jams::tesseral_key(6,  3): return fraction(165165, 32);
    case jams::tesseral_key(6,  4): return fraction(99099, 1024);
    case jams::tesseral_key(6,  5): return fraction(9009, 512);
    case jams::tesseral_key(6,  6): return fraction(3003, 2048);
    default: throw std::invalid_argument("unsupported tesseral harmonic");
  }
}

double stevens_scale(const int l, const int m)
{
  switch (jams::tesseral_key(l, m)) {
    case jams::tesseral_key(2, -2): return 2.0;
    case jams::tesseral_key(2, -1): return 1.0;
    case jams::tesseral_key(2,  0): return 3.0;
    case jams::tesseral_key(2,  1): return 1.0;
    case jams::tesseral_key(2,  2): return 1.0;

    case jams::tesseral_key(4, -4): return -8.0;
    case jams::tesseral_key(4, -3): return 3.0;
    case jams::tesseral_key(4, -2): return 14.0;
    case jams::tesseral_key(4, -1): return 7.0;
    case jams::tesseral_key(4,  0): return 35.0;
    case jams::tesseral_key(4,  1): return 7.0;
    case jams::tesseral_key(4,  2): return -14.0;
    case jams::tesseral_key(4,  3): return -4.0;
    case jams::tesseral_key(4,  4): return 1.0;

    case jams::tesseral_key(6, -6): return 6.0;
    case jams::tesseral_key(6, -5): return 5.0;
    case jams::tesseral_key(6, -4): return -88.0;
    case jams::tesseral_key(6, -3): return -44.0;
    case jams::tesseral_key(6, -2): return 66.0;
    case jams::tesseral_key(6, -1): return 33.0;
    case jams::tesseral_key(6,  0): return 231.0;
    case jams::tesseral_key(6,  1): return 33.0;
    case jams::tesseral_key(6,  2): return -66.0;
    case jams::tesseral_key(6,  3): return -44.0;
    case jams::tesseral_key(6,  4): return 11.0;
    case jams::tesseral_key(6,  5): return 1.0;
    case jams::tesseral_key(6,  6): return 1.0;
    default: throw std::invalid_argument("unsupported tesseral harmonic");
  }
}

} // namespace

TEST(TesseralHarmonicsTest, CartesianFunctionsAndGradientsMatchSphericalAtUnitRadius)
{
  constexpr double r = 1.0;
  constexpr double tolerance = 1e-8;
  const std::array<double, 3> theta_values = {0.37, 1.19, 2.31};
  const std::array<double, 4> phi_values = {-2.4, -0.6, 0.8, 2.2};

  for (const auto& [l, m] : supported_tesseral_harmonics()) {
    for (const auto theta : theta_values) {
      for (const auto phi : phi_values) {
        const double sin_theta = std::sin(theta);
        const double x = sin_theta * std::cos(phi);
        const double y = sin_theta * std::sin(phi);
        const double z = std::cos(theta);

        const auto key = jams::tesseral_key(l, m);
        const auto cartesian_value = jams::tesseral_monic_polynomial_key_lookup(key, x, y, z);
        const auto spherical_value = tesseral_monic_polynomial_spherical(l, m, r, theta, phi);
        EXPECT_NEAR(cartesian_value, spherical_value, tolerance)
            << "l=" << l << " m=" << m << " theta=" << theta << " phi=" << phi;

        double cartesian_gradient[3];
        jams::tesseral_monic_polynomial_grad_key_lookup(key, x, y, z, cartesian_gradient);
        const auto spherical_derivatives = tesseral_monic_polynomial_spherical_derivatives(
            l, m, r, theta, phi);
        const auto spherical_gradient = spherical_gradient_to_cartesian(
            spherical_derivatives, r, theta, phi);

        for (auto i = 0; i < 3; ++i) {
          EXPECT_NEAR(cartesian_gradient[i], spherical_gradient[i], tolerance)
              << "l=" << l << " m=" << m << " theta=" << theta << " phi=" << phi << " component=" << i;
        }
      }
    }
  }
}

TEST(TesseralHarmonicsTest, NormalisationScaleFactors)
{
  constexpr double relative_tolerance = 1e-14;

  for (const auto& [l, m] : supported_tesseral_harmonics()) {
    EXPECT_DOUBLE_EQ(
        jams::tesseral_monic_polynomial_normalisation_scale<double>(
            jams::TesseralHarmonicNormalisation::monic, l, m),
        1.0);

    const auto scale = jams::tesseral_monic_polynomial_normalisation_scale<double>(
        jams::TesseralHarmonicNormalisation::condon_shortley, l, m);
    const auto expected = condon_shortley_scale_squared_times_pi(l, m);
    EXPECT_NEAR(scale * scale * M_PI,
                expected,
                relative_tolerance * std::max(1.0, std::abs(expected)))
        << "l=" << l << " m=" << m;

    const auto racah_scale = jams::tesseral_monic_polynomial_normalisation_scale<double>(
        jams::TesseralHarmonicNormalisation::racah, l, m);
    EXPECT_DOUBLE_EQ(racah_scale, jams::tesseral_racah_normalisation_scale_lookup<double>(l, m));
    const auto expected_racah_scale_squared =
        4.0 * condon_shortley_scale_squared_times_pi(l, m) / static_cast<double>(2 * l + 1);
    EXPECT_NEAR(racah_scale * racah_scale,
                expected_racah_scale_squared,
                relative_tolerance * std::max(1.0, std::abs(expected_racah_scale_squared)))
        << "l=" << l << " m=" << m;

    EXPECT_DOUBLE_EQ(
        jams::tesseral_monic_polynomial_normalisation_scale<double>(
            jams::TesseralHarmonicNormalisation::stevens, l, m),
        stevens_scale(l, m));
  }

  constexpr double z = 0.42;
  const auto racah_l2_m0 = jams::tesseral_monic_polynomial_normalisation_scale<double>(
      jams::TesseralHarmonicNormalisation::racah, 2, 0)
      * jams::tesseral_monic_polynomial_l2_0(0.0, 0.0, z);
  EXPECT_NEAR(racah_l2_m0, 0.5 * (3.0 * z * z - 1.0), 1e-15);

  const auto racah_l4_m0 = jams::tesseral_monic_polynomial_normalisation_scale<double>(
      jams::TesseralHarmonicNormalisation::racah, 4, 0)
      * jams::tesseral_monic_polynomial_l4_0(0.0, 0.0, z);
  EXPECT_NEAR(racah_l4_m0,
              0.125 * (3.0 - 30.0 * z * z + 35.0 * z * z * z * z),
              1e-15);

  const auto z2 = z * z;
  const auto z4 = z2 * z2;
  const auto z6 = z4 * z2;
  const auto racah_l6_m0 = jams::tesseral_monic_polynomial_normalisation_scale<double>(
      jams::TesseralHarmonicNormalisation::racah, 6, 0)
      * jams::tesseral_monic_polynomial_l6_0(0.0, 0.0, z);
  EXPECT_NEAR(racah_l6_m0, (231.0 * z6 - 315.0 * z4 + 105.0 * z2 - 5.0) / 16.0, 1e-15);

  const auto stevens_l2_m0 = jams::tesseral_monic_polynomial_normalisation_scale<double>(
      jams::TesseralHarmonicNormalisation::stevens, 2, 0)
      * jams::tesseral_monic_polynomial_l2_0(0.0, 0.0, z);
  EXPECT_NEAR(stevens_l2_m0, 3.0 * z * z - 1.0, 1e-15);
}

#endif // JAMS_TEST_TESSERAL_HARMONICS_H
