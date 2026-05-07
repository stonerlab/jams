#include "jams/hamiltonian/anisotropy_polynomial.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/exception.h"
#include "jams/maths/tesseral_harmonics.h"

#include <libconfig.h++>

#include <string>
#include <utility>
#include <vector>

namespace {
using libconfig::Setting;

constexpr jams::Vec<jams::Real, 3> kDefaultU = {1.0, 0.0, 0.0};
constexpr jams::Vec<jams::Real, 3> kDefaultV = {0.0, 1.0, 0.0};
constexpr jams::Vec<jams::Real, 3> kDefaultW = {0.0, 0.0, 1.0};

struct AnisotropyPolynomialSetting {
    int motif_position = -1;
    int material_id = -1;
    jams::Vec<jams::Real, 3> u = kDefaultU;
    jams::Vec<jams::Real, 3> v = kDefaultV;
    jams::Vec<jams::Real, 3> w = kDefaultW;
    std::vector<std::pair<int, jams::Real>> coefficients;
};

bool is_integer_setting(const Setting& setting)
{
    const auto type = setting.getType();
    return type == Setting::TypeInt || type == Setting::TypeInt64;
}

int read_integer(const Setting& setting, const char* name)
{
    if (!is_integer_setting(setting)) {
        throw jams::ConfigException(setting, name, " must be an integer");
    }

    return int(setting);
}

jams::Vec<jams::Real, 3> read_axis(const Setting& setting)
{
    if (!(setting.isArray() || setting.isList()) || setting.getLength() != 3) {
        throw jams::ConfigException(setting, "axis must contain exactly three numeric components");
    }

    for (auto i = 0; i < 3; ++i) {
        if (!setting[i].isNumber()) {
            throw jams::ConfigException(setting[i], "axis component must be numeric");
        }
    }

    jams::Vec<jams::Real, 3> axis = {
        jams::Real(setting[0]),
        jams::Real(setting[1]),
        jams::Real(setting[2])
    };

    const auto length = jams::norm(axis);
    if (::approximately_zero(length, decltype(length)(jams::defaults::lattice_tolerance))) {
        throw jams::ConfigException(setting, "axis must not be zero");
    }

    return axis / length;
}

bool is_axis_setting(const Setting& setting)
{
    if (!(setting.isArray() || setting.isList()) || setting.getLength() != 3) {
        return false;
    }

    for (auto i = 0; i < 3; ++i) {
        if (!setting[i].isNumber()) {
            return false;
        }
    }

    return true;
}

bool axes_are_orthogonal(const jams::Vec<jams::Real, 3>& u,
                         const jams::Vec<jams::Real, 3>& v,
                         const jams::Vec<jams::Real, 3>& w)
{
    const auto tolerance = jams::Real(jams::defaults::lattice_tolerance);
    return ::approximately_zero(jams::dot(u, v), tolerance)
        && ::approximately_zero(jams::dot(v, w), tolerance)
        && ::approximately_zero(jams::dot(w, u), tolerance);
}

bool axes_match(const jams::Vec<jams::Real, 3>& lhs_u,
                const jams::Vec<jams::Real, 3>& lhs_v,
                const jams::Vec<jams::Real, 3>& lhs_w,
                const jams::Vec<jams::Real, 3>& rhs_u,
                const jams::Vec<jams::Real, 3>& rhs_v,
                const jams::Vec<jams::Real, 3>& rhs_w)
{
    const auto tolerance = jams::Real(jams::defaults::lattice_tolerance);
    return jams::approximately_equal(lhs_u, rhs_u, tolerance)
        && jams::approximately_equal(lhs_v, rhs_v, tolerance)
        && jams::approximately_equal(lhs_w, rhs_w, tolerance);
}

bool is_coefficient_setting(const Setting& setting)
{
    if (!(setting.isArray() || setting.isList()) || setting.getLength() != 3) {
        return false;
    }

    if (!is_integer_setting(setting[0]) || !is_integer_setting(setting[1]) || !setting[2].isNumber()) {
        return false;
    }

    return jams::valid_tesseral_lm(int(setting[0]), int(setting[1]));
}

std::pair<int, jams::Real> read_coefficient_setting(const Setting& setting, const double energy_unit_conversion)
{
    if (!(setting.isArray() || setting.isList()) || setting.getLength() != 3) {
        throw jams::ConfigException(setting, "coefficient must be a list containing l, m and coefficient");
    }

    const auto l = read_integer(setting[0], "l");
    const auto m = read_integer(setting[1], "m");
    if (!jams::valid_tesseral_lm(l, m)) {
        throw jams::ConfigException(setting, "l and m must satisfy l = 2, 4 or 6 and -l <= m <= l");
    }

    if (!setting[2].isNumber()) {
        throw jams::ConfigException(setting[2], "anisotropy coefficient must be numeric");
    }

    return {jams::tesseral_key(l, m), jams::Real(setting[2]) * jams::Real(energy_unit_conversion)};
}

void read_coefficient_settings(std::vector<std::pair<int, jams::Real>>& coefficients,
                               const Setting& setting,
                               const double energy_unit_conversion)
{
    if ((setting.isArray() || setting.isList()) && setting.getLength() > 0 && (setting[0].isArray() || setting[0].isList())) {
        for (auto i = 0; i < setting.getLength(); ++i) {
            coefficients.push_back(read_coefficient_setting(setting[i], energy_unit_conversion));
        }
        return;
    }

    coefficients.push_back(read_coefficient_setting(setting, energy_unit_conversion));
}

AnisotropyPolynomialSetting read_anisotropy_setting(const Setting& setting, const double energy_unit_conversion)
{
    if (!setting.isList()) {
        throw jams::ConfigException(setting, "anisotropy must be a list");
    }

    const auto length = setting.getLength();
    if (length < 2) {
        throw jams::ConfigException(setting, "anisotropy must contain a target and at least one coefficient");
    }

    AnisotropyPolynomialSetting result;

    if (is_integer_setting(setting[0])) {
        result.motif_position = read_integer(setting[0], "unit cell position") - 1;
        if (result.motif_position < 0 || result.motif_position >= globals::lattice->num_basis_sites()) {
            throw jams::ConfigException(setting[0],
                                        "unit cell position must be between 1 and ",
                                        globals::lattice->num_basis_sites());
        }
    } else if (setting[0].isString()) {
        const std::string material = setting[0].c_str();
        if (!globals::lattice->material_exists(material)) {
            throw jams::ConfigException(setting[0], "material ", material, " does not exist in config file");
        }
        result.material_id = globals::lattice->material_index(material);
    } else {
        throw jams::ConfigException(setting[0], "must be a unit cell position or material name");
    }

    auto coefficient_start = 1;
    const auto has_some_axis_settings = length > 1 && is_axis_setting(setting[1]) && !is_coefficient_setting(setting[1]);
    if (has_some_axis_settings) {
        if (length < 5 || !is_axis_setting(setting[2]) || !is_axis_setting(setting[3])) {
            throw jams::ConfigException(setting, "anisotropy must specify all three axes or no axes");
        }

        result.u = read_axis(setting[1]);
        result.v = read_axis(setting[2]);
        result.w = read_axis(setting[3]);

        if (!axes_are_orthogonal(result.u, result.v, result.w)) {
            throw jams::ConfigException(setting, "u, v and w axes must be orthogonal");
        }

        coefficient_start = 4;
    }

    for (auto i = coefficient_start; i < length; ++i) {
        read_coefficient_settings(result.coefficients, setting[i], energy_unit_conversion);
    }

    if (result.coefficients.empty()) {
        throw jams::ConfigException(setting, "anisotropy must contain at least one coefficient");
    }

    return result;
}

bool applies_to_spin(const AnisotropyPolynomialSetting& setting, const int spin_index)
{
    if (setting.motif_position >= 0) {
        return int(globals::lattice->lattice_site_basis_index(spin_index)) == setting.motif_position;
    }

    return globals::lattice->lattice_site_material_id(spin_index) == setting.material_id;
}

jams::Vec<jams::Real, 3> axis_for_spin(const jams::MultiArray<jams::Real, 2>& axes, const int spin_index)
{
    return {axes(spin_index, 0), axes(spin_index, 1), axes(spin_index, 2)};
}

void write_axes_for_spin(jams::MultiArray<jams::Real, 2>& u_axes,
                         jams::MultiArray<jams::Real, 2>& v_axes,
                         jams::MultiArray<jams::Real, 2>& w_axes,
                         const int spin_index,
                         const AnisotropyPolynomialSetting& setting)
{
    for (auto j = 0; j < 3; ++j) {
        u_axes(spin_index, j) = setting.u[j];
        v_axes(spin_index, j) = setting.v[j];
        w_axes(spin_index, j) = setting.w[j];
    }
}

} // namespace

AnisotropyPolynomialHamiltonian::AnisotropyPolynomialHamiltonian(const libconfig::Setting& settings,
    const unsigned int size)
    : Hamiltonian(settings, size)
{
    if (!settings.exists("anisotropies")) {
        throw jams::ConfigException(settings, "missing anisotropies");
    }

    const auto& anisotropy_settings = settings["anisotropies"];
    if (!anisotropy_settings.isList()) {
        throw jams::ConfigException(anisotropy_settings, "anisotropies must be a list");
    }

    zero(spin_pointer_.resize(size + 1));
    zero(u_axes_.resize(size, 3));
    zero(v_axes_.resize(size, 3));
    zero(w_axes_.resize(size, 3));

    for (auto i = 0u; i < size; ++i) {
        u_axes_(i, 0) = kDefaultU[0];
        u_axes_(i, 1) = kDefaultU[1];
        u_axes_(i, 2) = kDefaultU[2];
        v_axes_(i, 0) = kDefaultV[0];
        v_axes_(i, 1) = kDefaultV[1];
        v_axes_(i, 2) = kDefaultV[2];
        w_axes_(i, 0) = kDefaultW[0];
        w_axes_(i, 1) = kDefaultW[1];
        w_axes_(i, 2) = kDefaultW[2];
    }

    std::vector<TesseralKeyCoefficientMap> spin_coefficients(size);
    std::vector<bool> spin_axes_set(size, false);

    for (auto n = 0; n < anisotropy_settings.getLength(); ++n) {
        const auto& anisotropy_setting = anisotropy_settings[n];
        const auto anisotropy = read_anisotropy_setting(anisotropy_setting, input_energy_unit_conversion_);

        for (auto i = 0u; i < size; ++i) {
            if (!applies_to_spin(anisotropy, int(i))) {
                continue;
            }

            if (spin_axes_set[i]) {
                const auto existing_u = axis_for_spin(u_axes_, int(i));
                const auto existing_v = axis_for_spin(v_axes_, int(i));
                const auto existing_w = axis_for_spin(w_axes_, int(i));
                if (!axes_match(existing_u, existing_v, existing_w, anisotropy.u, anisotropy.v, anisotropy.w)) {
                    throw jams::ConfigException(anisotropy_setting,
                                                "anisotropy axes are specified inconsistently for spin ",
                                                i);
                }
            } else {
                write_axes_for_spin(u_axes_, v_axes_, w_axes_, int(i), anisotropy);
                spin_axes_set[i] = true;
            }

            for (const auto& [key, coefficient] : anisotropy.coefficients) {
                spin_coefficients[i][key] += coefficient;
            }
        }
    }

    auto total_terms = 0;
    for (auto i = 0u; i < size; ++i) {
        spin_pointer_(i) = total_terms;
        total_terms += int(spin_coefficients[i].size());
    }
    spin_pointer_(size) = total_terms;

    tesseral_keys_.resize(total_terms);
    tesseral_coefficients_.resize(total_terms);

    auto term_index = 0;
    for (const auto& coefficients : spin_coefficients) {
        for (const auto& [key, coefficient] : coefficients) {
            tesseral_keys_(term_index) = key;
            tesseral_coefficients_(term_index) = coefficient;
            ++term_index;
        }
    }
}

jams::Vec<jams::Real, 3> AnisotropyPolynomialHamiltonian::calculate_field(int i, jams::Real time)
{
    jams::Vec<jams::Real, 3> field = {0.0, 0.0, 0.0};

    const jams::Vec<jams::Real, 3> s_global = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    const auto u = axis_for_spin(u_axes_, i);
    const auto v = axis_for_spin(v_axes_, i);
    const auto w = axis_for_spin(w_axes_, i);
    const jams::Vec<jams::Real, 3> s = {jams::dot(s_global, u), jams::dot(s_global, v), jams::dot(s_global, w)};

    for (auto n = spin_pointer_(i); n < spin_pointer_(i + 1); ++n)
    {
        const auto key = tesseral_keys_(n);
        const auto coeff = tesseral_coefficients_(n);
        const auto h = jams::array_cast<jams::Real>(jams::tesseral_monic_polynomial_grad_key_lookup(key, s[0], s[1], s[2]));
        field += coeff * (h[0] * u + h[1] * v + h[2] * w);
    }

    return field;
}

jams::Real AnisotropyPolynomialHamiltonian::calculate_energy(int i, jams::Real time)
{
    jams::Real energy = 0.0;

    const jams::Vec<jams::Real, 3> s_global = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    const auto u = axis_for_spin(u_axes_, i);
    const auto v = axis_for_spin(v_axes_, i);
    const auto w = axis_for_spin(w_axes_, i);
    const jams::Vec<jams::Real, 3> s = {jams::dot(s_global, u), jams::dot(s_global, v), jams::dot(s_global, w)};

    for (auto n = spin_pointer_(i); n < spin_pointer_(i + 1); ++n) {
        const auto key = tesseral_keys_(n);
        const auto coeff = tesseral_coefficients_(n);
        energy += coeff * jams::tesseral_monic_polynomial_key_lookup(key, s[0], s[1], s[2]);
    }
    return energy;
}
