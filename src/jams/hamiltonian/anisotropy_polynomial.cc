#include "jams/hamiltonian/anisotropy_polynomial.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/tesseral_polynomial_evaluator.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/utils.h"
#include "jams/maths/tesseral_harmonics.h"

#include <libconfig.h++>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <stdexcept>
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
    bool has_axes = false;
    bool has_full_axes = false;
    jams::Vec<jams::Real, 3> u = kDefaultU;
    jams::Vec<jams::Real, 3> v = kDefaultV;
    jams::Vec<jams::Real, 3> w = kDefaultW;
    std::vector<std::pair<int, jams::Real>> coefficients;
};

struct AnisotropyProfile {
    jams::Vec<jams::Real, 3> u = kDefaultU;
    jams::Vec<jams::Real, 3> v = kDefaultV;
    jams::Vec<jams::Real, 3> w = kDefaultW;
    int axis_mode = jams::tesseral_polynomial::kProfileAxesDefault;
    std::array<jams::Real, 4> axial_polynomial = {0.0, 0.0, 0.0, 0.0};
    std::vector<std::pair<int, jams::Real>> terms;
};

int read_integer(const Setting& setting, const char* name)
{
    if (!jams::is_integer_setting(setting)) {
        throw jams::ConfigException(setting, name, " must be an integer");
    }

    if (setting.getType() == Setting::TypeInt64) {
        return int(static_cast<int64_t>(setting));
    }

    return int(setting);
}

jams::Real read_real(const Setting& setting, const char* name)
{
    if (!setting.isNumber()) {
        throw jams::ConfigException(setting, name, " must be numeric");
    }

    if (setting.getType() == Setting::TypeInt) {
        return jams::Real(int(setting));
    }
    if (setting.getType() == Setting::TypeInt64) {
        return jams::Real(static_cast<int64_t>(setting));
    }

    return jams::Real(double(setting));
}

jams::Vec<jams::Real, 3> read_axis(const Setting& setting)
{
    jams::is_vec3_setting(setting);

    jams::Vec<jams::Real, 3> axis = {
        read_real(setting[0], "axis component"),
        read_real(setting[1], "axis component"),
        read_real(setting[2], "axis component")
    };

    const auto length = jams::norm(axis);
    if (::approximately_zero(length, decltype(length)(jams::defaults::lattice_tolerance))) {
        throw jams::ConfigException(setting, "axis must not be zero");
    }

    return axis / length;
}



jams::TesseralHarmonicNormalisation read_tesseral_normalisation(const Setting& settings)
{
    const Setting* normalisation_setting = nullptr;
    if (settings.exists("normalisation")) {
        normalisation_setting = &settings["normalisation"];
    }
    if (settings.exists("normalization")) {
        if (normalisation_setting != nullptr) {
            throw jams::ConfigException(settings, "specify either normalisation or normalization, not both");
        }
        normalisation_setting = &settings["normalization"];
    }

    if (normalisation_setting == nullptr) {
        return jams::TesseralHarmonicNormalisation::monic;
    }

    if (!normalisation_setting->isString()) {
        throw jams::ConfigException(*normalisation_setting, "normalisation must be a string");
    }

    std::string normalisation = lowercase(normalisation_setting->c_str());
    std::replace(normalisation.begin(), normalisation.end(), '_', '-');
    if (normalisation == "monic") {
        return jams::TesseralHarmonicNormalisation::monic;
    }
    if (normalisation == "condon-shortley") {
        return jams::TesseralHarmonicNormalisation::condon_shortley;
    }
    if (normalisation == "racah" || normalisation == "wybourne" ||
        normalisation == "racah-wybourne" || normalisation == "wybourne-racah" ||
        normalisation == "crystal-field") {
        return jams::TesseralHarmonicNormalisation::racah;
    }
    if (normalisation == "stevens" || normalisation == "stevens-operators") {
        return jams::TesseralHarmonicNormalisation::stevens;
    }

    throw jams::ConfigException(*normalisation_setting,
                                "normalisation must be one of: monic, condon-shortley, racah, stevens");
}

bool axes_match(const jams::Vec<jams::Real, 3>& lhs, const jams::Vec<jams::Real, 3>& rhs)
{
    const auto tolerance = jams::Real(jams::defaults::lattice_tolerance);
    return jams::vecs_are_approximately_equal(lhs, rhs, tolerance);
}

bool axes_match(const jams::Vec<jams::Real, 3>& lhs_u,
                const jams::Vec<jams::Real, 3>& lhs_v,
                const jams::Vec<jams::Real, 3>& lhs_w,
                const jams::Vec<jams::Real, 3>& rhs_u,
                const jams::Vec<jams::Real, 3>& rhs_v,
                const jams::Vec<jams::Real, 3>& rhs_w)
{
    return axes_match(lhs_u, rhs_u)
        && axes_match(lhs_v, rhs_v)
        && axes_match(lhs_w, rhs_w);
}

bool coefficient_is_axial(const std::pair<int, jams::Real>& coefficient)
{
    return jams::tesseral_polynomial::axial_coefficient_index_from_key(coefficient.first) >= 0;
}

bool coefficients_are_axial(const std::vector<std::pair<int, jams::Real>>& coefficients)
{
    return std::all_of(coefficients.begin(), coefficients.end(), coefficient_is_axial);
}

bool coefficients_are_axial(const AnisotropyPolynomialHamiltonian::TesseralKeyCoefficientMap& coefficients)
{
    return std::all_of(coefficients.begin(), coefficients.end(), [](const auto& term) {
        return term.second == jams::Real{0}
            || jams::tesseral_polynomial::axial_coefficient_index_from_key(term.first) >= 0;
    });
}

std::pair<int, jams::Real> read_coefficient_setting(
    const Setting& setting,
    const double energy_unit_conversion,
    const jams::TesseralHarmonicNormalisation normalisation)
{
    if (!setting.isList() || setting.getLength() != 3) {
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

    const auto normalisation_scale = jams::tesseral_monic_polynomial_normalisation_scale<jams::Real>(
        normalisation, l, m);
    return {jams::tesseral_key(l, m),
            read_real(setting[2], "anisotropy coefficient") * jams::Real(energy_unit_conversion) * normalisation_scale};
}

void read_coefficient_settings(std::vector<std::pair<int, jams::Real>>& coefficients,
                               const Setting& setting,
                               const double energy_unit_conversion,
                               const jams::TesseralHarmonicNormalisation normalisation)
{
    if (setting.isList() && setting.getLength() > 0 && setting[0].isList()) {
        for (auto i = 0; i < setting.getLength(); ++i) {
            coefficients.push_back(read_coefficient_setting(setting[i], energy_unit_conversion, normalisation));
        }
        return;
    }

    coefficients.push_back(read_coefficient_setting(setting, energy_unit_conversion, normalisation));
}

AnisotropyPolynomialSetting read_anisotropy_setting(
    const Setting& setting,
    const double energy_unit_conversion,
    const jams::TesseralHarmonicNormalisation normalisation)
{
    if (!setting.isList()) {
        throw jams::ConfigException(setting, "anisotropy must be a list");
    }

    const auto length = setting.getLength();
    if (length < 2) {
        throw jams::ConfigException(setting, "anisotropy must contain a target and at least one coefficient");
    }

    AnisotropyPolynomialSetting result;

    if (jams::is_integer_setting(setting[0])) {
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
    const auto has_some_axis_settings = length > 1 && jams::is_vec3_setting(setting[1]);
    if (has_some_axis_settings) {
        result.has_axes = true;
        if (length > 2 && jams::is_vec3_setting(setting[2])) {
            if (length < 5 || !jams::is_vec3_setting(setting[3])) {
                throw jams::ConfigException(setting, "anisotropy must specify either one axial axis, all three axes or no axes");
            }

            result.has_full_axes = true;
            result.u = read_axis(setting[1]);
            result.v = read_axis(setting[2]);
            result.w = read_axis(setting[3]);

            if (!jams::vecs_are_orthogonal(result.u, result.v, result.w)) {
                throw jams::ConfigException(setting, "u, v and w axes must be orthogonal");
            }

            coefficient_start = 4;
        } else {
            result.w = read_axis(setting[1]);
            coefficient_start = 2;
        }
    }

    for (auto i = coefficient_start; i < length; ++i) {
        read_coefficient_settings(result.coefficients, setting[i], energy_unit_conversion, normalisation);
    }

    if (result.coefficients.empty()) {
        throw jams::ConfigException(setting, "anisotropy must contain at least one coefficient");
    }
    if (result.has_axes && !result.has_full_axes && !coefficients_are_axial(result.coefficients)) {
        throw jams::ConfigException(setting, "a single anisotropy axis can only be used when all coefficients have m = 0");
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
        if (setting.has_full_axes) {
            u_axes(spin_index, j) = setting.u[j];
            v_axes(spin_index, j) = setting.v[j];
        }
        w_axes(spin_index, j) = setting.w[j];
    }
}

void add_axial_term_to_polynomial(std::array<jams::Real, 4>& axial_polynomial_coefficients,
                                  const int key,
                                  const jams::Real coefficient)
{
    switch (key) {
    case jams::tesseral_key(2, 0):
        axial_polynomial_coefficients[0] += coefficient * jams::Real(-1.0 / 3.0);
        axial_polynomial_coefficients[1] += coefficient;
        return;
    case jams::tesseral_key(4, 0):
        axial_polynomial_coefficients[0] += coefficient * jams::Real(3.0 / 35.0);
        axial_polynomial_coefficients[1] += coefficient * jams::Real(-6.0 / 7.0);
        axial_polynomial_coefficients[2] += coefficient;
        return;
    case jams::tesseral_key(6, 0):
        axial_polynomial_coefficients[0] += coefficient * jams::Real(-5.0 / 231.0);
        axial_polynomial_coefficients[1] += coefficient * jams::Real(5.0 / 11.0);
        axial_polynomial_coefficients[2] += coefficient * jams::Real(-15.0 / 11.0);
        axial_polynomial_coefficients[3] += coefficient;
        return;
    default:
        throw std::invalid_argument("non-axial tesseral key passed to axial polynomial builder");
    }
}

bool profiles_equal(const AnisotropyProfile& lhs, const AnisotropyProfile& rhs)
{
    return lhs.u == rhs.u
        && lhs.v == rhs.v
        && lhs.w == rhs.w
        && lhs.axis_mode == rhs.axis_mode
        && lhs.axial_polynomial == rhs.axial_polynomial
        && lhs.terms == rhs.terms;
}

bool profile_has_terms(const AnisotropyProfile& profile)
{
    return !profile.terms.empty()
        || profile.axial_polynomial[0] != jams::Real{0}
        || profile.axial_polynomial[1] != jams::Real{0}
        || profile.axial_polynomial[2] != jams::Real{0}
        || profile.axial_polynomial[3] != jams::Real{0};
}

int find_or_add_profile(std::vector<AnisotropyProfile>& profiles, const AnisotropyProfile& profile)
{
    const auto existing = std::find_if(profiles.begin(), profiles.end(),
        [&](const auto& candidate) {
            return profiles_equal(candidate, profile);
        });

    if (existing != profiles.end()) {
        return int(std::distance(profiles.begin(), existing));
    }

    profiles.push_back(profile);
    return int(profiles.size() - 1);
}

} // namespace

bool AnisotropyPolynomialHamiltonian::is_local_axis_setting(const libconfig::Setting& setting)
{
    return jams::is_vec3_setting(setting);
}

AnisotropyPolynomialHamiltonian::LocalAxes AnisotropyPolynomialHamiltonian::read_optional_local_axes(
    const libconfig::Setting& setting,
    const int axis_start_index,
    const char* setting_name,
    int& value_start_index)
{
    LocalAxes axes;
    value_start_index = axis_start_index;

    const auto length = setting.getLength();
    const auto has_some_axis_settings = length > axis_start_index && is_local_axis_setting(setting[axis_start_index]);
    if (!has_some_axis_settings) {
        return axes;
    }

    axes.has_axes = true;
    if (length > axis_start_index + 1 && is_local_axis_setting(setting[axis_start_index + 1])) {
        if (length < axis_start_index + 3 || !is_local_axis_setting(setting[axis_start_index + 2])) {
            throw jams::ConfigException(setting, setting_name, " must specify either one axial axis, all three axes or no axes");
        }

        axes.has_full_axes = true;
        axes.u = read_axis(setting[axis_start_index]);
        axes.v = read_axis(setting[axis_start_index + 1]);
        axes.w = read_axis(setting[axis_start_index + 2]);

        if (!jams::vecs_are_orthogonal(axes.u, axes.v, axes.w)) {
            throw jams::ConfigException(setting, "u, v and w axes must be orthogonal");
        }

        value_start_index = axis_start_index + 3;
    } else {
        axes.w = read_axis(setting[axis_start_index]);
        value_start_index = axis_start_index + 1;
    }
    return axes;
}

void AnisotropyPolynomialHamiltonian::write_local_axes_for_spin(
    const int spin_index,
    const LocalAxes& axes)
{
    if (!axes.has_axes) {
        return;
    }

    for (auto j = 0; j < 3; ++j) {
        if (axes.has_full_axes) {
            u_axes_(spin_index, j) = axes.u[j];
            v_axes_(spin_index, j) = axes.v[j];
        }
        w_axes_(spin_index, j) = axes.w[j];
    }
}

AnisotropyPolynomialHamiltonian::AnisotropyPolynomialHamiltonian(const libconfig::Setting& settings,
    const unsigned int size)
    : Hamiltonian(settings, size)
{
    if (!settings.exists("anisotropies")) {
        throw jams::ConfigException(settings, "missing anisotropies");
    }

    const auto normalisation = read_tesseral_normalisation(settings);

    const auto& anisotropy_settings = settings["anisotropies"];
    if (!anisotropy_settings.isList()) {
        throw jams::ConfigException(anisotropy_settings, "anisotropies must be a list");
    }
    if (anisotropy_settings.getLength() == 0) {
        throw jams::ConfigException(anisotropy_settings, "anisotropies must contain at least one entry");
    }

    initialise_tesseral_storage(size);

    std::vector<TesseralKeyCoefficientMap> spin_coefficients(size);
    std::vector<bool> spin_axes_explicitly_set(size, false);
    std::vector<bool> spin_axes_are_full(size, false);

    for (auto n = 0; n < anisotropy_settings.getLength(); ++n) {
        const auto& anisotropy_setting = anisotropy_settings[n];
        const auto anisotropy = read_anisotropy_setting(
            anisotropy_setting, input_energy_unit_conversion_, normalisation);

        for (auto i = 0u; i < size; ++i) {
            if (!applies_to_spin(anisotropy, int(i))) {
                continue;
            }

            if (anisotropy.has_axes && spin_axes_explicitly_set[i]) {
                const auto existing_u = axis_for_spin(u_axes_, int(i));
                const auto existing_v = axis_for_spin(v_axes_, int(i));
                const auto existing_w = axis_for_spin(w_axes_, int(i));
                if (anisotropy.has_full_axes && spin_axes_are_full[i] &&
                    !axes_match(existing_u, existing_v, existing_w, anisotropy.u, anisotropy.v, anisotropy.w)) {
                    throw jams::ConfigException(anisotropy_setting,
                                                "anisotropy axes are specified inconsistently for spin ",
                                                i);
                }
                if ((!anisotropy.has_full_axes || !spin_axes_are_full[i]) && !axes_match(existing_w, anisotropy.w)) {
                    throw jams::ConfigException(anisotropy_setting,
                                                "anisotropy axes are specified inconsistently for spin ",
                                                i);
                }
                if (anisotropy.has_full_axes && !spin_axes_are_full[i]) {
                    write_axes_for_spin(u_axes_, v_axes_, w_axes_, int(i), anisotropy);
                    spin_axes_are_full[i] = true;
                }
            } else if (anisotropy.has_axes) {
                write_axes_for_spin(u_axes_, v_axes_, w_axes_, int(i), anisotropy);
                spin_axes_explicitly_set[i] = true;
                spin_axes_are_full[i] = anisotropy.has_full_axes;
            }

            for (const auto& [key, coefficient] : anisotropy.coefficients) {
                spin_coefficients[i][key] += coefficient;
            }
        }
    }

    for (auto i = 0u; i < size; ++i) {
        if (spin_axes_explicitly_set[i] && !spin_axes_are_full[i] && !coefficients_are_axial(spin_coefficients[i])) {
            throw jams::ConfigException(
                anisotropy_settings,
                "a single anisotropy axis can only be used when all non-zero terms for a spin have m = 0");
        }
    }

    set_tesseral_terms(spin_coefficients);
}

AnisotropyPolynomialHamiltonian::AnisotropyPolynomialHamiltonian(
    const libconfig::Setting& settings,
    const unsigned int size,
    EmptyStorageTag)
    : Hamiltonian(settings, size)
{
    initialise_tesseral_storage(size);
}

void AnisotropyPolynomialHamiltonian::initialise_tesseral_storage(const unsigned int size)
{
    zero(spin_profile_.resize(size));
    zero(active_spin_indices_.resize(0));
    zero(profile_pointer_.resize(1));
    zero(axial_polynomial_coefficients_.resize(size, 4));
    zero(u_axes_.resize(size, 3));
    zero(v_axes_.resize(size, 3));
    zero(w_axes_.resize(size, 3));
    zero(profile_axis_modes_.resize(size));

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
}

void AnisotropyPolynomialHamiltonian::set_tesseral_terms(
    const std::vector<TesseralKeyCoefficientMap>& spin_coefficients)
{
    if (spin_profile_.elements() != spin_coefficients.size()) {
        throw std::invalid_argument("spin coefficient count does not match Hamiltonian size");
    }
    if (u_axes_.extent(0) != int(spin_coefficients.size()) ||
        v_axes_.extent(0) != int(spin_coefficients.size()) ||
        w_axes_.extent(0) != int(spin_coefficients.size())) {
        throw std::invalid_argument("spin axes count does not match Hamiltonian size");
    }

    std::vector<AnisotropyProfile> profiles;
    for (auto i = 0u; i < spin_coefficients.size(); ++i) {
        AnisotropyProfile profile;
        profile.u = axis_for_spin(u_axes_, int(i));
        profile.v = axis_for_spin(v_axes_, int(i));
        profile.w = axis_for_spin(w_axes_, int(i));

        for (const auto& [key, coefficient] : spin_coefficients[i]) {
            if (coefficient == jams::Real{0}) {
                continue;
            }

            const auto axial_index = jams::tesseral_polynomial::axial_coefficient_index_from_key(key);
            if (axial_index >= 0) {
                add_axial_term_to_polynomial(profile.axial_polynomial, key, coefficient);
            } else {
                profile.terms.push_back({key, coefficient});
            }
        }

        const auto has_default_axes = profile.u == kDefaultU && profile.v == kDefaultV && profile.w == kDefaultW;
        if (has_default_axes) {
            profile.axis_mode = jams::tesseral_polynomial::kProfileAxesDefault;
        } else if (profile.terms.empty()) {
            profile.axis_mode = jams::tesseral_polynomial::kProfileAxesAxial;
        } else {
            profile.axis_mode = jams::tesseral_polynomial::kProfileAxesFull;
        }

        spin_profile_(i) = find_or_add_profile(profiles, profile);
    }

    zero(u_axes_.resize(profiles.size(), 3));
    zero(v_axes_.resize(profiles.size(), 3));
    zero(w_axes_.resize(profiles.size(), 3));
    zero(profile_axis_modes_.resize(profiles.size()));
    zero(axial_polynomial_coefficients_.resize(profiles.size(), 4));
    zero(profile_pointer_.resize(profiles.size() + 1));

    auto total_terms = 0;
    for (auto i = 0u; i < profiles.size(); ++i) {
        profile_pointer_(i) = total_terms;
        total_terms += int(profiles[i].terms.size());
        profile_axis_modes_(i) = profiles[i].axis_mode;

        for (auto j = 0; j < 3; ++j) {
            u_axes_(i, j) = profiles[i].u[j];
            v_axes_(i, j) = profiles[i].v[j];
            w_axes_(i, j) = profiles[i].w[j];
        }

        for (auto j = 0; j < 4; ++j) {
            axial_polynomial_coefficients_(i, j) = profiles[i].axial_polynomial[j];
        }
    }
    profile_pointer_(profiles.size()) = total_terms;

    std::vector<int> active_spin_indices;
    active_spin_indices.reserve(spin_coefficients.size());
    for (auto i = 0u; i < spin_coefficients.size(); ++i) {
        if (profile_has_terms(profiles[spin_profile_(i)])) {
            active_spin_indices.push_back(int(i));
        }
    }
    active_spin_indices_.resize(active_spin_indices.size());
    for (auto i = 0u; i < active_spin_indices.size(); ++i) {
        active_spin_indices_(i) = active_spin_indices[i];
    }

    tesseral_keys_.resize(total_terms);
    tesseral_coefficients_.resize(total_terms);

    auto term_index = 0;
    for (const auto& profile : profiles) {
        for (const auto& [key, coefficient] : profile.terms) {
            tesseral_keys_(term_index) = key;
            tesseral_coefficients_(term_index) = coefficient;
            ++term_index;
        }
    }
}

jams::Vec<jams::Real, 3> AnisotropyPolynomialHamiltonian::calculate_field(int i, jams::Real time)
{
    const auto spins = std::as_const(globals::s).host_view();
    jams::Real field[3];
    jams::tesseral_polynomial::field_for_spin_with_profiles(
        i,
        jams::Real(spins(i, 0)),
        jams::Real(spins(i, 1)),
        jams::Real(spins(i, 2)),
        std::as_const(spin_profile_).host_data(),
        std::as_const(u_axes_).host_data(),
        std::as_const(v_axes_).host_data(),
        std::as_const(w_axes_).host_data(),
        std::as_const(profile_axis_modes_).host_data(),
        std::as_const(profile_pointer_).host_data(),
        std::as_const(tesseral_keys_).host_data(),
        std::as_const(tesseral_coefficients_).host_data(),
        std::as_const(axial_polynomial_coefficients_).host_data(),
        field);

    return {field[0], field[1], field[2]};
}

jams::Real AnisotropyPolynomialHamiltonian::calculate_energy(int i, jams::Real time)
{
    const auto spins = std::as_const(globals::s).host_view();
    return jams::tesseral_polynomial::energy_for_spin_with_profiles(
        i,
        jams::Real(spins(i, 0)),
        jams::Real(spins(i, 1)),
        jams::Real(spins(i, 2)),
        std::as_const(spin_profile_).host_data(),
        std::as_const(u_axes_).host_data(),
        std::as_const(v_axes_).host_data(),
        std::as_const(w_axes_).host_data(),
        std::as_const(profile_axis_modes_).host_data(),
        std::as_const(profile_pointer_).host_data(),
        std::as_const(tesseral_keys_).host_data(),
        std::as_const(tesseral_coefficients_).host_data(),
        std::as_const(axial_polynomial_coefficients_).host_data());
}

jams::Real AnisotropyPolynomialHamiltonian::calculate_energy_for_spin(int i, const jams::Vec<double, 3> &spin, jams::Real time)
{
    return jams::tesseral_polynomial::energy_for_spin_with_profiles(
        i,
        jams::Real(spin[0]),
        jams::Real(spin[1]),
        jams::Real(spin[2]),
        std::as_const(spin_profile_).host_data(),
        std::as_const(u_axes_).host_data(),
        std::as_const(v_axes_).host_data(),
        std::as_const(w_axes_).host_data(),
        std::as_const(profile_axis_modes_).host_data(),
        std::as_const(profile_pointer_).host_data(),
        std::as_const(tesseral_keys_).host_data(),
        std::as_const(tesseral_coefficients_).host_data(),
        std::as_const(axial_polynomial_coefficients_).host_data());
}
