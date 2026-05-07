#ifndef JAMS_ANISOTROPY_POLYNOMIAL_H
#define JAMS_ANISOTROPY_POLYNOMIAL_H
#include "jams/core/hamiltonian.h"

#include <map>

///
/// Hamiltonian for single-ion anisotropy written as a polynomial expansion in
/// real tesseral harmonics
///
/// \f[
///     \mathcal{H}_i = \sum_{l=2,4,6} \sum_{m=-l}^{l} C_{l,m}^{(i)}
///         Z_{l,m}(\mathbf{s}_i \cdot \mathbf{u}_i,
///                 \mathbf{s}_i \cdot \mathbf{v}_i,
///                 \mathbf{s}_i \cdot \mathbf{w}_i)
/// \f]
///
/// where \f$Z_{l,m}\f$ are monic tesseral polynomial harmonics. The Hamiltonian
/// is only implemented for l=2,4,6 and -l <= m <= l.
///
/// ---------------------------------------------------------------------------
/// config settings
/// ---------------------------------------------------------------------------
///
/// energy_units: energy_units (optional | string)
///     Energy units of the anisotropy coefficients in one of the JAMS supported
///     units.
///
/// normalisation: monic (optional | string)
///     Normalisation convention used by the coefficients in the input file.
///     The American spelling "normalization" is also accepted. The selected
///     convention is converted to the internal monic polynomial basis when the
///     Hamiltonian is constructed, so the input coefficient C_lm is interpreted
///     as multiplying the requested normalised form.
///
///     Supported values:
///       "monic"
///           Coefficients multiply the internal monic tesseral polynomials
///           directly. For example, the l=2,m=0 term is z^2 - 1/3.
///
///       "condon-shortley"
///           Coefficients multiply unit-normalised real tesseral harmonics
///           using the Condon-Shortley phase convention. The internal monic
///           coefficient is multiplied by the corresponding normalisation
///           factor from jams::tesseral_monic_polynomial_normalisation_scale.
///
///       "racah"
///           Coefficients multiply Racah-normalised real tesseral harmonics,
///           C_lm = sqrt(4*pi/(2*l + 1)) Y_lm. This is the convention used by
///           the crystal-field Hamiltonian angular functions. For example, the
///           l=2,m=0 term is (3z^2 - 1)/2. The aliases "wybourne",
///           "racah-wybourne" and "wybourne-racah" are also accepted.
///
///       "stevens"
///           Coefficients multiply the classical Stevens tesseral polynomial
///           convention. For example, the l=2,m=0 term is 3z^2 - 1. The aliases
///           "stevens-operators" and "crystal-field" are also accepted.
///
/// anisotropies: (required | list)
///     A list of anisotropy definitions for each material or unit cell position.
///     Each definition has the format:
///         (target, u, v, w, coefficient...)
///     where target is either a material name or a unit cell position, u, v and w
///     are the local reference axes, and each coefficient has the format:
///         (l, m, C_lm)
///
///     The axes may be omitted, in which case the defaults are:
///         u = [1.0, 0.0, 0.0]
///         v = [0.0, 1.0, 0.0]
///         w = [0.0, 0.0, 1.0]
///
///     If axes are provided, all three axes must be specified. They are
///     normalised on input and must be mutually orthogonal. If anisotropies are
///     specified for the same spin by both material and unit cell position, the
///     local axes must be consistent.
///
/// Example
/// -------
///
///  hamiltonians = (
///      {
///        module = "anisotropy-polynomial";
///        energy_units = "meV";
///        anisotropies = (
///            ("A", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
///                (2, 0, 1.0)),
///            ("B", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
///                (2, 0, 2.0),
///                (4, 0, 0.1)),
///            (1, (2, 2, 0.5))
///        );
///      }
///  );
///
class AnisotropyPolynomialHamiltonian : public Hamiltonian
{
public:
    using TesseralKeyCoefficientMap = std::map<int, jams::Real>;

    AnisotropyPolynomialHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

protected:
    jams::Real calculate_energy_for_spin(int i, const jams::Vec<double, 3> &spin, jams::Real time) override;

    // Local reference axes for the anisotropy of a given spin
    jams::MultiArray<jams::Real,2> u_axes_; /// u_axes_(spin_index, cart_component)
    jams::MultiArray<jams::Real,2> v_axes_; /// v_axes_(spin_index, cart_component)
    jams::MultiArray<jams::Real,2> w_axes_; /// w_axes_(spin_index, cart_component)

    // An array similar to CSR format where beginning and end index of the data
    // for a given spin in the key and coefficient arrays is stored.
    jams::MultiArray<int, 1> spin_pointer_;
    jams::MultiArray<int, 1> tesseral_keys_;
    jams::MultiArray<jams::Real, 1> tesseral_coefficients_;
};

#endif //JAMS_ANISOTROPY_POLYNOMIAL_H
