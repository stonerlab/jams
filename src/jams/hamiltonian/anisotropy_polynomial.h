#ifndef JAMS_ANISOTROPY_POLYNOMIAL_H
#define JAMS_ANISOTROPY_POLYNOMIAL_H
#include "jams/core/hamiltonian.h"

#include <map>
#include <vector>

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
///           "racah-wybourne", "wybourne-racah" and "crystal-field" are also
///           accepted.
///
///       "stevens"
///           Coefficients multiply the classical Stevens tesseral polynomial
///           convention. For example, the l=2,m=0 term is 3z^2 - 1. The alias
///           "stevens-operators" is also accepted.
///
/// anisotropies: (required | list)
///     A list of anisotropy definitions for each material or unit cell position.
///     Each definition has the format:
///         (target, u, v, w, coefficient...)
///     or, for purely axial m=0 anisotropy:
///         (target, w, coefficient...)
///     where target is either a material name or a unit cell position, u, v and w
///     are the local reference axes, w is the local axial direction, and each
///     coefficient has the format:
///         (l, m, C_lm)
///
///     The axes may be omitted, in which case the defaults are:
///         u = [1.0, 0.0, 0.0]
///         v = [0.0, 1.0, 0.0]
///         w = [0.0, 0.0, 1.0]
///
///     If axes are provided, either one axial w axis or all three u, v and w
///     axes must be specified. Axes are normalised on input; full u, v and w
///     axes must be mutually orthogonal. A single w axis is only valid when all
///     non-zero terms applying to that spin have m=0. Omitted axes use the
///     default frame unless another matching anisotropy definition provides
///     explicit axes for the same spin. All explicit local axes that apply to a
///     given spin must be consistent; it is malformed input to define multiple
///     anisotropies for the same spin with different local frames. For one-axis
///     axial definitions, consistency is checked using the w axis.
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
/// Design notes
/// ------------
///
/// The fundamental internal basis is the monic tesseral polynomial basis. Input
/// normalisation conventions are converted once, during construction, by
/// multiplying the user coefficient by the scale factor needed to obtain the
/// corresponding monic polynomial. This keeps the runtime CPU and CUDA
/// evaluators independent of the input convention, avoids duplicating Racah,
/// Stevens or Condon-Shortley factors in kernels, and gives the evaluator a
/// small set of explicit polynomial functions with simple rational
/// coefficients. It also lets the crystal-field Hamiltonian reuse this class by
/// converting its angular functions onto the same monic basis.
///
/// The storage is organised around unique anisotropy profiles rather than one
/// full coefficient list per spin. Each spin stores an integer profile index.
/// A profile contains the local axes, a combined axial polynomial, and a
/// residual CSR-style list of non-axial tesseral terms. This matches the common
/// input pattern where many spins share the same material or unit-cell
/// anisotropy, reducing duplicated coefficient and axis data on both CPU and
/// CUDA.
///
/// Terms with m=0 are handled separately because they are expected to dominate
/// typical input files and depend only on z = s.w in the local frame. The l=2,
/// l=4 and l=6 axial coefficients are folded into a single polynomial
///     E(z) = A0 + A2 z^2 + A4 z^4 + A6 z^6
/// so energy and field evaluation can use a short Horner-style expression and
/// its analytic derivative. Axial-only profiles need only the w axis. Profiles
/// with default axes avoid axis loads altogether, axial profiles with a custom
/// axis load only w, and only profiles with residual non-axial terms need the
/// full u/v/w local-coordinate transform.
///
/// CUDA kernels launch over active_spin_indices_ rather than all spins. A spin
/// is active when its profile has a non-zero axial polynomial or residual
/// tesseral terms. CUDA field and energy arrays are zeroed before the active
/// spin kernels run, so inactive spins contribute exactly zero without
/// consuming a thread. This is beneficial for sparse anisotropy definitions
/// where only a subset of materials or basis sites carry anisotropy.
///
/// calculate_field returns the unconstrained Cartesian gradient field -dE/ds.
/// It is not projected onto the tangent plane of the unit sphere. This is
/// deliberate: the Hamiltonian reports the derivative of the energy function in
/// spin-coordinate space, while spin-length constraints should be handled
/// consistently by the solver or optimisation algorithm. Many spin dynamics
/// solvers enforce the constraint through cross products, renormalisation, or a
/// solver-level tangent projection; projecting inside one Hamiltonian but not
/// others would make the summed field convention inconsistent. A constrained
/// tangent field may be needed for algorithms that use the Hamiltonian field
/// directly as a constrained descent direction on |s|=1. In that case the
/// projection should be applied once at the integrator or optimiser level, to
/// the total field from all Hamiltonian terms.
///
class AnisotropyPolynomialHamiltonian : public Hamiltonian
{
public:
    using TesseralKeyCoefficientMap = std::map<int, jams::Real>;

    AnisotropyPolynomialHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

protected:
    struct LocalAxes {
        bool has_axes = false;
        bool has_full_axes = false;
        jams::Vec<jams::Real, 3> u = {1.0, 0.0, 0.0};
        jams::Vec<jams::Real, 3> v = {0.0, 1.0, 0.0};
        jams::Vec<jams::Real, 3> w = {0.0, 0.0, 1.0};
    };

    struct EmptyStorageTag {};

    AnisotropyPolynomialHamiltonian(const libconfig::Setting &settings, const unsigned int size, EmptyStorageTag);

    void initialise_tesseral_storage(const unsigned int size);
    void set_tesseral_terms(const std::vector<TesseralKeyCoefficientMap>& spin_coefficients);
    void write_local_axes_for_spin(int spin_index, const LocalAxes& axes);

    static bool is_local_axis_setting(const libconfig::Setting& setting);
    static LocalAxes read_optional_local_axes(const libconfig::Setting& setting,
                                             int axis_start_index,
                                             const char* setting_name,
                                             int& value_start_index);

    jams::Real calculate_energy_for_spin(int i, const jams::Vec<double, 3> &spin, jams::Real time) override;

    // Profile index used by each spin.
    jams::MultiArray<int, 1> spin_profile_;

    // Spins whose profile has a non-zero axial polynomial or residual terms.
    jams::MultiArray<int, 1> active_spin_indices_;

    // Local reference axes for each unique anisotropy profile.
    jams::MultiArray<jams::Real,2> u_axes_; /// u_axes_(profile_index, cart_component)
    jams::MultiArray<jams::Real,2> v_axes_; /// v_axes_(profile_index, cart_component)
    jams::MultiArray<jams::Real,2> w_axes_; /// w_axes_(profile_index, cart_component)
    jams::MultiArray<int, 1> profile_axis_modes_;

    // An array similar to CSR format where beginning and end index of the data
    // for a given profile in the key and coefficient arrays is stored.
    jams::MultiArray<int, 1> profile_pointer_;
    jams::MultiArray<int, 1> tesseral_keys_;
    jams::MultiArray<jams::Real, 1> tesseral_coefficients_;

    // Polynomial coefficients A0,A2,A4,A6 for the combined axial m=0 terms,
    // E(z) = A0 + A2 z^2 + A4 z^4 + A6 z^6. Stored separately from the
    // generic tesseral CSR terms so the common axial path can avoid the key
    // lookup and full local-axis transform.
    jams::MultiArray<jams::Real, 2> axial_polynomial_coefficients_; /// axial_polynomial_coefficients_(profile_index, power_index)
};

#endif //JAMS_ANISOTROPY_POLYNOMIAL_H
