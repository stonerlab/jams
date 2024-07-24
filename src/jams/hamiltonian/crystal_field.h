#ifndef JAMS_HAMILTONIAN_CRYSTAL_FIELD_H
#define JAMS_HAMILTONIAN_CRYSTAL_FIELD_H

#include <jams/core/hamiltonian.h>

///
/// Hamiltonian for crystal fields
///
/// \f[
///     \mathcal{H} = \sum_{i} \sum_{l=2,4,6}\sum_{m=-l}^l A_{l} B_{l,m} Y_{l,m}(\vec{S}_i)
/// \f]
///
/// Where the A's are defined as
///
/// \f[
///     A_2 &= J (J - 0.5) \alpha_J \\
///     A_4 &= J (J - 0.5) (J - 1) (J - 1.5) \beta_J
///     A_6 &= J (J - 0.5) (J - 1) (J - 1.5) (J - 2) (J - 2.5) \gamma_J
/// \f]
///
/// with Stevens factors \alpha_J, \beta_J, \gamma_J and J from Hund's rules. B_{l,m} is
/// a complex crystal field coefficient and Y_{l,m} is a spherical harmonic. The Hamiltonian
/// is only implemented for l=2,4,6.
///
///
/// ---------------------------------------------------------------------------
/// config settings
/// ---------------------------------------------------------------------------
///
/// energy_units: (sting) name of the units of energy of the input.
///
/// energy_cutoff: (float, required) absolute energies which are smaller than
///                this are set to zero. This is a required setting because
///                it is used to check that the imaginary part of the tesseral
///                harmonics is zero.
///
/// crystal_field_coefficients: (list of lists) each sub list takes the form
///                (material, J, alphaJ, betaJ, gammaJ, cf_param_filename)
///                where material can be a name or unit cell positions,
///                J, alphaJ, betaJ, gammaJ are floats defined in the crystal
///                field Hamiltonian and cf_param_filename is a filename which
///                contains the values of B_lm for this material. The file
///                should contain 6 columns: l m upR upIm dnR dnIm.
///
/// Example
/// -------
///
/// hamiltonians = (
/// {
///   module = "crystal-field";
///   debug = false;
///   energy_units = "meV";
///   energy_cutoff = 0.001;
///   crystal_field_coefficients = (
///     // (material, J, alphaJ, betaJ, gammaJ, cf_param_filename)
///     ("Tb", 6, -0.01010101, 0.00012244, -0.00000112, "Tb.CFparameters.dat")
///   );
/// }
/// );
///

class CrystalFieldHamiltonian : public Hamiltonian {

public:
    enum class CrystalFieldSpinType {kSpinUp, kSpinDown};

    using SphericalHarmonicCoefficientMap = std::map<std::pair<int, int>, std::complex<double>>;
    using TesseralHarmonicCoefficientMap = std::map<std::pair<int, int>, double>;

    CrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

protected:
    // Reads a crystal field coefficient file and returns a SphericalHarmonicCoefficientMap where the key is the pair
    // {l, m} and the value is the complex crystal field coefficient.
    //
    // The file should have columns 'l, m, Re(B_lm_up), Im(B_lm_up), Re(B_lm_down), Im(B_lm_down)', although the
    // columns Re(B_lm_down) and Im(B_lm_down) are ignored. Any missing values of l,m are set to zero.
    //
    // The returned map will be order (intrinsically due to C++ map) increasing in l and in m, e.g.
    // {0, 0}, {2, -2}, {2, 1}, {2, 0}, {2, 1}, {2, 2}, {4, -4}, {4, -3} ...
    SphericalHarmonicCoefficientMap read_crystal_field_coefficients_from_file(std::string filename);

    TesseralHarmonicCoefficientMap convert_spherical_to_tesseral(const SphericalHarmonicCoefficientMap& spherical_coefficients, const double zero_epsilon);

    // Maximum number of crystal field coefficients supported. 27 corresponds to l=2,4,6
    unsigned int kCrystalFieldNumCoeff_ = 27;

    CrystalFieldSpinType crystal_field_spin_type = CrystalFieldSpinType::kSpinUp;

    // Energy cutoff (in same units as input) to determine zero
    double energy_cutoff_ = 0.0;

    // Boolean array of whether a spin has a non-zero crystal field
    jams::MultiArray<bool, 1>  spin_has_crystal_field_;

    // Stores the tesseral crystal field coefficients for each spin.
    // (i.e. after transforming the complex crystal field coefficients
    // into the tesseral convention).
    //
    // index 0:
    //   tesseral coefficients (size kMaxCrystalFieldCoeff_)
    //   ordered in increasing l and with m -> -l...l
    // index 1:
    //   spin index (size num_spins)
    //
    // The first few values are
    // cf_coeff_[0,0] => C_{2,-2}, S_0
    // cf_coeff_[1,0] => C_{2,-1}, S_0
    // cf_coeff_[1,0] => C_{2, 0}, S_0
    // ...
    // cf_coeff_[0,1] => C_{2,-2}, S_1
    //
    jams::MultiArray<double, 2> crystal_field_tesseral_coeff_;

};

#endif  // JAMS_HAMILTONIAN_CRYSTAL_FIELD_H