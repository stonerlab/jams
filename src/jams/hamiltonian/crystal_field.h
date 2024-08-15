#ifndef JAMS_HAMILTONIAN_CRYSTAL_FIELD_H
#define JAMS_HAMILTONIAN_CRYSTAL_FIELD_H

#include <jams/core/hamiltonian.h>

///
/// Hamiltonian for crystal fields
///
/// \f[
///     \mathcal{H} = \sum_{l=2,4,6} A_l(J) \sum_{m=-l}^{l} B_{l,-m} (-1)^m d_{l}^{0m}(\theta)e^{-i m \phi}
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
/// energy_units: energy_units (optional | string)
///     Energy units of the crystal field coefficients in one of the JAMS supported units
///
/// energy_cutoff: (required | float)
///     Coefficients with an absolute value less than this (technically tesseral coefficient |C_{l,m}| < E_{cutoff})
///     will be set to zero. This setting is also used to check that the imaginary part of the energy is less than :
///     E_{cutoff} after conversion from complex crystal field coefficients B{l,m} to tesseral coefficients C{l,m}.
///     If this check fails then JAMS will error and the input should be checked. Units for the cutoff are the same as
///     `energy_units` so the cutoff and the interpretation of a negligible energy should be with respect to these units.
///
/// crystal_field_spin_type: (required | "up" or "down")
///     The crystal field input file contains data for both spin up and spin down. This setting selects which data to
///     use. The choice should be made based on the physics of the local moment and the filling of the f-shell.
///
/// crystal_field_coefficients (required | list)
///      A list of the crystal field parameters for each material or unit cell position. Each list element is another
///      list with the format: (material, J, alphaJ, betaJ, gammaJ, cf_param_filename), where material can be a material
///      name or unit cell position, and cf_param_filename is a filename for the file which contains the crystal field
///      coefficients B_{l,m} for that material.
///
/// Crystal Field File Format
/// -------------------------
///
/// The crystal field input file should have columns of data in the format :code:`l m upRe upIm dnRe dnIm` which
/// are `l`, `m`, `\Re(B_{l,m}^{\uparrow})`, `\Im(B_{l,m}^{\uparrow})`, `\Re(B_{l,m}^{\downarrow})`,
/// `\Im(B_{l,m}^{\downarrow})` with the units given in the `energy_units` setting. Coefficients should only be given
/// for `l=0,2,4,6` and `m = -l \dots l`. Any missing coefficients will be set to zero.
///
/// Example
/// -------
///
///  hamiltonians = (
///      {
///        module = "crystal-field";
///        debug = false;
///        energy_units = "Kelvin"
///        energy_cutoff = 1e-1;
///        crystal_field_spin_type = "down";
///        crystal_field_coefficients = (
///            ("Tb", 6, -0.01010101, 0.00012244, -0.00000112, "Tb.CFparameters.dat"));
///      }
///  );
///

class CrystalFieldHamiltonian : public Hamiltonian {

public:
  using SphericalHarmonicCoefficientMap = std::map<std::pair<int, int>, std::complex<double>>;
  using TesseralHarmonicCoefficientMap = std::map<std::pair<int, int>, double>;

  enum class CrystalFieldSpinType {
    kSpinUp,
    kSpinDown
  };

  CrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

  double calculate_total_energy(double time) override;

  void calculate_energies(double time) override;

  void calculate_fields(double time) override;

  Vec3 calculate_field(int i, double time) override;

  double calculate_energy(int i, double time) override;

  double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

protected:
  double crystal_field_energy(int i, const Vec3& s);

  // Reads a crystal field coefficient file and returns a SphericalHarmonicCoefficientMap where the key is the pair
  // {l, m} and the value is the complex crystal field coefficient.
  //
  // The file should have columns 'l, m, Re(B_lm_up), Im(B_lm_up), Re(B_lm_down), Im(B_lm_down)', although the
  // columns Re(B_lm_down) and Im(B_lm_down) are ignored. Any missing values of l,m are set to zero.
  //
  // The returned map will be order (intrinsically due to C++ map) increasing in l and in m, e.g.
  // {0, 0}, {2, -2}, {2, 1}, {2, 0}, {2, 1}, {2, 2}, {4, -4}, {4, -3} ...
  SphericalHarmonicCoefficientMap read_crystal_field_coefficients_from_file(const std::string& filename);

  static TesseralHarmonicCoefficientMap convert_spherical_to_tesseral(const SphericalHarmonicCoefficientMap& spherical_coefficients, const double zero_epsilon);

  // Maximum number of crystal field coefficients supported. 27 corresponds to l=2,4,6
  const unsigned int kCrystalFieldNumCoeff_ = 27;

  // Energy cutoff in input units
  double energy_cutoff_;

  CrystalFieldSpinType crystal_field_spin_type_;

  // Boolean array of whether a spin has a non-zero crystal field
  jams::MultiArray<bool, 1>  spin_has_crystal_field_;

  // Tesseral crystal field coefficients for each spin (axis 0: coefficient index, 1: spin index)
  jams::MultiArray<double, 2> crystal_field_tesseral_coeff_;

};

#endif  // JAMS_HAMILTONIAN_CRYSTAL_FIELD_H