#ifndef JAMS_HAMILTONIAN_CRYSTAL_FIELD_H
#define JAMS_HAMILTONIAN_CRYSTAL_FIELD_H

#include <jams/core/hamiltonian.h>

///
/// The crystal field Hamiltonian will often be used only for rare-earths
/// within more complex magnets. We read coefficients and calculate the
/// Hamiltonian only for that subset of spins (to avoid having to write
/// a lot of zeros in the input file).
///


/// Example
/// -------
///
/// hamiltonians = (
/// {
///   module = "crystal-field";
///   unit_name = "meV";
///   # One array per position in the unit cell
///   # unitcell position or materials, 'B' indices, real and imaginary parts of crystal field coefficient
///   #  ( atom, [l, m], [Re(B_lm), Im(B_lm)])
///   crystal_field_coefficients =
///     (( 1, [0, 0], [0.25, 0.0]),);
/// }
/// );
///

class CrystalFieldHamiltonian : public Hamiltonian {

public:
    CrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

    using SphericalHarmonicCoefficientMap = std::map<std::pair<int, int>, std::complex<double>>;
    using TesseralHarmonicCoefficientMap = std::map<std::pair<int, int>, double>;

protected:
    // Reads a crystal field coefficient file and returns a map where the key is the pair {l, m} and the value is
    // the complex crystal field coefficient.
    //
    // The file to be read should have columns 'l, m, Re(B_lm_up), Im(B_lm_up), Re(B_lm_down), Im(B_lm_down)'. The
    // 'up' and 'down' data will be averaged.
    //
    // The returned map will be order (intrinsically due to C++ map) increasing in l and in m, e.g.
    // {0, 0}, {2, -2}, {2, 1}, {2, 0}, {2, 1}, {2, 2}, {4, -4}, {4, -3} ...
    SphericalHarmonicCoefficientMap read_crystal_field_coefficients_from_file(std::string filename);

    TesseralHarmonicCoefficientMap convert_spherical_to_tesseral(const SphericalHarmonicCoefficientMap& spherical_coefficients, const double zero_epsilon);

    // Maximum number of crystal field coefficients supported
    // 27 corresponds to l=2,4,6, m = -l...l.
    unsigned int kCrystalFieldNumCoeff_ = 27;

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