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
///   # Coefficients are ordered as
///   # B_{2,-2}, B_{2,-1}, B_{2, 0}, B_{2, 1}, B_{2, 2}, B_{4,-4}, B_{4,-3}, B_{4,-2}, B_{4,-1}, B_{4, 0}, B_{4, 1}, B_{4, 2}, B_{4, 3}, B_{4, 4}, B_{6,-6}, B_{6,-5}, B_{6,-4}, B_{6,-3}, B_{6,-2}, B_{6,-1}, B_{6, 0}, B_{6, 1}, B_{6, 2}, B_{6, 3}, B_{6, 4}, B_{6, 5}, B_{6, 6}
///   crystal_field_coefficients =
///     ( 1, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
/// }
/// );
///

//class CrystalFieldCoefficient {
//public:
//
//    inline void insert(int l, int m, std::complex<double> value) {
//        coefficients_.insert({{l, m}, value});
//    }
//
//    std::complex<double> operator()(int l, int m) {}
//private:
//
//    std::map<std::pair<int,int>, std::complex<double>> coefficients_;
//};

class CPUCrystalFieldHamiltonian : public Hamiltonian {

public:
    CPUCrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

protected:
    // Maximum number of crystal field coefficients supported
    // 27 corresponds to l=2,4,6, m = -l...l.
    const unsigned kCrystalFieldNumCoeff_ = 27;

    jams::MultiArray<int, 1> crystal_field_spin_indices_;

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