// landau.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_HAMILTONIAN_LANDAU
#define INCLUDED_JAMS_HAMILTONIAN_LANDAU

#include <jams/core/hamiltonian.h>

///
/// Hamiltonian for the Landau model
///
/// \f[
///     \mathcal{H} = \sum_i A_i |S_i|^2 + B_i |S_i|^4 + C_i |S_i|^6
/// \f]
///
/// @details A_i, B_i, C_i are energies which describe the Landau energy
/// surface.
///
/// The effective field from the Hamiltonian is
///
/// \f[
///     \vec{H}_i = -2 A_i S_i - 4 B_i S_i |S_i|^2 - 6 C_i S_i |S_i|^4
/// \f]
///
/// Settings
/// --------
/// The Hamiltonian settings are:
///
/// module       : "landau"
///
/// energy_units : (string) units of the energies specified in this module
///
/// A            : List of values of 'A' in the Hamiltonian, one for each
///                material, in the order of the materials group.
///
/// B            : List of values of 'B' in the Hamiltonian, one for each
///                material, in the order of the materials group.
///
/// C            : List of values of 'C' in the Hamiltonian, one for each
///                material, in the order of the materials group.
///
/// Example
/// -------
///
/// hamiltonians = (
///   {
///     module = "landau";
///     energy_units = "meV";
///     A = [-440.0, 100.0];
///     B = [ 150.0,   0.0];
///     C = [ 50.0,    0.0];
///   }
/// );
///

class LandauHamiltonian : public Hamiltonian {

public:
    LandauHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

protected:
    jams::MultiArray<double,1> landau_A_;
    jams::MultiArray<double,1> landau_B_;
    jams::MultiArray<double,1> landau_C_;
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------