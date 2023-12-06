//
// Created by Sean Stansill [ll14s26s] on 28/10/2019.
//

#ifndef JAMS_CUBIC_ANISOTROPY_H
#define JAMS_CUBIC_ANISOTROPY_H

#include <jams/core/hamiltonian.h>

///
/// Hamiltonian for cubic anisotropy
///
/// \f[
///     \mathcal{H} = -\sum_i K1_i [ (S_i . u_i)^2 (S_i . v_i)^2 + (S_i . v_i)^2 (S_i . w_i)^2 + (S_i . u_i)^2 (S_i . w_i)^2 ] -\sum_i K2_i [ (S_i . u_i)^2 (S_i . v_i)^2 (S_i . w_i)^2 ]
/// \f]
///
/// @details K1_i, K2_i are the first and second cubic anisotropy energies
/// on each site and u_i, v_i, w_i are the local reference axes for each site.
///
/// Settings
/// --------
/// The Hamiltonian settings are:
///
/// module       : "cubic-anisotropy"
///
/// energy_units : (string) units of the energies specified in this module
///
/// K1            : List of values of anisotropy settings where each setting
///                 is a list of (energy, [ux, uy, uz], [vx, vy, vz], [wx, wy, wz])
///
/// K2            : List of values of anisotropy settings where each setting
///                 is a list of (energy, [ux, uy, uz], [vx, vy, vz], [wx, wy, wz])
///
/// Notes
/// -----
/// The uvw axes in the config do not need to be normalised. We normalise them
/// on reading. This allows the directions to be given in clearer crystallographic
/// notation.
///
/// The uvw axes must be orthogonal. JAMS will raise an error if they are not.
///
/// Example
/// -------
///
/// hamiltonians = (
/// {
///   module = "cubic";
///   unit_name = "meV";
///   K1 = ( ( 0.01, [ 2, -1, -1 ], [ 0, 1, -1 ], [ 1, 1, 1 ] ));
/// }
/// );
///

class CubicHamiltonian : public Hamiltonian {
    friend class CudaCubicHamiltonian;

public:
    CubicHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

private:
    jams::MultiArray<unsigned, 1> order_;
    jams::MultiArray<double, 2> u_axes_;
    jams::MultiArray<double, 2> v_axes_;
    jams::MultiArray<double, 2> w_axes_;
    jams::MultiArray<double, 1> magnitude_;
};

#endif //JAMS_CUBIC_ANISOTROPY_H
