//
// Created by Sean Stansill [ll14s26s] on 28/10/2019.
//

#ifndef JAMS_CUBIC_ANISOTROPY_H
#define JAMS_CUBIC_ANISOTROPY_H
#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class CubicHamiltonian : public Hamiltonian {
    friend class CudaCubicHamiltonian;  // Don't forget to make this
public:
    CubicHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CubicHamiltonian() {};

    double calculate_total_energy_cube();
    double calculate_one_spin_energy_cube(const int i);
    double calculate_one_spin_energy_difference_cube(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

    void   calculate_energies_cube();

    void   calculate_one_spin_field_cube(const int i, double h[3]);
    void   calculate_fields_cube();
private:
    unsigned num_coefficients_ = 0;
    jams::MultiArray<unsigned, 2> power_;
    jams::MultiArray<double, 4>     axis_; // Changed from 3. Need the fourth item for the three different cubic axes
    jams::MultiArray<double, 2>   magnitude_;
};

#endif //JAMS_CUBIC_ANISOTROPY_H
