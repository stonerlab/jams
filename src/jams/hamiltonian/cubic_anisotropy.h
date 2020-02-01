//
// Created by Sean Stansill [ll14s26s] on 28/10/2019.
//

#ifndef JAMS_CUBIC_ANISOTROPY_H
#define JAMS_CUBIC_ANISOTROPY_H
#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class CubicHamiltonian : public Hamiltonian {
    friend class CudaCubicHamiltonian;
public:
    CubicHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CubicHamiltonian() {};

    double calculate_total_energy();
    double calculate_one_spin_energy(const int i);
    double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

    void   calculate_energies();

    void   calculate_one_spin_field(const int i, double h[3]);
    void   calculate_fields();
private:
    unsigned num_coefficients_ = 0;
    jams::MultiArray<unsigned, 2> order_;
    jams::MultiArray<Vec3, 2>   axis1_;
    jams::MultiArray<Vec3, 2>   axis2_;
    jams::MultiArray<Vec3, 2>   axis3_;
    jams::MultiArray<double, 2>   magnitude_;
};

#endif //JAMS_CUBIC_ANISOTROPY_H
