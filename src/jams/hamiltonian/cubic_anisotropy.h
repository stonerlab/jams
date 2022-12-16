//
// Created by Sean Stansill [ll14s26s] on 28/10/2019.
//

#ifndef JAMS_CUBIC_ANISOTROPY_H
#define JAMS_CUBIC_ANISOTROPY_H

#include <jams/core/hamiltonian.h>

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
    unsigned num_coefficients_ = 0;
    jams::MultiArray<unsigned, 2> order_;
    jams::MultiArray<Vec3, 2> axis1_;
    jams::MultiArray<Vec3, 2> axis2_;
    jams::MultiArray<Vec3, 2> axis3_;
    jams::MultiArray<double, 2> magnitude_;
};

#endif //JAMS_CUBIC_ANISOTROPY_H
