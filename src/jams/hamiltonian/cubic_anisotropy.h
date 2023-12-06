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
    jams::MultiArray<unsigned, 1> order_;
    jams::MultiArray<double, 2> u_axes_;
    jams::MultiArray<double, 2> v_axes_;
    jams::MultiArray<double, 2> w_axes_;
    jams::MultiArray<double, 1> magnitude_;
};

#endif //JAMS_CUBIC_ANISOTROPY_H
