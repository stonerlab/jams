//
// Created by Sean Stansill [ll14s26s] on 29/03/2023.
//

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_GENERALISED_H
#define JAMS_HAMILTONIAN_UNIAXIAL_GENERALISED_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/multiarray.h>

#include <vector>

class UniaxialGeneralisedHamiltonian : public Hamiltonian {
    friend class CudaUniaxialGeneralisedHamiltonian;

public:
    UniaxialGeneralisedHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

private:
    double a1_;
    double a2_;
    double a3_;
    double a4_;
    double a5_;
    double a6_;
    jams::MultiArray<Vec3, 1> axis1_;
    jams::MultiArray<Vec3, 1> axis2_;
    jams::MultiArray<Vec3, 1> axis3_;
    jams::MultiArray<double, 1> magnitude_; // magnitude of local anisotropy
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_GENERALISED_H