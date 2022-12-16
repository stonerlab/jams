// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/multiarray.h>

class ZeemanHamiltonian : public Hamiltonian {
    friend class CudaZeemanHamiltonian;

public:
    ZeemanHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

private:
    jams::MultiArray<double, 2> dc_local_field_;
    jams::MultiArray<double, 2> ac_local_field_;
    jams::MultiArray<double, 1> ac_local_frequency_;

    bool has_ac_local_field_;
};

#endif  // JAMS_HAMILTONIAN_ZEEMAN_H