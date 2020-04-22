// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class ZeemanHamiltonian : public Hamiltonian {
    friend class CudaZeemanHamiltonian;

public:
    ZeemanHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_one_spin_field(int i) override;

    double calculate_one_spin_energy(int i) override;

    double calculate_one_spin_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    jams::MultiArray<double, 2> dc_local_field_;
    jams::MultiArray<double, 2> ac_local_field_;
    jams::MultiArray<double, 1> ac_local_frequency_;

    bool has_ac_local_field_;
};

#endif  // JAMS_HAMILTONIAN_ZEEMAN_H