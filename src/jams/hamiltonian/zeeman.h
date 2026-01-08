// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/multiarray.h>

class ZeemanHamiltonian : public Hamiltonian {
    friend class CudaZeemanHamiltonian;

public:
    ZeemanHamiltonian(const libconfig::Setting &settings, unsigned int size);

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

private:
    jams::MultiArray<jams::Real, 2> dc_local_field_;
    jams::MultiArray<jams::Real, 2> ac_local_field_;
    jams::MultiArray<jams::Real, 1> ac_local_frequency_;

    bool has_ac_local_field_;
};

#endif  // JAMS_HAMILTONIAN_ZEEMAN_H