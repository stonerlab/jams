// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_TENSOR_H
#define JAMS_HAMILTONIAN_DIPOLE_TENSOR_H

#include <jams/hamiltonian/sparse_interaction.h>

class DipoleTensorHamiltonian : public SparseInteractionHamiltonian {
public:
    DipoleTensorHamiltonian(const libconfig::Setting &settings, unsigned int size);

private:
    jams::Real r_cutoff_; // cutoff radius for dipole interaction
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_TENSOR_H