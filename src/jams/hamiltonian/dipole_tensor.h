// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_TENSOR_H
#define JAMS_HAMILTONIAN_DIPOLE_TENSOR_H

#include <libconfig.h++>

#include "jams/hamiltonian/sparse_interaction.h"
#include "jams/containers/multiarray.h"

class DipoleTensorHamiltonian : public SparseInteractionHamiltonian {
public:
    DipoleTensorHamiltonian(const libconfig::Setting &settings, unsigned int size);

private:
    double r_cutoff_; // cutoff radius for dipole interaction
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_TENSOR_H