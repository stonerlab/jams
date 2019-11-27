// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_TENSOR_H
#define JAMS_HAMILTONIAN_DIPOLE_TENSOR_H

#include <libconfig.h++>


#include "jams/hamiltonian/sparse_interaction.h"
#include "jams/containers/multiarray.h"


class DipoleHamiltonianTensor : public SparseInteractionHamiltonian {
    public:
        DipoleHamiltonianTensor(const libconfig::Setting &settings, const unsigned int size);
    private:
        double r_cutoff_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_TENSOR_H