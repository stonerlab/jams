// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_NEARTREE_H
#define JAMS_HAMILTONIAN_DIPOLE_NEARTREE_H

#include <jams/core/hamiltonian.h>
#include <jams/lattice/interaction_neartree.h>

#include <utility>
#include <functional>

namespace jams
{
    template <typename CoordType>
    class InteractionNearTree;
}

class DipoleNearTreeHamiltonian : public Hamiltonian {
public:
    using NearTreeFunctorType = std::function<jams::Real(const std::pair<Vec<jams::Real, 3>, int>& a, const std::pair<Vec<jams::Real, 3>, int>& b)>;
    using NearTreeType = jams::NearTree<std::pair<Vec<jams::Real, 3>, int>, NearTreeFunctorType>;

    DipoleNearTreeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec<double, 3> &spin_initial, const Vec<double, 3> &spin_final, jams::Real time) override;

private:
    jams::Real r_cutoff_; // cutoff radius for dipole interaction

    jams::InteractionNearTree<jams::Real> neartree_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_NEARTREE_H
