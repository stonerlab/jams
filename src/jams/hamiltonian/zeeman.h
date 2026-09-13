// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/multiarray.h>

class ZeemanHamiltonian : public Hamiltonian {
    friend class CudaZeemanHamiltonian;

public:
    ZeemanHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Vec<jams::Real, 3> calculate_field(int i, double time) override;

    jams::Real calculate_energy(int i, double time) override;

    EnergyCurrentInteractionSupport energy_current_interaction_support() const override {
      return EnergyCurrentInteractionSupport::None;
    }

protected:
    jams::Real calculate_energy_from_spins(int i, double time, const SpinHostView& spins) override;

    jams::Real calculate_energy_for_spin(int i, const jams::Vec<double, 3> &spin, double time) override;

private:
    jams::MultiArray<jams::Real, 2> dc_local_field_;
    jams::MultiArray<jams::Real, 2> ac_local_field_;
    jams::MultiArray<double, 1> ac_local_frequency_;

    bool has_ac_local_field_;
};

#endif  // JAMS_HAMILTONIAN_ZEEMAN_H
