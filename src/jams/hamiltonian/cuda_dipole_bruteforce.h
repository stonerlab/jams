// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H

#include <jams/core/hamiltonian.h>

class CudaDipoleBruteforceHamiltonian : public Hamiltonian {
    public:
        CudaDipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size);

        jams::Real calculate_total_energy(jams::Real time) override;
        jams::Real calculate_energy(const int i, jams::Real time) override;
        jams::Real calculate_one_spin_energy(const int i, const Vec3 &s_i, jams::Real time);
        jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;
        void   calculate_energies(jams::Real time) override;

        Vec3R   calculate_field(const int i, jams::Real time) override;
        void   calculate_fields(jams::Real time) override;

    private:
        jams::Real r_cutoff_;
        jams::Real dipole_prefactor_;

        jams::MultiArray<jams::Real, 2> r_float_;
        jams::MultiArray<jams::Real, 1> mus_float_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
