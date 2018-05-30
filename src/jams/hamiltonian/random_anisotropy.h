//
// Created by Joe Barker on 2018/05/28.
//
#ifndef JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H
#define JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"
#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class RandomAnisotropyHamiltonian : public Hamiltonian {
    public:
    RandomAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~RandomAnisotropyHamiltonian() override = default;

        double calculate_total_energy() override;
        double calculate_one_spin_energy(const int i) override;
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

        void   calculate_energies() override;

        void   calculate_one_spin_field(const int i, double h[3]) override;
        void   calculate_fields() override;
    protected:
        void output_anisotropy_axes(std::ofstream &outfile);

        std::vector<double> magnitude_;
        std::vector<Vec3>   direction_;
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H