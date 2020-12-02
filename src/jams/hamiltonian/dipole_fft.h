// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_DIPOLE_FFT_H

#include <libconfig.h++>

#include "jams/types.h"
#include "jams/core/hamiltonian.h"
#include "jams/interface/fft.h"

class DipoleFFTHamiltonian : public Hamiltonian {
public:
    DipoleFFTHamiltonian(const libconfig::Setting &settings, unsigned int size);

    ~DipoleFFTHamiltonian() override;

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:

    jams::MultiArray<Complex, 5> generate_kspace_dipole_tensor(int pos_i, const int pos_j, std::vector<Vec3>& generated_positions);

    bool debug_ = false;
    bool check_radius_ = true;
    bool check_symmetry_ = true;

    double r_cutoff_ = 0.0;
    double r_distance_tolerance_ = jams::defaults::lattice_tolerance;

    jams::MultiArray<double, 4> rspace_s_;
    jams::MultiArray<double, 4> rspace_h_;

    std::array<unsigned, 3> kspace_size_ = {0, 0, 0};
    std::array<unsigned, 3> kspace_padded_size_ = {0, 0, 0};
    jams::MultiArray<Complex, 4> kspace_s_;
    jams::MultiArray<Complex, 4> kspace_h_;

    std::vector<std::vector<jams::MultiArray<Complex, 5>>> kspace_tensors_;

    fftw_plan fft_s_rspace_to_kspace = nullptr;
    fftw_plan fft_h_kspace_to_rspace = nullptr;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_FFT_H