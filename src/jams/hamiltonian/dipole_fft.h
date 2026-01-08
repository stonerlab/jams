// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_DIPOLE_FFT_H

#include <jams/core/hamiltonian.h>
#include <jams/core/types.h>
#include <jams/interface/fft.h>
#include <jams/helpers/mixed_precision.h>

class DipoleFFTHamiltonian : public Hamiltonian {
public:
    DipoleFFTHamiltonian(const libconfig::Setting &settings, unsigned int size);

    ~DipoleFFTHamiltonian() override;

    jams::Real calculate_total_energy(jams::Real time) override;

    void calculate_fields(jams::Real time) override;

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;

private:

    jams::MultiArray<jams::ComplexLo, 5> generate_kspace_dipole_tensor(int pos_i, const int pos_j, std::vector<Vec3>& generated_positions);

    bool debug_ = false;
    bool check_radius_ = true;
    bool check_symmetry_ = true;

    jams::Real r_cutoff_ = 0.0;
    jams::Real r_distance_tolerance_ = jams::defaults::lattice_tolerance;

    jams::MultiArray<jams::RealHi, 4> rspace_s_;
    jams::MultiArray<jams::RealHi, 4> rspace_h_;

    std::array<unsigned, 3> kspace_size_ = {0, 0, 0};
    std::array<unsigned, 3> kspace_padded_size_ = {0, 0, 0};
    jams::MultiArray<jams::ComplexHi, 4> kspace_s_;
    jams::MultiArray<jams::ComplexHi, 4> kspace_h_;

    std::vector<std::vector<jams::MultiArray<jams::ComplexLo, 5>>> kspace_tensors_;

    fftw_plan fft_s_rspace_to_kspace = nullptr;
    fftw_plan fft_h_kspace_to_rspace = nullptr;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_FFT_H