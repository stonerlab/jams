// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_DIPOLE_FFT_H

#include <fftw3.h>
#include <libconfig.h++>

#include "jams/types.h"
#include "jams/hamiltonian/strategy.h"


class DipoleHamiltonianFFT : public HamiltonianStrategy {
public:
    DipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size);
    ~DipoleHamiltonianFFT();

    double calculate_total_energy();
    double calculate_one_spin_energy(const int i);
    double calculate_one_spin_energy(const int i, const Vec3 &s_i);
    double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
    void   calculate_energies(jams::MultiArray<double, 1>& energies);
    void   calculate_one_spin_field(const int i, double h[3]);
    void   calculate_fields(jams::MultiArray<double, 2>& fields);

private:

    jams::MultiArray<Complex, 5> generate_kspace_dipole_tensor(const int pos_i, const int pos_j);

    bool debug_ = false;
    bool check_radius_   = true;
    bool check_symmetry_ = true;

    double r_cutoff_ = 0.0;
    double r_distance_tolerance_ = jams::defaults::lattice_tolerance;

    jams::MultiArray<double, 4> rspace_s_;
    jams::MultiArray<double, 4> rspace_h_;
    jams::MultiArray<double, 2> h_;

    std::array<unsigned,3>           kspace_size_ = {0, 0, 0};
    std::array<unsigned,3>           kspace_padded_size_ = {0, 0, 0};
    jams::MultiArray<Complex, 4>   kspace_s_;
    jams::MultiArray<Complex, 4>   kspace_h_;

    std::vector<std::vector<jams::MultiArray<Complex, 5>>> kspace_tensors_;

    fftw_plan fft_s_rspace_to_kspace = nullptr;
    fftw_plan fft_h_kspace_to_rspace = nullptr;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_FFT_H