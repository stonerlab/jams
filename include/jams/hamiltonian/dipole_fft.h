// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_DIPOLE_FFT_H

#include <fftw3.h>
#include <libconfig.h++>

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

#include "jams/hamiltonian/strategy.h"


class DipoleHamiltonianFFT : public HamiltonianStrategy {
    public:
        DipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size);

        ~DipoleHamiltonianFFT();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
    private:

        void calculate_nonlocal_field();

        jblib::Array<fftw_complex, 5> generate_kspace_dipole_tensor(const int pos_i, const int pos_j);

        double                          r_cutoff_;
        int                             k_cutoff_;

        jblib::Array<double, 4>         rspace_s_;
        jblib::Array<double, 4>         rspace_h_;
        jblib::Array<double, 2>         h_;

        jblib::Vec3<int>                kspace_size_;
        jblib::Vec3<int>                kspace_padded_size_;
        jblib::Array<fftw_complex, 4>   kspace_s_;
        jblib::Array<fftw_complex, 4>   kspace_h_;

        std::vector<std::vector<jblib::Array<fftw_complex, 5>>> kspace_tensors_;

        fftw_plan                       fft_s_rspace_to_kspace;
        fftw_plan                       fft_h_kspace_to_rspace;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_FFT_H