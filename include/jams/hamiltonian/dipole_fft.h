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

        ~DipoleHamiltonianFFT() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
    private:

        void calculate_nonlocal_field();

        double r_cutoff_;
        int k_cutoff_;

        jblib::Vec3<int> kspace_size_;
        jblib::Vec3<int> kspace_padded_size_;

        jblib::Array<double, 4> h_nonlocal_;
        jblib::Array<double, 4> s_nonlocal_;
        jblib::Array<double, 2> s_old_;

        jblib::Array<fftw_complex, 4> s_recip_;
        jblib::Array<fftw_complex, 4> h_recip_;
        jblib::Array<fftw_complex, 5> w_recip_;

        fftw_plan spin_fft_forward_transform_;
        fftw_plan field_fft_backward_transform_;
        fftw_plan interaction_fft_transform_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_FFT_H