// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H

#include <array>

#include <cublas_v2.h>
#include <cufft.h>

#include <libconfig.h++>

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

#include "strategy.h"

#include "jams/cuda/wrappers/stream.h"

class CudaDipoleHamiltonianFFT : public HamiltonianStrategy {
    public:
        CudaDipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size);

        ~CudaDipoleHamiltonianFFT();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
        void   calculate_fields(jblib::CudaArray<double, 1>& fields);
    private:
        bool debug_ = false;
        bool check_radius_   = true;
        bool check_symmetry_ = true;

        jblib::Array<fftw_complex, 5> generate_kspace_dipole_tensor(const int pos_i, const int pos_j);
        std::array<CudaStream, 4> dev_stream_;

        double                          r_cutoff_;
        double                          distance_tolerance_;

        jblib::CudaArray<double, 1>     h_;

        jblib::Vec3<int>                    kspace_size_;
        jblib::Vec3<int>                    kspace_padded_size_;
        jblib::CudaArray<cufftDoubleComplex, 1>   kspace_s_;
        jblib::CudaArray<cufftDoubleComplex, 1>   kspace_h_;

        std::vector<std::vector<jblib::CudaArray<cufftDoubleComplex, 1>>> kspace_tensors_;

        cufftHandle                     cuda_fft_s_rspace_to_kspace;
        cufftHandle                     cuda_fft_h_kspace_to_rspace;
};

#endif  // JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
