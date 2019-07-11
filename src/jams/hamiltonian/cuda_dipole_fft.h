// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H

#include <array>

#include <libconfig.h++>
#include <cublas_v2.h>
#include <cufft.h>

#include "jams/core/types.h"
#include "jams/hamiltonian/strategy.h"
#include "jams/cuda/cuda_stream.h"

class CudaDipoleHamiltonianFFT : public HamiltonianStrategy {
    public:
        CudaDipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size);

        ~CudaDipoleHamiltonianFFT();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies(jams::MultiArray<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jams::MultiArray<double, 2>& fields);
    private:
        bool debug_ = false;
        bool check_radius_   = true;
        bool check_symmetry_ = true;

        jams::MultiArray<Complex, 5> generate_kspace_dipole_tensor(const int pos_i, const int pos_j);

        double                          r_cutoff_;
        double                          distance_tolerance_;


        Vec3i                    kspace_size_;
        Vec3i                    kspace_padded_size_;
        jams::MultiArray<cufftDoubleComplex, 1>   kspace_s_;
        jams::MultiArray<cufftDoubleComplex, 1>   kspace_h_;

    std::array<CudaStream, 4> dev_stream_;
    jams::MultiArray<double, 2>     dipole_fields_;


    std::vector<std::vector<jams::MultiArray<cufftDoubleComplex, 1>>> kspace_tensors_;

        cufftHandle                     cuda_fft_s_rspace_to_kspace;
        cufftHandle                     cuda_fft_h_kspace_to_rspace;
};

#endif  // JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
