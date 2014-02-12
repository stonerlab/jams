// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/cuda_fields.h"

#include <cublas.h>

#include "core/cuda_sparse.h"
#include "core/cuda_sparse_types.h"
#include "core/globals.h"


void cuda_device_compute_fields(
        const devDIA & J1ij_s_dev,
        const devDIA & J1ij_t_dev,
        const devDIA & J2ij_s_dev,
        const devDIA & J2ij_t_dev,
        const devCSR & J4ijkl_s_dev,
        const float *  sf_dev,
        const float *  r_dev,
        const float *  r_max_dev,
        const float *  mat_dev,
        const bool *   pbc_dev,
        float *        h_dev,
        float *        h_dipole_dev,
        const bool     dipole_toggle
){
    using namespace globals;

    // used to zero the first field which is calculated (i.e. not to use the
    // field from the last time step)
    float beta = 0.0;

    // Bilinear Scalar Fields
    if(J1ij_s.nonZero() > 0){

        bilinear_scalar_interaction_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>
            (num_spins, num_spins, J1ij_s.diags(), J1ij_s_dev.pitch, 1.0, beta,
             J1ij_s_dev.row, J1ij_s_dev.val, sf_dev, h_dev);

        beta = 1.0;
    }

    // Bilinear Tensor Fields
    if(J1ij_t.nonZero() > 0){

        spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>
            (num_spins3, num_spins3, J1ij_t.diags(), J1ij_t_dev.pitch, beta, 1.0,
             J1ij_t_dev.row, J1ij_t_dev.val, sf_dev, h_dev);

        beta = 1.0;

    }

    // Biquadratic Scalar Fields
    if(J2ij_s.nonZero() > 0){

        biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>
            (num_spins, num_spins, J2ij_s.diags(), J2ij_s_dev.pitch, 2.0, beta,
             J2ij_s_dev.row, J2ij_s_dev.val, sf_dev, h_dev);

        beta = 1.0;

    }

    // Biquadratic Tensor Fields
    if(J2ij_t.nonZero() > 0){

        spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>
            (num_spins3, num_spins3, J2ij_t.diags(), J2ij_t_dev.pitch, 2.0, beta,
             J2ij_t_dev.row, J2ij_t_dev.val, sf_dev, h_dev);

        beta = 1.0;

    }

    // Fourspin Scalar Fields
    if(J4ijkl_s.nonZeros() > 0){

        fourspin_scalar_interaction_csr_kernel<<< J4ijkl_s_dev.blocks, CSR_4D_BLOCK_SIZE>>>
            (num_spins, num_spins, 1.0, beta,
             J4ijkl_s_dev.pointers, J4ijkl_s_dev.coords, J4ijkl_s_dev.val, sf_dev, h_dev);

        beta = 1.0;

    }

    // Dipole-Dipole Fields
    const float dipole_omega = 0.00092740096; // (muB*mu0/4pi)/nm^3

    // We only really need to update the dipole field for the first integration
    // step in any scheme. This toggle is used to determine this. If the field
    // is not updated then the cached result is still added below
    if( dipole_toggle == true ){
        if(globalSteps%100 == 0){
            const int nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;
            bruteforce_dipole_interaction_kernel<<<nblocks, BLOCKSIZE >>>
                (dipole_omega, 0.0, sf_dev, mat_dev, h_dipole_dev,
                 r_dev, r_max_dev, pbc_dev, num_spins);
        }
    }

    // add cached dipole-dipole field
    cublasSaxpy(num_spins3, 1.0, h_dipole_dev, 1, h_dev, 1);

}
