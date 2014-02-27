// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/cuda_fields.h"

#include <cublas.h>

#include "core/cuda_sparse.h"
#include "core/cuda_sparse_types.h"
#include "core/globals.h"


void cuda_device_compute_fields(
        const devDIA & J1ij_t_dev,
        const float *  sf_dev,
        const float *  mat_dev,
        float *        h_dev
){
    using namespace globals;

    // used to zero the first field which is calculated (i.e. not to use the
    // field from the last time step)
    float beta = 0.0;

    // Bilinear Tensor Fields
    if(J1ij_t.nonZero() > 0){

        spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>
            (num_spins3, num_spins3, J1ij_t.diags(), J1ij_t_dev.pitch, beta, 1.0,
             J1ij_t_dev.row, J1ij_t_dev.val, sf_dev, h_dev);

        beta = 1.0;

    }
}
