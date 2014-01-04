#ifndef JAMS_CUDA_FIELDS_H
#define JAMS_CUDA_FIELDS_H

#include "core/cuda_sparse_types.h"

void CUDACalculateFields(
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
);

#endif  // JAMS_CUDA_FIELDS_H