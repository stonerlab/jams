// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_FIELDS_H
#define JAMS_CUDA_FIELDS_H

#include "core/cuda_sparse_types.h"

void cuda_device_compute_fields(
        const devDIA & J1ij_t_dev,
        const float *  sf_dev,
        const float *  mat_dev,
        float *        h_dev
);

#endif  // JAMS_CUDA_FIELDS_H
