// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include <cublas_v2.h>

#include "jams/core/solver.h"
#include "jams/containers/multiarray.h"

class CudaSolver : public Solver {
 public:
  CudaSolver() = default;
  ~CudaSolver() = default;

  void initialize(const libconfig::Setting& settings);
  virtual void run() = 0;

    void compute_fields();
};

#endif  // JAMS_CORE_CUDASOLVER_H
