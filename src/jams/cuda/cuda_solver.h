// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include "jams/core/solver.h"

class CudaSolver : public Solver {
 public:
  bool is_cuda_solver() const override { return true; }
  void compute_fields() override;
};

#endif  // JAMS_CORE_CUDASOLVER_H
