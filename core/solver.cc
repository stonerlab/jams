// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/solver.h"

#include "core/consts.h"

#include "core/utils.h"
#include "core/globals.h"

#include "solvers/cuda_heunllg.h"
#include "solvers/cuda_srk4llg.h"
#include "solvers/heunllg.h"
#include "solvers/metropolismc.h"

void Solver::initialize(int argc, char **argv, double idt) {
  if (initialized_ == true) {
    jams_error("Solver is already initialized");
  }

  real_time_step_ = idt;
  time_step_ = idt*gamma_electron_si;

  ::output.write("\ninitialising base solver class\n");
  ::output.write("  converting interaction matrix J1ij format from MAP to CSR\n");
  globals::J1ij_t.convertMAP2CSR();
  ::output.write("  J1ij matrix memory (CSR): %f MB\n", globals::J1ij_t.calculateMemory());

  initialized_ = true;
}

void Solver::run() {
}

void Solver::compute_fields() {
  using namespace globals;
  int i, j;
  std::fill(h.data(), h.data()+num_spins3, 0.0);

//-----------------------------------------------------------------------------
// bilinear interactions
//-----------------------------------------------------------------------------
  if (J1ij_t.nonZero() > 0) {
    char transa[1] = {'N'};
    char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
    double one = 1.0;
    mkl_dcsrmv(transa, &num_spins3, &num_spins3, &one, matdescra, J1ij_t.valPtr(),
      J1ij_t.colPtr(), J1ij_t.ptrB(), J1ij_t.ptrE(), s.data(), &one, h.data());
#else
    jams_dcsrmv(transa, num_spins3, num_spins3, 1.0, matdescra, J1ij_t.valPtr(),
      J1ij_t.colPtr(), J1ij_t.ptrB(), J1ij_t.ptrE(), s.data(), 1.0, h.data());
#endif
  }

  // normalize by the gyroscopic factor
  for (i = 0; i < num_spins; ++i) {
    for (j = 0; j < 3; ++j) {
      h(i, j) = (h(i, j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
    }
  }
}

Solver* Solver::create(const std::string &solver_name) {

  if (capitalize(solver_name) == "HEUNLLG") {
    return new HeunLLGSolver;
  }

  if (capitalize(solver_name) == "METROPOLISMC") {
    return new MetropolisMCSolver;
  }

#ifdef CUDA
  if (capitalize(solver_name) == "CUDAHEUNLLG") {
    return new CUDAHeunLLGSolver;
  }

  if (capitalize(solver_name) == "CUDASRK4LLG") {
    return new CUDALLGSolverSRK4;
  }
#endif

  jams_error("Unknown solver '%s' selected.", solver_name.c_str());
  return NULL;
}
