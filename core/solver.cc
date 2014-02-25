// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/solver.h"

#include "core/consts.h"
#include "core/fields.h"
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

  time_step_ = idt*gamma_electron_si;

  ::output.write("Initialising solver (CPU)\n");

  ::output.write("  * Converting MAP to CSR\n");
  globals::J1ij_s.convertMAP2CSR();
  globals::J1ij_t.convertMAP2CSR();
  globals::J2ij_s.convertMAP2CSR();
  globals::J2ij_t.convertMAP2CSR();

  ::output.write("  * J1ij Scalar matrix memory (CSR): %f MB\n",
    globals::J1ij_s.calculateMemory());
  ::output.write("  * J1ij Tensor matrix memory (CSR): %f MB\n",
    globals::J1ij_t.calculateMemory());
  ::output.write("  * J2ij Scalar matrix memory (CSR): %f MB\n",
    globals::J2ij_s.calculateMemory());
  ::output.write("  * J2ij Tensor matrix memory (CSR): %f MB\n",
    globals::J2ij_t.calculateMemory());

  initialized_ = true;
}

void Solver::run() {
}

void Solver::compute_fields() {
  using namespace globals;
  int i, j;
  std::fill(h.data(), h.data()+num_spins3, 0.0);

  if (J1ij_s.nonZero() > 0) {
    compute_bilinear_scalar_interactions_csr(J1ij_s.valPtr(), J1ij_s.colPtr(), J1ij_s.ptrB(),
      J1ij_s.ptrE(), h);
  }
  if (J1ij_t.nonZero() > 0) {
    compute_bilinear_tensor_interactions_csr(J1ij_t.valPtr(), J1ij_t.colPtr(), J1ij_t.ptrB(),
      J1ij_t.ptrE(), h);
  }
  if (J2ij_s.nonZero() > 0) {
    compute_biquadratic_scalar_interactions_csr(J2ij_s.valPtr(), J2ij_s.colPtr(), J2ij_s.ptrB(),
      J2ij_s.ptrE(), h);
  }
  if (J2ij_t.nonZero() > 0) {
    compute_biquadratic_tensor_interactions_csr(J2ij_t.valPtr(), J2ij_t.colPtr(), J2ij_t.ptrB(),
      J2ij_t.ptrE(), h);
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
