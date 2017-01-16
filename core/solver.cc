// Copyright 2014 Joseph Barker. All rights reserved.

#include <fftw3.h>

#include "core/blas.h"
#include "core/solver.h"
#include "core/hamiltonian.h"
#include "core/monitor.h"
#include "core/consts.h"

#include "core/utils.h"
#include "core/globals.h"

#include "solvers/cuda_heunllg.h"
#include "solvers/heunllg.h"
#include "solvers/metropolismc.h"
#include "solvers/constrainedmc.h"
#include "solvers/monte-carlo-wolff.h"
#include "solvers/cuda_constrainedmc.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif

Solver::~Solver() {
  for (int i = 0, iend = monitors_.size(); i < iend; ++i) {
    delete monitors_[i];
  }
}

//---------------------------------------------------------------------

void Solver::initialize(int argc, char **argv, double idt) {
  using namespace globals;
  if (initialized_ == true) {
    jams_error("Solver is already initialized");
  }

  real_time_step_ = idt;
  time_step_ = idt*kGyromagneticRatio;


  // FFTW Planning
  // Real to complex transform means that just over half the data is stored. Hermitian
  // redundancy means that out[i] is the conjugate of out[n-i].
  // (http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data)

  // globals::sq.resize(globals::wij.size(0), globals::wij.size(1), (globals::wij.size(2)/2)+1, 3);
  // globals::hq.resize(globals::wij.size(0), globals::wij.size(1), (globals::wij.size(2)/2)+1, 3);
  // globals::wq.resize(globals::wij.size(0), globals::wij.size(1), (globals::wij.size(2)/2)+1, 3, 3);

  // const int kspace_dimensions[3] = {globals::wij.size(0), globals::wij.size(1), globals::wij.size(2)};

  // ::output.write("kspace dimensions: %d %d %d", globals::wij.size(0), globals::wij.size(1), globals::wij.size(2));

  // ::output.write("\nFFT planning\n");


  // spin_fft_forward_transform   = fftw_plan_many_dft_r2c(3, kspace_dimensions, 3, s.data(),  NULL, 3, 1, sq.data(), NULL, 3, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
  // field_fft_backward_transform = fftw_plan_many_dft_c2r(3, kspace_dimensions, 3, hq.data(), NULL, 3, 1, h_dipole.data(),  NULL, 3, 1, FFTW_ESTIMATE);
  // interaction_fft_transform    = fftw_plan_many_dft_r2c(3, kspace_dimensions, 9, wij.data(),  NULL, 9, 1, wq.data(), NULL, 9, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

  // ::output.write("\nFFT transform interaction matrix\n");

  // fftw_execute(interaction_fft_transform);

  initialized_ = true;
}

//---------------------------------------------------------------------

void Solver::run() {
}

//---------------------------------------------------------------------

void Solver::compute_fields() {
  using namespace globals;

  // zero the effective field array
  std::fill(h.data(), h.data()+num_spins3, 0.0);

  // calculate each hamiltonian term's fields
  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    (*it)->calculate_fields();
  }

  // sum hamiltonian field contributions into effective field
  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    cblas_daxpy(num_spins3, 1.0, (*it)->ptr_field(), 1, h.data(), 1);
  }
}

Solver* Solver::create(const std::string &solver_name) {

  if (capitalize(solver_name) == "LLG-HEUN-CPU" || capitalize(solver_name) == "HEUNLLG") {
    return new HeunLLGSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-METROPOLIS-CPU" || capitalize(solver_name) == "METROPOLISMC") {
    return new MetropolisMCSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-CONSTRAINED-CPU" || capitalize(solver_name) == "CONSTRAINEDMC") {
    return new ConstrainedMCSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-WOLFF-CPU") {
    return new MonteCarloWolffSolver;
  }

#ifdef CUDA
  if (capitalize(solver_name) == "LLG-HEUN-GPU" || capitalize(solver_name) == "CUDAHEUNLLG") {
    return new CUDAHeunLLGSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-CONSTRAINED-GPU" || capitalize(solver_name) == "CUDACONSTRAINEDMC") {
    return new CudaConstrainedMCSolver;
  }
#endif

  jams_error("Unknown solver '%s' selected.", solver_name.c_str());
  return NULL;
}

//---------------------------------------------------------------------

void Solver::register_physics_module(Physics* package) {
    physics_module_ = package;
}

//---------------------------------------------------------------------

void Solver::update_physics_module() {
    physics_module_->update(iteration_, time(), time_step_);
}

//---------------------------------------------------------------------

void Solver::register_monitor(Monitor* monitor) {
  monitors_.push_back(monitor);
}

//---------------------------------------------------------------------

void Solver::register_hamiltonian(Hamiltonian* hamiltonian) {
  hamiltonians_.push_back(hamiltonian);
}

//---------------------------------------------------------------------

void Solver::notify_monitors() {
  for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
    if((*it)->is_updating(iteration_)){
      (*it)->update(this);
    }
  }
}

bool Solver::is_converged() {
  for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
    if((*it)->is_converged()){
      return true;
    }
  }
  return false;
}
