// Copyright 2014 Joseph Barker. All rights reserved.

#include <fftw3.h>

#include "core/solver.h"

#include "core/consts.h"

#include "core/utils.h"
#include "core/globals.h"

#include "solvers/cuda_heunllg.h"
#include "solvers/heunllg.h"
#include "solvers/metropolismc.h"
#include "solvers/constrainedmc.h"
#include "solvers/cuda_constrainedmc.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif

void Solver::initialize(int argc, char **argv, double idt) {
  using namespace globals;
  if (initialized_ == true) {
    jams_error("Solver is already initialized");
  }

  real_time_step_ = idt;
  time_step_ = idt*gamma_electron_si;


  // FFTW Planning
  // Real to complex transform means that just over half the data is stored. Hermitian
  // redundancy means that out[i] is the conjugate of out[n-i].
  // (http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data)

  globals::sq.resize(globals::wij.size(0), globals::wij.size(1), (globals::wij.size(2)/2)+1, 3);
  globals::hq.resize(globals::wij.size(0), globals::wij.size(1), (globals::wij.size(2)/2)+1, 3);
  globals::wq.resize(globals::wij.size(0), globals::wij.size(1), (globals::wij.size(2)/2)+1, 3, 3);

  const int kspace_dimensions[3] = {globals::wij.size(0), globals::wij.size(1), globals::wij.size(2)};

  ::output.write("kspace dimensions: %d %d %d", globals::wij.size(0), globals::wij.size(1), globals::wij.size(2));

  ::output.write("\nFFT planning\n");


  spin_fft_forward_transform   = fftw_plan_many_dft_r2c(3, kspace_dimensions, 3, s.data(),  NULL, 3, 1, sq.data(), NULL, 3, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
  field_fft_backward_transform = fftw_plan_many_dft_c2r(3, kspace_dimensions, 3, hq.data(), NULL, 3, 1, h_dipole.data(),  NULL, 3, 1, FFTW_ESTIMATE);
  interaction_fft_transform    = fftw_plan_many_dft_r2c(3, kspace_dimensions, 9, wij.data(),  NULL, 9, 1, wq.data(), NULL, 9, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

  ::output.write("\nFFT transform interaction matrix\n");

  fftw_execute(interaction_fft_transform);

  initialized_ = true;
}

void Solver::run() {
}

void Solver::compute_fields() {
  using namespace globals;
  int i, j, k, m, n, iend, jend, kend;
  std::fill(h.data(), h.data()+num_spins3, 0.0);

//-----------------------------------------------------------------------------
// dipole interactions
//-----------------------------------------------------------------------------

  fftw_execute(spin_fft_forward_transform);


  // perform convolution as multiplication in fourier space
  for (i = 0, iend = globals::wij.size(0); i < iend; ++i) {
    for (j = 0, jend = globals::wij.size(1); j < jend; ++j) {
      for (k = 0, kend = (globals::wij.size(2)/2)+1; k < kend; ++k) {
        for(m = 0; m < 3; ++m) {
          hq(i,j,k,m)[0] = 0.0; hq(i,j,k,m)[1] = 0.0;
          for(n = 0; n < 3; ++n) {
            hq(i,j,k,m)[0] = hq(i,j,k,m)[0] + ( wq(i,j,k,m,n)[0]*sq(i,j,k,n)[0]-wq(i,j,k,m,n)[1]*sq(i,j,k,n)[1] );
            hq(i,j,k,m)[1] = hq(i,j,k,m)[1] + ( wq(i,j,k,m,n)[0]*sq(i,j,k,n)[1]+wq(i,j,k,m,n)[1]*sq(i,j,k,n)[0] );
          }
        }
      }
    }
  }

  fftw_execute(field_fft_backward_transform);

    // normalise
  for (i = 0; i < num_spins3; ++i) {
    h_dipole[i] /= static_cast<double>(num_spins);
  }


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

//-----------------------------------------------------------------------------
// anisotropy interactions
//-----------------------------------------------------------------------------
  for (i = 0; i < num_spins; ++i) {
    h(i, 2) += d2z(i)*3.0*s(i,2) + d4z(i)*(17.5*s(i,2)*s(i,2)*s(i,2)-7.5*s(i,2)) + d6z(i)*(86.625*s(i,2)*s(i,2)*s(i,2)*s(i,2)*s(i,2) - 78.75*s(i,2)*s(i,2)*s(i,2) + 13.125*s(i,2));
  }

  // normalize by the gyroscopic factor
  for (i = 0; i < num_spins; ++i) {
    for (j = 0; j < 3; ++j) {
      h(i, j) = (h(i, j) + h_dipole(i,j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
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

  if (capitalize(solver_name) == "CONSTRAINEDMC") {
    return new ConstrainedMCSolver;
  }

#ifdef CUDA
  if (capitalize(solver_name) == "CUDAHEUNLLG") {
    return new CUDAHeunLLGSolver;
  }

  if (capitalize(solver_name) == "CUDACONSTRAINEDMC") {
    return new CudaConstrainedMCSolver;
  }
#endif

  jams_error("Unknown solver '%s' selected.", solver_name.c_str());
  return NULL;
}
