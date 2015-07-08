// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_GLOBALS_H
#define JAMS_CORE_GLOBALS_H

#include <string>

#include "core/error.h"
#include "core/lattice.h"
#include "core/output.h"
#include "core/rand.h"
#include "core/sparsematrix.h"
#include "core/solver.h"
#include "core/physics.h"

#include "jblib/containers/array.h"
#include "jblib/containers/sparsematrix.h"

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>

GLOBAL cudaStream_t* cuda_streams;
#endif
GLOBAL Solver *solver;
GLOBAL Physics *physics_module;
GLOBAL Lattice lattice;
GLOBAL libconfig::Config config;  ///< Config object
GLOBAL Output output;
GLOBAL Random rng;
GLOBAL std::string seedname;
GLOBAL bool verbose_output_is_set;

namespace optimize {
    GLOBAL bool use_fft;
}

namespace globals {
  GLOBAL bool is_cuda_solver_used;

  GLOBAL int num_spins;
  GLOBAL int num_spins3;

  GLOBAL jblib::Array<double, 2> s;
  GLOBAL jblib::Array<double, 2> h;
  GLOBAL jblib::Array<double, 2> h_dipole;

  GLOBAL jblib::Array<float, 2> atom_pos;

  GLOBAL SparseMatrix<double> J1ij_t;  // bilinear tensor interactions

  GLOBAL jblib::Array<double, 5> wij; // general interaction matrix (for FFT)

  GLOBAL jblib::Array<fftw_complex, 4> sq;
  GLOBAL jblib::Array<fftw_complex, 4> hq;
  GLOBAL jblib::Array<fftw_complex, 5> wq;

  GLOBAL jblib::Array<double, 1> alpha;
  GLOBAL jblib::Array<double, 1> mus;
  GLOBAL jblib::Array<double, 1> gyro;
}  // namespace globals
#undef GLOBAL
#endif  // JAMS_CORE_GLOBALS_H
