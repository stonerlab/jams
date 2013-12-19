#ifndef __GLOBALS_H__
#define __GLOBALS_H__


#include "error.h"
#include "output.h"
#include "rand.h"
#include "lattice.h"

#include <containers/sparsematrix.h>
#include <containers/array.h>

//#include <libconfig.h++>

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif

GLOBAL Lattice lattice;

GLOBAL libconfig::Config config;  ///< Config object

GLOBAL Output output;

GLOBAL Random rng;

GLOBAL std::string seedname;

#include "sparsematrix.h"
#include "sparsematrix4d.h"

namespace globals
{
  GLOBAL int nspins;
  GLOBAL int nspins3;

  GLOBAL int globalSteps;

  GLOBAL double h_app[3];
  GLOBAL double globalTemperature;

  GLOBAL jblib::Array<double,2> s;
  GLOBAL jblib::Array<double,2> h;
  GLOBAL jblib::Array<double,2> w;
  
  GLOBAL jblib::Array<float,2> atom_pos;
#ifdef CUDA
  GLOBAL SparseMatrix<float> J1ij_s;  // bilinear scalar interactions
  GLOBAL SparseMatrix<float> J1ij_t;  // bilinear tensor interactions
  GLOBAL SparseMatrix<float> J2ij_s;  // biquadratic scalar interactions
  GLOBAL SparseMatrix<float> J2ij_t;  // biquadratic tensor interactions
  GLOBAL jblib::Sparsematrix<float,4> J4ijkl_s;  // fourspin scalar interactions
#else
  GLOBAL SparseMatrix<double> J1ij_s;  // bilinear scalar interactions
  GLOBAL SparseMatrix<double> J1ij_t;  // bilinear tensor interactions
  GLOBAL SparseMatrix<double> J2ij_s;  // biquadratic scalar interactions
  GLOBAL SparseMatrix<double> J2ij_t;  // biquadratic tensor interactions
  GLOBAL jblib::Sparsematrix<double,4> J4ijkl_s;  // fourspin scalar interactions
#endif

  GLOBAL jblib::Array<double,1> alpha;
  GLOBAL jblib::Array<double,1> mus;
  GLOBAL jblib::Array<double,1> gyro;
  GLOBAL jblib::Array<double,1> omega_corr;

} // namespace global



#undef GLOBAL

#endif // __GLOBALS_H_
