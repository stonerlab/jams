// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/globals.h"
#include "core/solver.h"
#include "core/cuda_solver.h"

#include "core/consts.h"

#include "core/utils.h"
#include "core/cuda_defs.h"

#include "solvers/cuda_heunllg.h"
#include "solvers/heunllg.h"
#include "solvers/metropolismc.h"
#include "core/cuda_sparsematrix.h"


void CudaSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

  Solver::initialize(argc, argv, idt);

  ::output.write("\ninitializing cuda solver class\n");

  ::output.write("  converting J1ij_t format from map to dia\n");
  J1ij_t.convertMAP2DIA();

  ::output.write("    memory usage (dia): %f MB\n", J1ij_t.calculateMemory());
  dev_J1ij_t_.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  ::output.write("    allocating on device\n");

//-----------------------------------------------------------------------------
// transfer sparse matrix to device - optionally converting double precision to
// single
//-----------------------------------------------------------------------------


  // allocate rows
  CUDA_CALL(cudaMalloc((void**)&dev_J1ij_t_.row, (J1ij_t.diags())*sizeof(int)));
  // allocate values
  CUDA_CALL(cudaMallocPitch((void**)&dev_J1ij_t_.val, &dev_J1ij_t_.pitch, (J1ij_t.rows())*sizeof(CudaFastFloat), J1ij_t.diags()));
  // copy rows
  CUDA_CALL(cudaMemcpy(dev_J1ij_t_.row, J1ij_t.dia_offPtr(), (size_t)((J1ij_t.diags())*(sizeof(int))), cudaMemcpyHostToDevice));
  // convert val array into CudaFastFloat which may be float or double
  std::vector<CudaFastFloat> float_values(J1ij_t.rows()*J1ij_t.diags(), 0.0);
  for (int i = 0; i < J1ij_t.rows()*J1ij_t.diags(); ++i) {
    float_values[i] = static_cast<CudaFastFloat>(J1ij_t.val(i));
  }
  // copy values
  CUDA_CALL(cudaMemcpy2D(dev_J1ij_t_.val, dev_J1ij_t_.pitch, &float_values[0], J1ij_t.rows()*sizeof(CudaFastFloat), J1ij_t.rows()*sizeof(CudaFastFloat), J1ij_t.diags(), cudaMemcpyHostToDevice));
  dev_J1ij_t_.pitch = dev_J1ij_t_.pitch/sizeof(CudaFastFloat);

//-----------------------------------------------------------------------------
// Transfer the the other arrays to the device
//-----------------------------------------------------------------------------

  ::output.write("  transfering arrays to device\n");

  // spin arrays
  jblib::Array<CudaFastFloat, 2> sf(num_spins, 3);
  for(int i = 0; i!=num_spins; ++i) {
    for(int j = 0; j!=3; ++j) {
      sf(i, j) = static_cast<CudaFastFloat>(s(i, j));
    }
  }
  dev_s_float_ = jblib::CudaArray<CudaFastFloat, 1>(sf);

  dev_s_        = jblib::CudaArray<double, 1>(s);
  dev_s_new_    = jblib::CudaArray<double, 1>(s);

  // field array
  jblib::Array<CudaFastFloat, 2> zero(num_spins, 3, 0.0);
  dev_h_        = jblib::CudaArray<CudaFastFloat, 1>(zero);

  // materials array
  jblib::Array<CudaFastFloat, 2> mat(num_spins, 4);
  jblib::Array<double, 1> sigma;

  sigma.resize(num_spins);
  for(int i = 0; i!=num_spins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (time_step_*mus(i)*mu_bohr_si) );
  }

  for(int i = 0; i!=num_spins; ++i){
    mat(i, 0) = static_cast<CudaFastFloat>(mus(i));
    mat(i, 1) = static_cast<CudaFastFloat>(gyro(i));
    mat(i, 2) = static_cast<CudaFastFloat>(alpha(i));
    mat(i, 3) = static_cast<CudaFastFloat>(sigma(i));
  }
  dev_mat_      = jblib::CudaArray<CudaFastFloat, 1>(mat);

  // anisotropy arrays
  jblib::Array<CudaFastFloat, 1> dz(num_spins);
  for (int i = 0; i < num_spins; ++i) {
    dz[i] = static_cast<CudaFastFloat>(globals::d2z[i]);
  }
  dev_d2z_ = jblib::CudaArray<CudaFastFloat, 1>(dz);

  for (int i = 0; i < num_spins; ++i) {
    dz[i] = static_cast<CudaFastFloat>(globals::d4z[i]);
  }
  dev_d4z_ = jblib::CudaArray<CudaFastFloat, 1>(dz);

  for (int i = 0; i < num_spins; ++i) {
    dz[i] = static_cast<CudaFastFloat>(globals::d6z[i]);
  }
  dev_d6z_ = jblib::CudaArray<CudaFastFloat, 1>(dz);

}

void CudaSolver::run() {
}

void CudaSolver::compute_fields() {
  using namespace globals;

  // bilinear interactions
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< dev_J1ij_t_.blocks, DIA_BLOCK_SIZE >>>
    (num_spins3, num_spins3, J1ij_t.diags(), dev_J1ij_t_.pitch, 1.0, 0.0,
     dev_J1ij_t_.row, dev_J1ij_t_.val, dev_s_float_.data(), dev_h_.data());
  }

  cuda_anisotropy_kernel<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>
  (num_spins, dev_d2z_.data(), dev_d4z_.data(), dev_d6z_.data(), dev_s_float_.data(), dev_h_.data());

  // anisotropy interactions
}

CudaSolver::~CudaSolver() {
  CUDA_CALL(cudaFree(dev_J1ij_t_.row));
  CUDA_CALL(cudaFree(dev_J1ij_t_.col));
  CUDA_CALL(cudaFree(dev_J1ij_t_.val));
}
