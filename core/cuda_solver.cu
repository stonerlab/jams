// Copyright 2014 Joseph Barker. All rights reserved.

#include <cublas.h>

#include "core/globals.h"
#include "core/solver.h"
#include "core/cuda_solver.h"
#include "core/cuda_solver_kernels.h"


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

  ::output.write("\ninitializing CUDA base solver\n");

  ::output.write("  initialising CUDA streams\n");

  is_cuda_solver_ = true;

  // if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
  //   jams_error("CudaSolver: CUBLAS initialization failed");
  // }

  dev_streams_ = new cudaStream_t[2];

  for (int i = 0; i < 2; ++i) {
    if (cudaStreamCreate(&dev_streams_[i]) != cudaSuccess){
      jams_error("Failed to create CUDA stream in CudaSolver");
    }
  }


//-----------------------------------------------------------------------------
// fourier transforms
//-----------------------------------------------------------------------------

  // for (int i = 0; i < 3; ++i) {
    // num_kpoints_[i] = globals::wij.size(i);
  // }

  // jblib::Vec3<int> num_hermitian_kpoints = num_kpoints_;
  // num_hermitian_kpoints.z = (num_kpoints_.z/2) + 1;

  // globals::wq.resize(num_kpoints_.x, num_kpoints_.y, (num_kpoints_.z/2)+1, 3, 3);

  // ::output.write("  kspace dimensions: %d %d %d\n", num_kpoints_.x, num_kpoints_.y, num_kpoints_.z);

  // ::output.write("  FFT planning\n");

  // perform the wij -> wq transformation on the host
  // fftw_plan interaction_fft_transform  = fftw_plan_many_dft_r2c(3, &num_kpoints_[0], 9, wij.data(),  NULL, 9, 1, wq.data(), NULL, 9, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
  // ::output.write("  FFT transform interaction matrix\n");
  // fftw_execute(interaction_fft_transform);

  // ::output.write("  FFT transfering arrays to device\n");

  // convert fftw_complex data into cufftDoubleComplex format and copy to the device
  // jblib::Array<cufftDoubleComplex, 5> convert_wq(num_hermitian_kpoints.x, num_hermitian_kpoints.y, num_hermitian_kpoints.z, 3, 3);

  // for (int i = 0; i < globals::wq.elements(); ++i) {
  //   convert_wq[i].x = globals::wq[i][0];
  //   convert_wq[i].y = globals::wq[i][1];
  // }
  // dev_wq_ = jblib::CudaArray<cufftDoubleComplex, 1>(convert_wq);

  // jblib::Array<double, 4> s3d(num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, 3, 0.0);
  // dev_s3d_ = jblib::CudaArray<double, 1>(s3d);
  // dev_h3d_ = jblib::CudaArray<double, 1>(s3d);

  // dev_sq_.resize(num_hermitian_kpoints.x*num_hermitian_kpoints.y*num_hermitian_kpoints.z*3);
  // dev_hq_.resize(num_hermitian_kpoints.x*num_hermitian_kpoints.y*num_hermitian_kpoints.z*3);

  // r_to_k_mapping_ = jblib::CudaArray<int, 1>(lattice.kspace_inv_map_);


  // if (cufftPlanMany(&spin_fft_forward_transform, 3, &num_kpoints_[0], &num_kpoints_[0], 3, 1, &num_hermitian_kpoints[0], 3, 1, CUFFT_D2Z, 3) != CUFFT_SUCCESS) {
  //   jams_error("CUFFT failure planning spin_fft_forward_transform");
  // }
  // if (cufftSetCompatibilityMode(spin_fft_forward_transform, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
  //   jams_error("CUFFT failure changing to compatability mode native for spin_fft_forward_transform");
  // }
  // if (cufftPlanMany(&field_fft_backward_transform, 3, &num_kpoints_[0], &num_hermitian_kpoints[0], 3, 1, &num_kpoints_[0], 3, 1, CUFFT_Z2D, 3) != CUFFT_SUCCESS) {
  //   jams_error("CUFFT failure planning field_fft_backward_transform");
  // }
  // if (cufftSetCompatibilityMode(field_fft_backward_transform, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
  //   jams_error("CUFFT failure changing to compatability mode native for field_fft_backward_transform");
  // }

//-----------------------------------------------------------------------------
// Transfer the the other arrays to the device
//-----------------------------------------------------------------------------

  ::output.write("  transfering array data to device\n");

  // spin arrays
  dev_s_        = jblib::CudaArray<double, 1>(s);
  dev_s_new_    = jblib::CudaArray<double, 1>(s);

  // field array
  jblib::Array<double, 2> zero(num_spins, 3, 0.0);
  dev_h_        = jblib::CudaArray<double, 1>(zero);

  // materials array
  jblib::Array<double, 2> mat(num_spins, 4);
  jblib::Array<double, 1> sigma(num_spins);

  // sigma.resize(num_spins);
  for(int i = 0; i < num_spins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (time_step_*mus(i)*mu_bohr_si) );
  }

  for(int i = 0; i < num_spins; ++i){
    mat(i, 0) = static_cast<double>(mus(i));
    mat(i, 1) = static_cast<double>(gyro(i));
    mat(i, 2) = static_cast<double>(alpha(i));
    mat(i, 3) = static_cast<double>(sigma(i));
  }
  dev_mat_      = jblib::CudaArray<double, 1>(mat);

  ::output.write("\n");
}

void CudaSolver::run() {
}

void CudaSolver::compute_fields() {
  using namespace globals;

  // zero the field array
  cudaMemsetAsync(dev_h_.data(), 0.0, num_spins3*sizeof(double), ::cuda_streams[0]);

  // if (optimize::use_fft) {
  //   cuda_realspace_to_kspace_mapping<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(dev_s_.data(), r_to_k_mapping_.data(), num_spins, num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, dev_s3d_.data());

  //   if (cufftExecD2Z(spin_fft_forward_transform, dev_s3d_.data(), dev_sq_.data()) != CUFFT_SUCCESS) {
  //     jams_error("CUFFT failure executing spin_fft_forward_transform");
  //   }

  //   const int convolution_size = num_kpoints_.x*num_kpoints_.y*((num_kpoints_.z/2)+1);
  //   const int real_size = num_kpoints_.x*num_kpoints_.y*num_kpoints_.z;

  //   cuda_fft_convolution<<<(convolution_size+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE >>>(convolution_size, real_size, dev_wq_.data(), dev_sq_.data(), dev_hq_.data());
  //   if (cufftExecZ2D(field_fft_backward_transform, dev_hq_.data(), dev_h3d_.data()) != CUFFT_SUCCESS) {
  //     jams_error("CUFFT failure executing field_fft_backward_transform");
  //   }

  //   cuda_kspace_to_realspace_mapping<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(dev_h3d_.data(), r_to_k_mapping_.data(), num_spins, num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, dev_h_.data());
  // }

  cudaStreamSynchronize(::cuda_streams[0]); // block until cudaMemsetAsync is finished

  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    (*it)->calculate_fields();
  }

  const double alpha = 1.0;
  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    cublasDaxpy(globals::num_spins3, alpha, (*it)->dev_ptr_field(), 1, dev_h_.data(), 1);
  }
}

CudaSolver::~CudaSolver() {
  for (int i = 0; i < 2; ++i) {
    cudaStreamDestroy(dev_streams_[i]);
  }
}
