// Copyright 2014 Joseph Barker. All rights reserved.

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

  dev_streams_ = new cudaStream_t[2];

  for (int i = 0; i < 2; ++i) {
    if (cudaStreamCreate(&dev_streams_[i]) != cudaSuccess){
      jams_error("Failed to create CUDA stream in CudaLangevinCothThermostat");
    }
  }

  ::output.write("  converting J1ij_t format from map to dia");
  J1ij_t.convertMAP2DIA();

  ::output.write("  estimated memory usage (dia): %f MB\n", J1ij_t.calculateMemory());
  dev_J1ij_t_.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  ::output.write("  allocating memory on device\n");

//-----------------------------------------------------------------------------
// fourier transforms
//-----------------------------------------------------------------------------

  for (int i = 0; i < 3; ++i) {
    num_kpoints_[i] = globals::wij.size(i);
  }

  jblib::Vec3<int> num_hermitian_kpoints = num_kpoints_;
  num_hermitian_kpoints.z = (num_kpoints_.z/2) + 1;

  globals::wq.resize(num_kpoints_.x, num_kpoints_.y, (num_kpoints_.z/2)+1, 3, 3);

  ::output.write("  kspace dimensions: %d %d %d\n", num_kpoints_.x, num_kpoints_.y, num_kpoints_.z);

  ::output.write("  FFT planning\n");

  // perform the wij -> wq transformation on the host
  fftw_plan interaction_fft_transform  = fftw_plan_many_dft_r2c(3, &num_kpoints_[0], 9, wij.data(),  NULL, 9, 1, wq.data(), NULL, 9, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
  ::output.write("  FFT transform interaction matrix\n");
  fftw_execute(interaction_fft_transform);

  ::output.write("  FFT transfering arrays to device\n");

  // convert fftw_complex data into cufftDoubleComplex format and copy to the device
  jblib::Array<cufftDoubleComplex, 5> convert_wq(num_hermitian_kpoints.x, num_hermitian_kpoints.y, num_hermitian_kpoints.z, 3, 3);

  for (int i = 0; i < globals::wq.elements(); ++i) {
    convert_wq[i].x = globals::wq[i][0];
    convert_wq[i].y = globals::wq[i][1];
  }
  dev_wq_ = jblib::CudaArray<cufftDoubleComplex, 1>(convert_wq);

  jblib::Array<double, 4> s3d(num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, 3, 0.0);
  dev_s3d_ = jblib::CudaArray<double, 1>(s3d);
  dev_h3d_ = jblib::CudaArray<double, 1>(s3d);

  dev_sq_.resize(num_hermitian_kpoints.x*num_hermitian_kpoints.y*num_hermitian_kpoints.z*3);
  dev_hq_.resize(num_hermitian_kpoints.x*num_hermitian_kpoints.y*num_hermitian_kpoints.z*3);

  r_to_k_mapping_ = jblib::CudaArray<int, 1>(lattice.kspace_inv_map_);


  if (cufftPlanMany(&spin_fft_forward_transform, 3, &num_kpoints_[0], &num_kpoints_[0], 3, 1, &num_hermitian_kpoints[0], 3, 1, CUFFT_D2Z, 3) != CUFFT_SUCCESS) {
    jams_error("CUFFT failure planning spin_fft_forward_transform");
  }
  if (cufftSetCompatibilityMode(spin_fft_forward_transform, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
    jams_error("CUFFT failure changing to compatability mode native for spin_fft_forward_transform");
  }
  if (cufftPlanMany(&field_fft_backward_transform, 3, &num_kpoints_[0], &num_hermitian_kpoints[0], 3, 1, &num_kpoints_[0], 3, 1, CUFFT_Z2D, 3) != CUFFT_SUCCESS) {
    jams_error("CUFFT failure planning field_fft_backward_transform");
  }
  if (cufftSetCompatibilityMode(field_fft_backward_transform, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
    jams_error("CUFFT failure changing to compatability mode native for field_fft_backward_transform");
  }


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

  ::output.write("  transfering array data to device\n");

  // spin arrays
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

  ::output.write("\n");
}

void CudaSolver::run() {
}

void CudaSolver::compute_fields() {
  using namespace globals;

  // zero the field array
  cudaMemsetAsync(dev_h_.data(), 0.0, num_spins3*sizeof(CudaFastFloat), ::cuda_streams[0]);

  if (optimize::use_fft) {
    cuda_realspace_to_kspace_mapping<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(dev_s_.data(), r_to_k_mapping_.data(), num_spins, num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, dev_s3d_.data());

    if (cufftExecD2Z(spin_fft_forward_transform, dev_s3d_.data(), dev_sq_.data()) != CUFFT_SUCCESS) {
      jams_error("CUFFT failure executing spin_fft_forward_transform");
    }

    const int convolution_size = num_kpoints_.x*num_kpoints_.y*((num_kpoints_.z/2)+1);
    const int real_size = num_kpoints_.x*num_kpoints_.y*num_kpoints_.z;

    cuda_fft_convolution<<<(convolution_size+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE >>>(convolution_size, real_size, dev_wq_.data(), dev_sq_.data(), dev_hq_.data());
    if (cufftExecZ2D(field_fft_backward_transform, dev_hq_.data(), dev_h3d_.data()) != CUFFT_SUCCESS) {
      jams_error("CUFFT failure executing field_fft_backward_transform");
    }

    cuda_kspace_to_realspace_mapping<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(dev_h3d_.data(), r_to_k_mapping_.data(), num_spins, num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, dev_h_.data());
  }

  cudaStreamSynchronize(::cuda_streams[0]); // block until cudaMemsetAsync is finished

  cuda_anisotropy_kernel<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE, 0, dev_streams_[1]>>>
  (num_spins, dev_d2z_.data(), dev_d4z_.data(), dev_d6z_.data(), dev_s_.data(), dev_h_.data());

  // bilinear interactions
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< dev_J1ij_t_.blocks, DIA_BLOCK_SIZE, 0, dev_streams_[0] >>>
    (num_spins3, num_spins3, J1ij_t.diags(), dev_J1ij_t_.pitch, 1.0, 1.0,
     dev_J1ij_t_.row, dev_J1ij_t_.val, dev_s_.data(), dev_h_.data());
  }

  // anisotropy interactions
}

CudaSolver::~CudaSolver() {
  CUDA_CALL(cudaFree(dev_J1ij_t_.row));
  CUDA_CALL(cudaFree(dev_J1ij_t_.col));
  CUDA_CALL(cudaFree(dev_J1ij_t_.val));

  for (int i = 0; i < 2; ++i) {
    cudaStreamDestroy(dev_streams_[i]);
  }
}
