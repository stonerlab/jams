#include <jams/hamiltonian/cuda_cubic_anisotropy.h>

#include <jams/cuda/cuda_common.h>
#include <jams/core/globals.h>
#include <jams/hamiltonian/cubic_anisotropy.h>

#include "jams/cuda/cuda_array_reduction.h"
#include "jams/cuda/cuda_device_vector_ops.h"

__global__ void cuda_cubic_energy_kernel(const int num_spins, const unsigned * order,
                                         const jams::Real * magnitude, const jams::Real * axis1, const jams::Real * axis2, const jams::Real * axis3, const double * dev_s, jams::Real * dev_e) {
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real3 s{static_cast<jams::Real>(dev_s[base + 0]), static_cast<jams::Real>(dev_s[base + 1]), static_cast<jams::Real>(dev_s[base + 2])};
  jams::Real energy = 0.0;

  const jams::Real3 u{axis1[3*idx], axis1[3*idx+1], axis1[3*idx+2]};
  const jams::Real3 v{axis2[3*idx], axis2[3*idx+1], axis2[3*idx+2]};
  const jams::Real3 w{axis3[3*idx], axis3[3*idx+1], axis3[3*idx+2]};


  jams::Real su2 = dot(s, u) * dot(s, u);
  jams::Real sv2 = dot(s, v) * dot(s, v);
  jams::Real sw2 = dot(s, w) * dot(s, w);

  if (order[idx] == 1){
    energy += -magnitude[idx] * (su2 * sv2 + sv2 * sw2 + sw2 * su2);
  }

  if (order[idx] == 2){
    energy += -magnitude[idx] * su2 * sv2 * sw2;
  }

  dev_e[idx] = energy;
}

__global__ void cuda_cubic_field_kernel(const int num_spins, const unsigned * order,
                                        const jams::Real * magnitude, const jams::Real * axis1, const jams::Real * axis2, const jams::Real * axis3, const double * dev_s, jams::Real * dev_h) {
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real3 s{static_cast<jams::Real>(dev_s[base + 0]), static_cast<jams::Real>(dev_s[base + 1]), static_cast<jams::Real>(dev_s[base + 2])};
  jams::Real3 field{0, 0,0};

  const jams::Real3 u {axis1[3*idx], axis1[3*idx+1], axis1[3*idx+2]};
  const jams::Real3 v {axis2[3*idx], axis2[3*idx+1], axis2[3*idx+2]};
  const jams::Real3 w {axis3[3*idx], axis3[3*idx+1], axis3[3*idx+2]};


  jams::Real su = dot(s, u);
  jams::Real sv = dot(s, v);
  jams::Real sw = dot(s, w);

  jams::Real pre = 2 * magnitude[idx];

  if (order[idx] == 1) {
    field.x += pre * ( u.x * su * (sv*sv + sw*sw) + v.x * sv * (sw*sw + su*su) + w.x * sw * (su*su + sv*sv) );
    field.y += pre * ( u.y * su * (sv*sv + sw*sw) + v.y * sv * (sw*sw + su*su) + w.y * sw * (su*su + sv*sv) );
    field.z += pre * ( u.z * su * (sv*sv + sw*sw) + v.z * sv * (sw*sw + su*su) + w.z * sw * (su*su + sv*sv) );
  }

  if (order[idx] == 2) {
    field.x += pre * ( u.x * su * sv*sv * sw*sw + v.x * sv * sw*sw * su*su + w.x * sw * su*su * sv*sv );
    field.y += pre * ( u.y * su * sv*sv * sw*sw + v.y * sv * sw*sw * su*su + w.y * sw * su*su * sv*sv );
    field.z += pre * ( u.z * su * sv*sv * sw*sw + v.z * sv * sw*sw * su*su + w.z * sw * su*su * sv*sv );
  }

    dev_h[base + 0] = field.x;
    dev_h[base + 1] = field.y;
    dev_h[base + 2] = field.z;
}



CudaCubicAnisotropyHamiltonian::CudaCubicAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
    : CubicAnisotropyHamiltonian(settings, num_spins)
{}

jams::Real CudaCubicAnisotropyHamiltonian::calculate_total_energy(jams::Real time) {
    calculate_energies(time);
    return scalar_field_reduce_cuda(energy_, cuda_stream_.get());
}

void CudaCubicAnisotropyHamiltonian::calculate_energies(jams::Real time) {
    cuda_cubic_energy_kernel<<<(globals::num_spins + dev_blocksize_ - 1) / dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
            (globals::num_spins, order_.device_data(), magnitude_.device_data(), u_axes_.device_data(),
             v_axes_.device_data(), w_axes_.device_data(), globals::s.device_data(), field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}


void CudaCubicAnisotropyHamiltonian::calculate_fields(jams::Real time) {
        cuda_cubic_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
                (globals::num_spins, order_.device_data(), magnitude_.device_data(), u_axes_.device_data(),
                 v_axes_.device_data(), w_axes_.device_data(), globals::s.device_data(), field_.device_data());
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
