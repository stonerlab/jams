#include "jams/monitors/spectrum_base.h"

#if HAS_CUDA

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_stream.h"
#include "jams/helpers/consts.h"
#include "jams/monitors/cuda_grouped_spin_reduction.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using CmplxStored = SpectrumBaseMonitor::CmplxStored;

__device__ inline cufftDoubleComplex add_z(const cufftDoubleComplex a, const cufftDoubleComplex b)
{
  return {a.x + b.x, a.y + b.y};
}

__device__ inline cufftDoubleComplex sub_z(const cufftDoubleComplex a, const cufftDoubleComplex b)
{
  return {a.x - b.x, a.y - b.y};
}

__device__ inline cufftDoubleComplex mul_z(const cufftDoubleComplex a, const cufftDoubleComplex b)
{
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ inline cufftDoubleComplex scale_z(const cufftDoubleComplex a, const double b)
{
  return {a.x * b, a.y * b};
}

__device__ inline cufftDoubleComplex conj_z(const cufftDoubleComplex a)
{
  return {a.x, -a.y};
}

__device__ inline double norm_z(const cufftDoubleComplex a)
{
  return a.x * a.x + a.y * a.y;
}

__device__ inline float2 to_float2(const cufftDoubleComplex z)
{
  return {static_cast<float>(z.x), static_cast<float>(z.y)};
}

__device__ inline cufftDoubleComplex from_float2(const float2 z)
{
  return {static_cast<double>(z.x), static_cast<double>(z.y)};
}

__device__ inline double atomic_add_double(double* address, const double value)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  auto old = *address_as_ull;
  unsigned long long int assumed;

  do
  {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
#else
  return atomicAdd(address, value);
#endif
}

__global__ void pack_dense_spins_kernel(
    const int num_values,
    const int* site_map,
    const double* spins,
    double* packed)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_values)
  {
    return;
  }

  const int slot = idx / 3;
  const int component = idx - 3 * slot;
  const int site = site_map[slot];
  packed[idx] = (site >= 0) ? spins[3 * site + component] : 0.0;
}

__global__ void scale_complex_kernel(
    const int n,
    const double scale,
    cufftDoubleComplex* values)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
  {
    return;
  }
  values[idx].x *= scale;
  values[idx].y *= scale;
}

__global__ void extract_compact_sample_kernel(
    const int num_basis,
    const int num_k,
    const int stored_channels,
    const int output_offset,
    const int kspace_y,
    const int kspace_z_r2c,
    const bool needs_local_frame,
    const bool scale_to_physical_spin,
    const double electron_g,
    const cufftDoubleComplex* sk_grid,
    const int* k_indices,
    const cufftDoubleComplex* basis_phase_factors,
    const cufftDoubleComplex* channel_weights,
    const jams::Real* moments,
    float2* out)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_basis * num_k;
  if (idx >= total)
  {
    return;
  }

  const int k = idx % num_k;
  const int a = idx / num_k;
  const int kx = k_indices[4 * k + 0];
  const int ky = k_indices[4 * k + 1];
  const int kz = k_indices[4 * k + 2];
  const bool conjugate = k_indices[4 * k + 3] != 0;
  const int k_flat = (kx * kspace_y + ky) * kspace_z_r2c + kz;
  const cufftDoubleComplex phase = basis_phase_factors[a * num_k + k];

  cufftDoubleComplex spin[3];
  for (int c = 0; c < 3; ++c)
  {
    cufftDoubleComplex value = sk_grid[(k_flat * num_basis + a) * 3 + c];
    if (conjugate)
    {
      value = conj_z(value);
    }
    spin[c] = mul_z(phase, value);
  }

  const int base = output_offset + (a * num_k + k) * stored_channels;
  if (needs_local_frame)
  {
    out[base + 0] = to_float2(spin[0]);
    out[base + 1] = to_float2(spin[1]);
    out[base + 2] = to_float2(spin[2]);
    return;
  }

  const double spin_scale = scale_to_physical_spin
      ? static_cast<double>(moments[a]) / electron_g
      : 1.0;
  for (int out_c = 0; out_c < stored_channels; ++out_c)
  {
    cufftDoubleComplex mapped = {0.0, 0.0};
    for (int xyz = 0; xyz < 3; ++xyz)
    {
      mapped = add_z(mapped, mul_z(channel_weights[3 * out_c + xyz], spin[xyz]));
    }
    out[base + out_c] = to_float2(scale_z(mapped, spin_scale));
  }
}

__global__ void direct_spatial_sum_kernel(
    const int num_direct_sites,
    const int num_k,
    const double spatial_scale,
    const int* site_indices,
    const int* basis_indices,
    const double* positions_frac,
    const double* window_weights,
    const double* k_hkl,
    const double* spins,
    cufftDoubleComplex* direct_sum)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_direct_sites * num_k;
  if (idx >= total)
  {
    return;
  }

  const int k = idx % num_k;
  const int direct_site = idx / num_k;
  const int site = site_indices[direct_site];
  const int basis = basis_indices[direct_site];
  const double* r = positions_frac + 3 * direct_site;
  const double* q = k_hkl + 3 * k;
  const double q_dot_r = q[0] * r[0] + q[1] * r[1] + q[2] * r[2];
  const double angle = -2.0 * M_PI * q_dot_r;
  const double phase_re = cos(angle);
  const double phase_im = sin(angle);
  const double scale = spatial_scale * window_weights[direct_site];

  for (int c = 0; c < 3; ++c)
  {
    const double value = scale * spins[3 * site + c];
    cufftDoubleComplex* out = direct_sum + (basis * num_k + k) * 3 + c;
    atomic_add_double(&out->x, value * phase_re);
    atomic_add_double(&out->y, value * phase_im);
  }
}

__global__ void map_direct_sum_sample_kernel(
    const int num_basis,
    const int num_k,
    const int stored_channels,
    const int output_offset,
    const bool needs_local_frame,
    const bool scale_to_physical_spin,
    const double electron_g,
    const cufftDoubleComplex* direct_sum,
    const cufftDoubleComplex* channel_weights,
    const jams::Real* moments,
    float2* out)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_basis * num_k;
  if (idx >= total)
  {
    return;
  }

  const int k = idx % num_k;
  const int a = idx / num_k;
  const cufftDoubleComplex* spin = direct_sum + (a * num_k + k) * 3;
  const int base = output_offset + (a * num_k + k) * stored_channels;

  if (needs_local_frame)
  {
    out[base + 0] = to_float2(spin[0]);
    out[base + 1] = to_float2(spin[1]);
    out[base + 2] = to_float2(spin[2]);
    return;
  }

  const double spin_scale = scale_to_physical_spin
      ? static_cast<double>(moments[a]) / electron_g
      : 1.0;
  for (int out_c = 0; out_c < stored_channels; ++out_c)
  {
    cufftDoubleComplex mapped = {0.0, 0.0};
    for (int xyz = 0; xyz < 3; ++xyz)
    {
      mapped = add_z(mapped, mul_z(channel_weights[3 * out_c + xyz], spin[xyz]));
    }
    out[base + out_c] = to_float2(scale_z(mapped, spin_scale));
  }
}

__global__ void compute_rotations_kernel(
    const int num_basis,
    const int periodogram_length,
    const int ring_offset,
    const double* basis_mag_ring,
    const double* window,
    double* rotations)
{
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_basis)
  {
    return;
  }

  double n_hat[3] = {0.0, 0.0, 0.0};
  for (int t = 0; t < periodogram_length; ++t)
  {
    const int physical_t = (ring_offset + t) % periodogram_length;
    const double w = window[t];
    const double* mag = basis_mag_ring + (physical_t * num_basis + a) * 3;
    n_hat[0] += w * mag[0];
    n_hat[1] += w * mag[1];
    n_hat[2] += w * mag[2];
  }

  double n_norm = sqrt(n_hat[0] * n_hat[0] + n_hat[1] * n_hat[1] + n_hat[2] * n_hat[2]);
  double* R = rotations + 9 * a;
  if (n_norm <= 0.0)
  {
    R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
    R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
    R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
    return;
  }

  n_hat[0] /= n_norm;
  n_hat[1] /= n_norm;
  n_hat[2] /= n_norm;

  const double ax = fabs(n_hat[0]);
  const double ay = fabs(n_hat[1]);
  const double az = fabs(n_hat[2]);
  double r[3] = {1.0, 0.0, 0.0};
  double a_min = ax;
  if (ay < a_min)
  {
    r[0] = 0.0; r[1] = 1.0; r[2] = 0.0;
    a_min = ay;
  }
  if (az < a_min)
  {
    r[0] = 0.0; r[1] = 0.0; r[2] = 1.0;
  }

  const double dot_rn = r[0] * n_hat[0] + r[1] * n_hat[1] + r[2] * n_hat[2];
  double e1[3] = {
      r[0] - dot_rn * n_hat[0],
      r[1] - dot_rn * n_hat[1],
      r[2] - dot_rn * n_hat[2]};
  const double e1_norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  if (e1_norm <= 0.0)
  {
    R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
    R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
    R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
    return;
  }
  e1[0] /= e1_norm;
  e1[1] /= e1_norm;
  e1[2] /= e1_norm;

  const double e2[3] = {
      n_hat[1] * e1[2] - n_hat[2] * e1[1],
      n_hat[2] * e1[0] - n_hat[0] * e1[2],
      n_hat[0] * e1[1] - n_hat[1] * e1[0]};

  R[0] = e1[0];    R[1] = e1[1];    R[2] = e1[2];
  R[3] = e2[0];    R[4] = e2[1];    R[5] = e2[2];
  R[6] = n_hat[0]; R[7] = n_hat[1]; R[8] = n_hat[2];
}

__global__ void prepare_time_fft_input_kernel(
    const int num_basis,
    const int num_k,
    const int stored_channels,
    const int output_channels,
    const int periodogram_length,
    const int ring_offset,
    const int kpoint_index,
    const bool needs_local_frame,
    const bool scale_to_physical_spin,
    const double electron_g,
    const float2* time_series,
    const double* rotations,
    const cufftDoubleComplex* channel_weights,
    const jams::Real* moments,
    const double* window,
    cufftDoubleComplex* scratch)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_basis * output_channels;
  if (idx >= total)
  {
    return;
  }

  const int c = idx % output_channels;
  const int a = idx / output_channels;
  const double time_norm = 1.0 / static_cast<double>(periodogram_length);
  const double spin_scale = scale_to_physical_spin
      ? static_cast<double>(moments[a]) / electron_g
      : 1.0;
  cufftDoubleComplex mean = {0.0, 0.0};

  for (int t = 0; t < periodogram_length; ++t)
  {
    const int physical_t = (ring_offset + t) % periodogram_length;
    const float2* stored = time_series
        + (((physical_t * num_basis + a) * num_k + kpoint_index) * stored_channels);

    cufftDoubleComplex value = {0.0, 0.0};
    if (needs_local_frame)
    {
      cufftDoubleComplex s[3] = {
          from_float2(stored[0]),
          from_float2(stored[1]),
          from_float2(stored[2])};
      const double* R = rotations + 9 * a;
      cufftDoubleComplex rotated[3];
      for (int row = 0; row < 3; ++row)
      {
        rotated[row] = add_z(
            add_z(scale_z(s[0], R[3 * row + 0]), scale_z(s[1], R[3 * row + 1])),
            scale_z(s[2], R[3 * row + 2]));
      }
      for (int xyz = 0; xyz < 3; ++xyz)
      {
        value = add_z(value, mul_z(channel_weights[3 * c + xyz], rotated[xyz]));
      }
      value = scale_z(value, spin_scale);
    }
    else
    {
      value = from_float2(stored[c]);
    }
    mean = add_z(mean, scale_z(value, time_norm));
    scratch[(a * output_channels + c) * periodogram_length + t] = value;
  }

  for (int t = 0; t < periodogram_length; ++t)
  {
    const cufftDoubleComplex centered = sub_z(
        scratch[(a * output_channels + c) * periodogram_length + t],
        mean);
    scratch[(a * output_channels + c) * periodogram_length + t] =
        scale_z(centered, time_norm * window[t]);
  }
}

__global__ void accumulate_magnon_power_kernel(
    const int num_basis,
    const int output_channels,
    const int periodogram_length,
    const int num_frequencies,
    const int num_k,
    const int kpoint_index,
    const double taper_weight,
    const cufftDoubleComplex* scratch,
    double* cumulative)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_frequencies * output_channels;
  if (idx >= total)
  {
    return;
  }

  const int c = idx % output_channels;
  const int f = idx / output_channels;
  double sum = 0.0;
  for (int a = 0; a < num_basis; ++a)
  {
    sum += norm_z(scratch[(a * output_channels + c) * periodogram_length + f]);
  }
  cumulative[(f * num_k + kpoint_index) * 3 + c] += taper_weight * sum;
}

__global__ void accumulate_tapered_spectrum_kernel(
    const int total,
    const double taper_weight,
    const cufftDoubleComplex* scratch,
    cufftDoubleComplex* complex_sum,
    double* power_sum)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
  {
    return;
  }

  const cufftDoubleComplex value = scratch[idx];
  complex_sum[idx] = add_z(complex_sum[idx], scale_z(value, taper_weight));
  power_sum[idx] += taper_weight * norm_z(value);
}

__global__ void reconstruct_tapered_spectrum_kernel(
    const int total,
    const cufftDoubleComplex* complex_sum,
    const double* power_sum,
    cufftDoubleComplex* scratch)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
  {
    return;
  }

  constexpr double kPhaseEpsilon = 1e-30;
  const cufftDoubleComplex mean = complex_sum[idx];
  const double mean_power = fmax(0.0, power_sum[idx]);
  const double mean_abs = sqrt(norm_z(mean));
  if (mean_abs > kPhaseEpsilon)
  {
    scratch[idx] = scale_z(mean, sqrt(mean_power) / mean_abs);
  }
  else
  {
    scratch[idx] = {0.0, 0.0};
  }
}

template<typename T>
void copy_vector_to_device_only(jams::MultiArray<T, 1>& target, const std::vector<T>& values)
{
  target.resize(values.size());
  if (values.empty())
  {
    return;
  }
  auto host = target.mutable_host_span();
  std::copy(values.begin(), values.end(), host.begin());
  target.device_data();
  target.release_stale_host();
}

template<typename T>
void copy_matrix_to_device_only(jams::MultiArray<T, 2>& target, const std::vector<T>& values, const int n0, const int n1)
{
  target.resize(n0, n1);
  if (values.empty())
  {
    return;
  }
  auto host = target.mutable_host_span();
  std::copy(values.begin(), values.end(), host.begin());
  target.device_data();
  target.release_stale_host();
}

class SpectrumCudaBackend final : public SpectrumBaseMonitor::CudaBackend {
public:
  SpectrumCudaBackend(
      const Lattice& lattice,
      const std::vector<jams::HKLIndex>& k_points,
      const jams::MultiArray<jams::ComplexHi, 2>& basis_phase_factors,
      const bool use_direct_sum,
      const std::vector<SpectrumBaseMonitor::DirectSumSite>& direct_sum_sites,
      const double direct_sum_spatial_scale)
      : num_basis_(lattice.num_basis_sites()),
        num_spins_(globals::num_spins),
        grid_size_(lattice.size()),
        padded_size_(lattice.kspace_size()),
        kspace_z_r2c_(padded_size_[2] / 2 + 1),
        num_k_points_(static_cast<int>(k_points.size())),
        use_direct_sum_(use_direct_sum),
        direct_sum_spatial_scale_(direct_sum_spatial_scale),
        use_dense_input_(lattice.has_cropping() || grid_size_ != padded_size_)
  {
    if (use_direct_sum_)
    {
      initialise_direct_sum_buffers(k_points, direct_sum_sites);
    }
    else
    {
      initialise_k_indices(k_points);
      initialise_basis_phase_factors(basis_phase_factors);
      initialise_spatial_buffers(lattice);
      initialise_spatial_plan();
    }
    initialise_basis_reduction(lattice);
  }

  ~SpectrumCudaBackend() override
  {
    if (spatial_plan_)
    {
      cufftDestroy(spatial_plan_);
    }
    if (time_plan_)
    {
      cufftDestroy(time_plan_);
    }
  }

  std::size_t spatial_memory_bytes() const override
  {
    return spatial_memory_bytes_;
  }

  std::size_t free_device_memory_bytes() const override
  {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CHECK_CUDA_STATUS(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
  }

  std::size_t estimate_time_fft_memory_bytes(
      const int periodogram_length,
      const int num_basis_atoms,
      const int num_k_points,
      const int stored_channels,
      const int output_channels,
      const int num_frequencies,
      const int multitaper_count,
      const bool needs_local_frame,
      const bool use_multitaper,
      const bool needs_magnon_accumulation,
      const bool needs_frequency_slices) const override
  {
    const auto bytes_time_series =
        bytes_count<float2>(periodogram_length, num_basis_atoms, num_k_points, stored_channels);
    const auto bytes_scratch =
        bytes_count<cufftDoubleComplex>(num_basis_atoms, output_channels, periodogram_length);
    const auto bytes_cumulative = needs_magnon_accumulation
        ? bytes_count<double>(num_frequencies, num_k_points, 3)
        : 0;
    const auto bytes_basis_mag = needs_local_frame
        ? bytes_count<double>(periodogram_length, num_basis_atoms, 3)
        : 0;
    const auto bytes_rotations = needs_local_frame
        ? bytes_count<double>(num_basis_atoms, 9)
        : 0;
    const auto bytes_windows = use_multitaper
        ? bytes_count<double>(multitaper_count, periodogram_length)
        : bytes_count<double>(periodogram_length);
    const auto bytes_frequency_taper_sum = (needs_frequency_slices && use_multitaper)
        ? bytes_count<cufftDoubleComplex>(num_basis_atoms, output_channels, periodogram_length)
        : 0;
    const auto bytes_frequency_taper_power_sum = (needs_frequency_slices && use_multitaper)
        ? bytes_count<double>(num_basis_atoms, output_channels, periodogram_length)
        : 0;
    return bytes_time_series + bytes_scratch + bytes_cumulative
        + bytes_basis_mag + bytes_rotations + bytes_windows
        + bytes_frequency_taper_sum + bytes_frequency_taper_power_sum;
  }

  void reset_time_storage() override
  {
    time_series_.clear();
    basis_mag_ring_.clear();
    rotations_.clear();
    time_scratch_.clear();
    cumulative_magnon_.clear();
    frequency_taper_sum_.clear();
    frequency_taper_power_sum_.clear();
    periodogram_window_.clear();
    multitaper_windows_.clear();
    multitaper_weights_.clear();
    ring_offset_ = 0;
    time_configured_ = false;
    magnon_accumulation_configured_ = false;
    frequency_slices_configured_ = false;
    if (time_plan_)
    {
      cufftDestroy(time_plan_);
      time_plan_ = 0;
    }
  }

  void configure_time_storage(
      const int periodogram_length,
      const int num_basis_atoms,
      const int num_k_points,
      const int stored_channels,
      const int output_channels,
      const int num_frequencies,
      const bool keep_negative_frequencies,
      const bool needs_local_frame,
      const bool needs_magnon_accumulation,
      const bool needs_frequency_slices,
      const bool use_multitaper) override
  {
    periodogram_length_ = periodogram_length;
    stored_channels_ = stored_channels;
    output_channels_ = output_channels;
    num_frequencies_ = num_frequencies;
    keep_negative_frequencies_ = keep_negative_frequencies;
    needs_local_frame_ = needs_local_frame;
    magnon_accumulation_configured_ = needs_magnon_accumulation;
    frequency_slices_configured_ = needs_frequency_slices;
    ring_offset_ = 0;

    time_series_.resize(periodogram_length, num_basis_atoms, num_k_points, stored_channels);
    time_series_.zero();
    time_series_.release_stale_host();

    if (needs_local_frame)
    {
      basis_mag_ring_.resize(periodogram_length, num_basis_atoms, 3);
      basis_mag_ring_.zero();
      basis_mag_ring_.release_stale_host();
      rotations_.resize(num_basis_atoms, 9);
      rotations_.zero();
      rotations_.release_stale_host();
    }
    else
    {
      basis_mag_ring_.clear();
      rotations_.clear();
    }

    time_scratch_.resize(num_basis_atoms, output_channels, periodogram_length);
    time_scratch_.zero();
    time_scratch_.release_stale_host();

    if (needs_magnon_accumulation)
    {
      cumulative_magnon_.resize(num_frequencies, num_k_points, 3);
      cumulative_magnon_.zero();
      cumulative_magnon_.release_stale_host();
    }
    else
    {
      cumulative_magnon_.clear();
    }

    if (needs_frequency_slices && use_multitaper)
    {
      frequency_taper_sum_.resize(num_basis_atoms, output_channels, periodogram_length);
      frequency_taper_sum_.zero();
      frequency_taper_sum_.release_stale_host();
      frequency_taper_power_sum_.resize(num_basis_atoms, output_channels, periodogram_length);
      frequency_taper_power_sum_.zero();
      frequency_taper_power_sum_.release_stale_host();
    }
    else
    {
      frequency_taper_sum_.clear();
      frequency_taper_power_sum_.clear();
    }

    if (time_plan_)
    {
      cufftDestroy(time_plan_);
      time_plan_ = 0;
    }
    const int rank = 1;
    int n[1] = {periodogram_length};
    const int batch = num_basis_atoms * output_channels;
    CHECK_CUFFT_STATUS(cufftPlanMany(
        &time_plan_,
        rank,
        n,
        nullptr,
        1,
        periodogram_length,
        nullptr,
        1,
        periodogram_length,
        CUFFT_Z2Z,
        batch));
    CHECK_CUFFT_STATUS(cufftSetStream(time_plan_, stream_.get()));
    time_configured_ = true;
  }

  void store_sample(
      const jams::MultiArray<double, 2>& spins,
      const int time_index,
      const int stored_channels,
      const bool needs_local_frame,
      const SpectrumBaseMonitor::ChannelTransform& channel_transform,
      const bool copy_sample_to_host,
      std::vector<CmplxStored>& host_sample,
      const bool copy_basis_magnetisation_to_host,
      std::vector<jams::Vec<double, 3>>& host_basis_magnetisation) override
  {
    ensure_channel_weights_uploaded(channel_transform);

    const int sample_size = num_basis_ * num_k_points_ * stored_channels;
    if (copy_sample_to_host)
    {
      sample_block_.resize(sample_size);
    }

    float2* output = copy_sample_to_host
        ? reinterpret_cast<float2*>(sample_block_.mutable_device_data())
        : reinterpret_cast<float2*>(time_series_.mutable_device_data());
    const int output_offset = copy_sample_to_host
        ? 0
        : map_ring_index(time_index) * sample_size;

    if (use_direct_sum_)
    {
      run_direct_sum_sample(
          spins,
          stored_channels,
          needs_local_frame,
          channel_transform.scale_to_physical_spin,
          output_offset,
          output);
    }
    else
    {
      run_spatial_fft(spins);

      const dim3 block(256);
      const dim3 grid((num_basis_ * num_k_points_ + block.x - 1) / block.x);
      extract_compact_sample_kernel<<<grid, block, 0, stream_.get()>>>(
          num_basis_,
          num_k_points_,
          stored_channels,
          output_offset,
          padded_size_[1],
          kspace_z_r2c_,
          needs_local_frame,
          channel_transform.scale_to_physical_spin,
          kElectronGFactor,
          sk_grid_.device_data(),
          k_indices_.device_data(),
          reinterpret_cast<const cufftDoubleComplex*>(basis_phase_factors_.device_data()),
          reinterpret_cast<const cufftDoubleComplex*>(channel_weights_.device_data()),
          globals::mus.device_data(),
          output);
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
    }

    if (needs_local_frame && (copy_basis_magnetisation_to_host || time_configured_))
    {
      double* basis_mag_output = nullptr;
      if (time_configured_)
      {
        basis_mag_output = basis_mag_ring_.mutable_device_data()
            + map_ring_index(time_index) * num_basis_ * 3;
      }
      else
      {
        basis_magnetisation_sample_.resize(num_basis_, 3);
        basis_mag_output = basis_magnetisation_sample_.mutable_device_data();
      }
      jams::monitors::execute_cuda_grouped_spin_sum_reduction(
          stream_,
          basis_chunks_.num_groups,
          basis_chunks_.num_chunks,
          basis_chunks_.chunk_begin_offsets.device_data(),
          basis_chunks_.chunk_end_offsets.device_data(),
          basis_chunks_.group_chunk_begin_offsets.device_data(),
          basis_chunks_.group_chunk_end_offsets.device_data(),
          basis_chunks_.spin_indices.device_data(),
          spins.device_data(),
          basis_chunk_sums_.mutable_device_data(),
          basis_mag_output);
      basis_chunk_sums_.release_stale_host();
    }

    if (copy_sample_to_host)
    {
      host_sample.resize(sample_size);
      CHECK_CUDA_STATUS(cudaMemcpyAsync(
          host_sample.data(),
          sample_block_.device_data(),
          sizeof(CmplxStored) * host_sample.size(),
          cudaMemcpyDeviceToHost,
          stream_.get()));
    }
    if (copy_basis_magnetisation_to_host)
    {
      host_basis_magnetisation.resize(num_basis_);
      CHECK_CUDA_STATUS(cudaMemcpyAsync(
          host_basis_magnetisation.data(),
          basis_magnetisation_sample_.device_data(),
          sizeof(jams::Vec<double, 3>) * host_basis_magnetisation.size(),
          cudaMemcpyDeviceToHost,
          stream_.get()));
    }
    stream_.synchronize();

    if (!copy_sample_to_host)
    {
      time_series_.release_stale_host();
    }
    if (time_configured_ && needs_local_frame)
    {
      basis_mag_ring_.release_stale_host();
    }
  }

  void configure_frequency_inputs(
      const jams::MultiArray<double, 1>& periodogram_window,
      const jams::MultiArray<double, 2>& multitaper_windows,
      const jams::MultiArray<double, 1>& multitaper_weights,
      const SpectrumBaseMonitor::ChannelTransform& channel_transform) override
  {
    scale_to_physical_spin_ = channel_transform.scale_to_physical_spin;
    copy_window_if_needed(periodogram_window, periodogram_window_);

    if (!multitaper_windows.empty())
    {
      multitaper_windows_ = multitaper_windows;
      multitaper_windows_.device_data();
      multitaper_windows_.release_stale_host();
    }
    if (!multitaper_weights.empty())
    {
      const auto weights = multitaper_weights.host_span();
      multitaper_weights_host_.assign(weights.begin(), weights.end());
      multitaper_weights_ = multitaper_weights;
      multitaper_weights_.device_data();
      multitaper_weights_.release_stale_host();
    }
    else
    {
      multitaper_weights_host_.clear();
    }
    ensure_channel_weights_uploaded(channel_transform);
  }

  void accumulate_magnon_spectrum(
      const int periodogram_length,
      const int num_basis_atoms,
      const int num_k_points,
      const int output_channels,
      const bool keep_negative_frequencies,
      const bool needs_local_frame,
      const bool use_multitaper,
      const int multitaper_count) override
  {
    if (!time_configured_)
    {
      throw std::runtime_error("CUDA time FFT storage is not configured");
    }
    if (!magnon_accumulation_configured_)
    {
      throw std::runtime_error("CUDA magnon spectrum accumulation is not configured");
    }
    if (periodogram_length != periodogram_length_
        || num_basis_atoms != num_basis_
        || num_k_points != num_k_points_
        || output_channels != output_channels_
        || keep_negative_frequencies != keep_negative_frequencies_
        || needs_local_frame != needs_local_frame_)
    {
      throw std::runtime_error("CUDA time FFT configuration does not match current spectrum settings");
    }

    if (needs_local_frame)
    {
      const dim3 block(128);
      const dim3 grid((num_basis_ + block.x - 1) / block.x);
      compute_rotations_kernel<<<grid, block, 0, stream_.get()>>>(
          num_basis_,
          periodogram_length_,
          ring_offset_,
          basis_mag_ring_.device_data(),
          periodogram_window_.device_data(),
          rotations_.mutable_device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
      rotations_.release_stale_host();
    }

    const int freq_count = keep_negative_frequencies_
        ? periodogram_length_
        : (periodogram_length_ / 2 + 1);
    const dim3 prepare_block(128);
    const dim3 prepare_grid((num_basis_ * output_channels_ + prepare_block.x - 1) / prepare_block.x);
    const dim3 accum_block(128);
    const dim3 accum_grid((freq_count * output_channels_ + accum_block.x - 1) / accum_block.x);

    for (int k = 0; k < num_k_points_; ++k)
    {
      if (!use_multitaper)
      {
        run_one_time_fft(k, periodogram_window_.device_data(), prepare_grid, prepare_block);
        accumulate_magnon_power_kernel<<<accum_grid, accum_block, 0, stream_.get()>>>(
            num_basis_,
            output_channels_,
            periodogram_length_,
            freq_count,
            num_k_points_,
            k,
            1.0,
            time_scratch_.device_data(),
            cumulative_magnon_.mutable_device_data());
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
        continue;
      }

      for (int taper = 0; taper < multitaper_count; ++taper)
      {
        const double* window = multitaper_windows_.device_data() + taper * periodogram_length_;
        run_one_time_fft(k, window, prepare_grid, prepare_block);
        const double taper_weight = multitaper_weights_host_.at(static_cast<std::size_t>(taper));
        accumulate_magnon_power_kernel<<<accum_grid, accum_block, 0, stream_.get()>>>(
            num_basis_,
            output_channels_,
            periodogram_length_,
            freq_count,
            num_k_points_,
            k,
            taper_weight,
            time_scratch_.device_data(),
            cumulative_magnon_.mutable_device_data());
        DEBUG_CHECK_CUDA_ASYNC_STATUS;
      }
    }
    stream_.synchronize();
    cumulative_magnon_.release_stale_host();
  }

  void copy_magnon_spectrum_to_host(
      jams::MultiArray<jams::Vec<double, 3>, 2>& cumulative) override
  {
    const auto host = cumulative_magnon_.host_view();
    for (int f = 0; f < cumulative_magnon_.extent(0); ++f)
    {
      for (int k = 0; k < cumulative_magnon_.extent(1); ++k)
      {
        cumulative(f, k)[0] = host(f, k, 0);
        cumulative(f, k)[1] = host(f, k, 1);
        cumulative(f, k)[2] = host(f, k, 2);
      }
    }
    cumulative_magnon_.release_stale_host();
  }

  void compute_frequency_spectrum_at_k(
      const int kpoint_index,
      const bool use_multitaper,
      const int multitaper_count) override
  {
    if (!time_configured_)
    {
      throw std::runtime_error("CUDA time FFT storage is not configured");
    }
    if (!frequency_slices_configured_)
    {
      throw std::runtime_error("CUDA frequency-slice output is not configured");
    }
    if (kpoint_index < 0 || kpoint_index >= num_k_points_)
    {
      throw std::runtime_error("CUDA frequency-slice k-point index is out of range");
    }

    if (needs_local_frame_)
    {
      const dim3 block(128);
      const dim3 grid((num_basis_ + block.x - 1) / block.x);
      compute_rotations_kernel<<<grid, block, 0, stream_.get()>>>(
          num_basis_,
          periodogram_length_,
          ring_offset_,
          basis_mag_ring_.device_data(),
          periodogram_window_.device_data(),
          rotations_.mutable_device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
      rotations_.release_stale_host();
    }

    const dim3 prepare_block(128);
    const dim3 prepare_grid((num_basis_ * output_channels_ + prepare_block.x - 1) / prepare_block.x);
    if (!use_multitaper)
    {
      run_one_time_fft(kpoint_index, periodogram_window_.device_data(), prepare_grid, prepare_block);
      stream_.synchronize();
      return;
    }

    if (frequency_taper_sum_.extent(0) != num_basis_
        || frequency_taper_sum_.extent(1) != output_channels_
        || frequency_taper_sum_.extent(2) != periodogram_length_
        || frequency_taper_power_sum_.extent(0) != num_basis_
        || frequency_taper_power_sum_.extent(1) != output_channels_
        || frequency_taper_power_sum_.extent(2) != periodogram_length_)
    {
      frequency_taper_sum_.resize(num_basis_, output_channels_, periodogram_length_);
      frequency_taper_power_sum_.resize(num_basis_, output_channels_, periodogram_length_);
    }

    const std::size_t total_bytes_complex =
        sizeof(cufftDoubleComplex) * static_cast<std::size_t>(frequency_taper_sum_.size());
    const std::size_t total_bytes_power =
        sizeof(double) * static_cast<std::size_t>(frequency_taper_power_sum_.size());
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        frequency_taper_sum_.mutable_device_data(),
        0,
        total_bytes_complex,
        stream_.get()));
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        frequency_taper_power_sum_.mutable_device_data(),
        0,
        total_bytes_power,
        stream_.get()));

    const int total = num_basis_ * output_channels_ * periodogram_length_;
    const dim3 accum_block(128);
    const dim3 accum_grid((total + accum_block.x - 1) / accum_block.x);
    for (int taper = 0; taper < multitaper_count; ++taper)
    {
      const double* window = multitaper_windows_.device_data() + taper * periodogram_length_;
      run_one_time_fft(kpoint_index, window, prepare_grid, prepare_block);
      const double taper_weight = multitaper_weights_host_.at(static_cast<std::size_t>(taper));
      accumulate_tapered_spectrum_kernel<<<accum_grid, accum_block, 0, stream_.get()>>>(
          total,
          taper_weight,
          time_scratch_.device_data(),
          frequency_taper_sum_.mutable_device_data(),
          frequency_taper_power_sum_.mutable_device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
    }

    reconstruct_tapered_spectrum_kernel<<<accum_grid, accum_block, 0, stream_.get()>>>(
        total,
        frequency_taper_sum_.device_data(),
        frequency_taper_power_sum_.device_data(),
        time_scratch_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    stream_.synchronize();
    time_scratch_.release_stale_host();
    frequency_taper_sum_.release_stale_host();
    frequency_taper_power_sum_.release_stale_host();
  }

  void copy_frequency_spectrum_slice_to_host(
      SpectrumBaseMonitor::CmplxMappedSlice& spectrum) override
  {
    if (spectrum.extent(0) != num_basis_
        || spectrum.extent(1) != periodogram_length_
        || spectrum.extent(2) != output_channels_)
    {
      spectrum.resize(num_basis_, periodogram_length_, output_channels_);
    }

    stream_.synchronize();
    const auto source = time_scratch_.host_view();
    auto destination = spectrum.mutable_host_view();
    for (int a = 0; a < num_basis_; ++a)
    {
      for (int c = 0; c < output_channels_; ++c)
      {
        for (int t = 0; t < periodogram_length_; ++t)
        {
          const auto value = source(a, c, t);
          destination(a, t, c) = jams::ComplexHi{value.x, value.y};
        }
      }
    }
  }

  void advance_ring_window(const int overlap) override
  {
    if (periodogram_length_ == 0)
    {
      return;
    }
    assert(overlap < periodogram_length_);
    ring_offset_ = (ring_offset_ + (periodogram_length_ - overlap)) % periodogram_length_;
  }

private:
  template<typename T>
  static std::size_t bytes_count(const int n0, const int n1 = 1, const int n2 = 1, const int n3 = 1)
  {
    return sizeof(T)
        * static_cast<std::size_t>(n0)
        * static_cast<std::size_t>(n1)
        * static_cast<std::size_t>(n2)
        * static_cast<std::size_t>(n3);
  }

  int map_ring_index(const int logical_time) const
  {
    return (ring_offset_ + logical_time) % periodogram_length_;
  }

  void initialise_k_indices(const std::vector<jams::HKLIndex>& k_points)
  {
    std::vector<int> indices;
    indices.reserve(k_points.size() * 4);
    for (const auto& point : k_points)
    {
      indices.push_back(point.index.offset[0]);
      indices.push_back(point.index.offset[1]);
      indices.push_back(point.index.offset[2]);
      indices.push_back(point.index.conj ? 1 : 0);
    }
    copy_matrix_to_device_only(k_indices_, indices, static_cast<int>(k_points.size()), 4);
  }

  void initialise_basis_phase_factors(
      const jams::MultiArray<jams::ComplexHi, 2>& basis_phase_factors)
  {
    basis_phase_factors_ = basis_phase_factors;
    basis_phase_factors_.device_data();
    basis_phase_factors_.release_stale_host();
  }

  void initialise_direct_sum_buffers(
      const std::vector<jams::HKLIndex>& k_points,
      const std::vector<SpectrumBaseMonitor::DirectSumSite>& direct_sum_sites)
  {
    std::vector<double> k_hkl;
    k_hkl.reserve(k_points.size() * 3);
    for (const auto& point : k_points)
    {
      k_hkl.push_back(point.hkl[0]);
      k_hkl.push_back(point.hkl[1]);
      k_hkl.push_back(point.hkl[2]);
    }
    copy_matrix_to_device_only(
        direct_k_hkl_,
        k_hkl,
        static_cast<int>(k_points.size()),
        3);
    spatial_memory_bytes_ += bytes_count<double>(static_cast<int>(k_hkl.size()));

    num_direct_sum_sites_ = static_cast<int>(direct_sum_sites.size());
    std::vector<int> site_indices;
    std::vector<int> basis_indices;
    std::vector<double> positions_frac;
    std::vector<double> window_weights;
    site_indices.reserve(direct_sum_sites.size());
    basis_indices.reserve(direct_sum_sites.size());
    positions_frac.reserve(direct_sum_sites.size() * 3);
    window_weights.reserve(direct_sum_sites.size());

    for (const auto& site : direct_sum_sites)
    {
      site_indices.push_back(site.site_index);
      basis_indices.push_back(site.basis_index);
      positions_frac.push_back(site.position_frac[0]);
      positions_frac.push_back(site.position_frac[1]);
      positions_frac.push_back(site.position_frac[2]);
      window_weights.push_back(site.window_weight);
    }

    copy_vector_to_device_only(direct_site_indices_, site_indices);
    copy_vector_to_device_only(direct_site_basis_indices_, basis_indices);
    copy_matrix_to_device_only(direct_site_positions_frac_, positions_frac, num_direct_sum_sites_, 3);
    copy_vector_to_device_only(direct_site_window_weights_, window_weights);

    direct_sum_buffer_.resize(num_basis_, num_k_points_, 3);
    direct_sum_buffer_.zero();
    direct_sum_buffer_.release_stale_host();

    spatial_memory_bytes_ += bytes_count<int>(num_direct_sum_sites_, 2);
    spatial_memory_bytes_ += bytes_count<double>(num_direct_sum_sites_, 4);
    spatial_memory_bytes_ += bytes_count<cufftDoubleComplex>(num_basis_, num_k_points_, 3);
  }

  void initialise_basis_reduction(const Lattice& lattice)
  {
    basis_chunks_ = jams::monitors::make_cuda_basis_spin_group_chunks(lattice, num_spins_);
    basis_chunk_sums_.resize(basis_chunks_.num_chunks, 3);
    spatial_memory_bytes_ += basis_chunks_.device_memory_bytes();
    spatial_memory_bytes_ += bytes_count<double>(basis_chunks_.num_chunks, 3);
  }

  void initialise_spatial_buffers(const Lattice& lattice)
  {
    const int num_kspace_values =
        padded_size_[0] * padded_size_[1] * kspace_z_r2c_ * num_basis_ * 3;
    sk_grid_.resize(num_kspace_values);
    sk_grid_.zero();
    sk_grid_.release_stale_host();
    spatial_memory_bytes_ += bytes_count<cufftDoubleComplex>(num_kspace_values);

    if (!use_dense_input_)
    {
      return;
    }

    const int num_dense_slots =
        padded_size_[0] * padded_size_[1] * padded_size_[2] * num_basis_;
    rspace_dense_.resize(num_dense_slots * 3);
    rspace_dense_.zero();
    rspace_dense_.release_stale_host();
    spatial_memory_bytes_ += bytes_count<double>(num_dense_slots * 3);

    std::vector<int> site_map(static_cast<std::size_t>(num_dense_slots), -1);
    for (int i = 0; i < padded_size_[0]; ++i)
    {
      for (int j = 0; j < padded_size_[1]; ++j)
      {
        for (int k = 0; k < padded_size_[2]; ++k)
        {
          for (int a = 0; a < num_basis_; ++a)
          {
            const int dense_slot = ((i * padded_size_[1] + j) * padded_size_[2] + k) * num_basis_ + a;
            if (i >= grid_size_[0] || j >= grid_size_[1] || k >= grid_size_[2])
            {
              continue;
            }
            const auto site_index = lattice.site_index_by_unit_cell_optional(i, j, k, a);
            if (site_index)
            {
              site_map[static_cast<std::size_t>(dense_slot)] = *site_index;
            }
          }
        }
      }
    }
    copy_vector_to_device_only(fft_site_map_, site_map);
    spatial_memory_bytes_ += bytes_count<int>(num_dense_slots);
  }

  void initialise_spatial_plan()
  {
    const int rank = 3;
    int fft_size[3] = {padded_size_[0], padded_size_[1], padded_size_[2]};
    int rspace_embed[3] = {padded_size_[0], padded_size_[1], padded_size_[2]};
    int kspace_embed[3] = {padded_size_[0], padded_size_[1], kspace_z_r2c_};
    const int num_transforms = 3 * num_basis_;
    const int stride = 3 * num_basis_;
    const int dist = 1;

    CHECK_CUFFT_STATUS(cufftPlanMany(
        &spatial_plan_,
        rank,
        fft_size,
        rspace_embed,
        stride,
        dist,
        kspace_embed,
        stride,
        dist,
        CUFFT_D2Z,
        num_transforms));
    CHECK_CUFFT_STATUS(cufftSetStream(spatial_plan_, stream_.get()));
  }

  void ensure_channel_weights_uploaded(const SpectrumBaseMonitor::ChannelTransform& channel_transform)
  {
    std::vector<jams::ComplexHi> weights(9, jams::ComplexHi{0.0, 0.0});
    for (int row = 0; row < 3; ++row)
    {
      for (int col = 0; col < 3; ++col)
      {
        weights[static_cast<std::size_t>(3 * row + col)] = channel_transform.weights[row][col];
      }
    }
    if (weights == channel_weights_host_)
    {
      return;
    }
    copy_matrix_to_device_only(channel_weights_, weights, 3, 3);
    channel_weights_host_ = std::move(weights);
  }

  void copy_window_if_needed(
      const jams::MultiArray<double, 1>& source,
      jams::MultiArray<double, 1>& destination)
  {
    if (destination.size() == source.size())
    {
      destination = source;
    }
    else
    {
      destination.resize(source.size());
      auto dst = destination.mutable_host_span();
      auto src = source.host_span();
      std::copy(src.begin(), src.end(), dst.begin());
    }
    destination.device_data();
    destination.release_stale_host();
  }

  void run_spatial_fft(const jams::MultiArray<double, 2>& spins)
  {
    const double* input = spins.device_data();
    if (use_dense_input_)
    {
      const int num_values = static_cast<int>(rspace_dense_.size());
      const dim3 block(256);
      const dim3 grid((num_values + block.x - 1) / block.x);
      pack_dense_spins_kernel<<<grid, block, 0, stream_.get()>>>(
          num_values,
          fft_site_map_.device_data(),
          spins.device_data(),
          rspace_dense_.mutable_device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
      input = rspace_dense_.device_data();
    }

    CHECK_CUFFT_STATUS(cufftExecD2Z(
        spatial_plan_,
        const_cast<cufftDoubleReal*>(input),
        sk_grid_.mutable_device_data()));

    const int num_complex = static_cast<int>(sk_grid_.size());
    const dim3 block(256);
    const dim3 grid((num_complex + block.x - 1) / block.x);
    const double scale = 1.0 / std::sqrt(
        static_cast<double>(padded_size_[0])
        * static_cast<double>(padded_size_[1])
        * static_cast<double>(padded_size_[2]));
    scale_complex_kernel<<<grid, block, 0, stream_.get()>>>(
        num_complex,
        scale,
        sk_grid_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    sk_grid_.release_stale_host();
  }

  void run_direct_sum_sample(
      const jams::MultiArray<double, 2>& spins,
      const int stored_channels,
      const bool needs_local_frame,
      const bool scale_to_physical_spin,
      const int output_offset,
      float2* output)
  {
    const std::size_t buffer_bytes =
        sizeof(cufftDoubleComplex) * static_cast<std::size_t>(direct_sum_buffer_.size());
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        direct_sum_buffer_.mutable_device_data(),
        0,
        buffer_bytes,
        stream_.get()));

    if (num_direct_sum_sites_ > 0)
    {
      const dim3 sum_block(128);
      const dim3 sum_grid((num_direct_sum_sites_ * num_k_points_ + sum_block.x - 1) / sum_block.x);
      direct_spatial_sum_kernel<<<sum_grid, sum_block, 0, stream_.get()>>>(
          num_direct_sum_sites_,
          num_k_points_,
          direct_sum_spatial_scale_,
          direct_site_indices_.device_data(),
          direct_site_basis_indices_.device_data(),
          direct_site_positions_frac_.device_data(),
          direct_site_window_weights_.device_data(),
          direct_k_hkl_.device_data(),
          spins.device_data(),
          direct_sum_buffer_.mutable_device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
    }

    const dim3 map_block(256);
    const dim3 map_grid((num_basis_ * num_k_points_ + map_block.x - 1) / map_block.x);
    map_direct_sum_sample_kernel<<<map_grid, map_block, 0, stream_.get()>>>(
        num_basis_,
        num_k_points_,
        stored_channels,
        output_offset,
        needs_local_frame,
        scale_to_physical_spin,
        kElectronGFactor,
        direct_sum_buffer_.device_data(),
        reinterpret_cast<const cufftDoubleComplex*>(channel_weights_.device_data()),
        globals::mus.device_data(),
        output);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    direct_sum_buffer_.release_stale_host();
  }

  void run_one_time_fft(
      const int k,
      const double* window,
      const dim3 prepare_grid,
      const dim3 prepare_block)
  {
    prepare_time_fft_input_kernel<<<prepare_grid, prepare_block, 0, stream_.get()>>>(
        num_basis_,
        num_k_points_,
        stored_channels_,
        output_channels_,
        periodogram_length_,
        ring_offset_,
        k,
        needs_local_frame_,
        scale_to_physical_spin_,
        kElectronGFactor,
        reinterpret_cast<const float2*>(time_series_.device_data()),
        needs_local_frame_ ? rotations_.device_data() : nullptr,
        reinterpret_cast<const cufftDoubleComplex*>(channel_weights_.device_data()),
        globals::mus.device_data(),
        window,
        time_scratch_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    CHECK_CUFFT_STATUS(cufftExecZ2Z(
        time_plan_,
        time_scratch_.mutable_device_data(),
        time_scratch_.mutable_device_data(),
        CUFFT_FORWARD));
    time_scratch_.release_stale_host();
  }

  int num_basis_ = 0;
  int num_spins_ = 0;
  jams::Vec<int, 3> grid_size_ {};
  jams::Vec<int, 3> padded_size_ {};
  int kspace_z_r2c_ = 0;
  int num_k_points_ = 0;
  bool use_direct_sum_ = false;
  double direct_sum_spatial_scale_ = 1.0;
  int num_direct_sum_sites_ = 0;
  bool use_dense_input_ = false;
  std::size_t spatial_memory_bytes_ = 0;

  CudaStream stream_ {};
  cufftHandle spatial_plan_ {};
  cufftHandle time_plan_ {};

  jams::MultiArray<cufftDoubleComplex, 1> sk_grid_;
  jams::MultiArray<double, 1> rspace_dense_;
  jams::MultiArray<int, 1> fft_site_map_;
  jams::monitors::CudaSpinGroupChunks basis_chunks_;
  jams::MultiArray<double, 2> basis_chunk_sums_;
  jams::MultiArray<int, 2> k_indices_;
  jams::MultiArray<double, 2> direct_k_hkl_;
  jams::MultiArray<int, 1> direct_site_indices_;
  jams::MultiArray<int, 1> direct_site_basis_indices_;
  jams::MultiArray<double, 2> direct_site_positions_frac_;
  jams::MultiArray<double, 1> direct_site_window_weights_;
  jams::MultiArray<cufftDoubleComplex, 3> direct_sum_buffer_;
  jams::MultiArray<jams::ComplexHi, 2> basis_phase_factors_;
  jams::MultiArray<jams::ComplexHi, 2> channel_weights_;
  jams::MultiArray<CmplxStored, 1> sample_block_;
  jams::MultiArray<double, 2> basis_magnetisation_sample_;

  bool time_configured_ = false;
  bool magnon_accumulation_configured_ = false;
  bool frequency_slices_configured_ = false;
  int periodogram_length_ = 0;
  int stored_channels_ = 0;
  int output_channels_ = 0;
  int num_frequencies_ = 0;
  bool keep_negative_frequencies_ = false;
  bool needs_local_frame_ = false;
  bool scale_to_physical_spin_ = true;
  int ring_offset_ = 0;

  jams::MultiArray<CmplxStored, 4> time_series_;
  jams::MultiArray<double, 3> basis_mag_ring_;
  jams::MultiArray<double, 2> rotations_;
  jams::MultiArray<cufftDoubleComplex, 3> time_scratch_;
  jams::MultiArray<double, 3> cumulative_magnon_;
  jams::MultiArray<cufftDoubleComplex, 3> frequency_taper_sum_;
  jams::MultiArray<double, 3> frequency_taper_power_sum_;
  jams::MultiArray<double, 1> periodogram_window_;
  jams::MultiArray<double, 2> multitaper_windows_;
  jams::MultiArray<double, 1> multitaper_weights_;
  std::vector<jams::ComplexHi> channel_weights_host_;
  std::vector<double> multitaper_weights_host_;
};

}  // namespace

std::unique_ptr<SpectrumBaseMonitor::CudaBackend>
SpectrumBaseMonitor::make_cuda_backend_() const
{
  return std::make_unique<SpectrumCudaBackend>(
      *globals::lattice,
      k_points_,
      basis_phase_factors_,
      use_direct_sum_(),
      direct_sum_sites_,
      direct_sum_spatial_scale_);
}

#endif  // HAS_CUDA
