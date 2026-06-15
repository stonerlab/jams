#include <complex>
#include <cmath>
#include <fstream>

#include <libconfig.h++>
#include <cufft.h>

#include "jams/interface/fft.h"
#include "jams/helpers/exception.h"
#include "jams/interface/config.h"

#include "jams/helpers/output.h"
#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_device_complex_ops.h"
#include "jams/hamiltonian/cuda_dipole_fft.h"
#include "jams/hamiltonian/dipole_interaction.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_array_kernels.h"
#include <jams/helpers/mixed_precision.h>


// Pack upper-triangular (i<=j) pairs into a 1D index.
// Number of pairs = n*(n+1)/2.
__host__ __device__ __forceinline__ int upper_tri_index(const int i, const int j, const int n) {
  // Requires: 0 <= i <= j < n
  return i * n - (i * (i - 1)) / 2 + (j - i);
}

constexpr int kDipoleTensorComponents = 6;
constexpr int kEnergyCurrentDirections = 3;
constexpr int kEnergyCurrentTensorComponents = kEnergyCurrentDirections * kDipoleTensorComponents;

struct TensorOffsetRange {
  int begin;
  int end;
};

TensorOffsetRange tensor_offset_range(const int size, const bool is_periodic) {
  if (is_periodic) {
    return {0, size};
  }
  return {1 - size, size};
}

template <typename ComplexType>
__device__ __forceinline__ ComplexType complex_conj(const ComplexType &z);

template <>
__device__ __forceinline__ cuComplex complex_conj<cuComplex>(const cuComplex &z) {
  return cuConjf(z);
}

template <>
__device__ __forceinline__ cuDoubleComplex complex_conj<cuDoubleComplex>(const cuDoubleComplex &z) {
  return cuConj(z);
}

template <typename ComplexType>
__device__ __forceinline__ ComplexType complex_negate(const ComplexType &z);

template <>
__device__ __forceinline__ cuComplex complex_negate<cuComplex>(const cuComplex &z) {
  return make_cuComplex(-z.x, -z.y);
}

template <>
__device__ __forceinline__ cuDoubleComplex complex_negate<cuDoubleComplex>(const cuDoubleComplex &z) {
  return make_cuDoubleComplex(-z.x, -z.y);
}

template<typename ComplexType>
__global__ void cuda_dipole_convolution(
  const unsigned int num_kpoints,
  const unsigned int num_pos,
  const bool use_full_tensor_storage,
  const int tensor_components,
  const int tensor_component_offset,
  const bool compact_reverse_is_odd,
  const ComplexType* sk,
  const ComplexType* wk,
  ComplexType* hk
)
{
  unsigned int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int pos_i = blockIdx.y;

  if (k_idx >= num_kpoints || pos_i >= num_pos) return;

  ComplexType hk_sum[3] = {0.0, 0.0, 0.0};

  for (int pos_j = 0; pos_j < num_pos; ++pos_j) {
    int batch_base_j = 3 * pos_j;
    int idx0 = (batch_base_j + 0) * num_kpoints + k_idx;
    int idx1 = (batch_base_j + 1) * num_kpoints + k_idx;
    int idx2 = (batch_base_j + 2) * num_kpoints + k_idx;

    const ComplexType sq0 = sk[idx0];
    const ComplexType sq1 = sk[idx1];
    const ComplexType sq2 = sk[idx2];

    int tensor_set = 0;
    bool conjugate_tensor = true;
    if (use_full_tensor_storage) {
      tensor_set = pos_i * (int)num_pos + pos_j;
      conjugate_tensor = false;
    } else {
      // wk is stored only for i<=j (upper-triangular in (pos_i,pos_j))
      const int a = (pos_i <= pos_j) ? pos_i : pos_j;
      const int b = (pos_i <= pos_j) ? pos_j : pos_i;
      const bool swapped = (pos_i > pos_j);

      tensor_set = upper_tri_index(a, b, (int)num_pos);
      conjugate_tensor = !swapped;
    }

    int base0 = ((tensor_set * tensor_components + tensor_component_offset + 0) * (int)num_kpoints) + (int)k_idx;
    int base1 = base0 + (int)num_kpoints;
    int base2 = base1 + (int)num_kpoints;

    // Compact storage relies on Hermitian symmetry: W_{ji}(k) = conj(W_{ij}(k)).
    // Staggering the access of w components gives about 1 us improvement on A30.
    ComplexType w0 = conjugate_tensor ? complex_conj(wk[base0]) : wk[base0];
    ComplexType w1 = conjugate_tensor ? complex_conj(wk[base1]) : wk[base1];
    ComplexType w2 = conjugate_tensor ? complex_conj(wk[base2]) : wk[base2];
    if (compact_reverse_is_odd && !use_full_tensor_storage && conjugate_tensor) {
      w0 = complex_negate(w0);
      w1 = complex_negate(w1);
      w2 = complex_negate(w2);
    }

    hk_sum[0] +=  w0 * sq0 + w1 * sq1 + w2 * sq2;

    int base3 = base2 + (int)num_kpoints;
    int base4 = base3 + (int)num_kpoints;

    ComplexType w3 = conjugate_tensor ? complex_conj(wk[base3]) : wk[base3];
    ComplexType w4 = conjugate_tensor ? complex_conj(wk[base4]) : wk[base4];
    if (compact_reverse_is_odd && !use_full_tensor_storage && conjugate_tensor) {
      w3 = complex_negate(w3);
      w4 = complex_negate(w4);
    }

    hk_sum[1] +=  w1 * sq0 + w3 * sq1 + w4 * sq2;

    int base5 = base4 + (int)num_kpoints;

    ComplexType w5 = conjugate_tensor ? complex_conj(wk[base5]) : wk[base5];
    if (compact_reverse_is_odd && !use_full_tensor_storage && conjugate_tensor) {
      w5 = complex_negate(w5);
    }

    hk_sum[2] +=  w2 * sq0 + w4 * sq1 + w5 * sq2;
  }

  int batch_base_i = 3 * pos_i;
  int out0 = (batch_base_i + 0) * num_kpoints + k_idx;
  int out1 = (batch_base_i + 1) * num_kpoints + k_idx;
  int out2 = (batch_base_i + 2) * num_kpoints + k_idx;

  hk[out0] = hk_sum[0];
  hk[out1] = hk_sum[1];
  hk[out2] = hk_sum[2];
}

__global__ void cuda_pack_lattice_vector_field(
    const unsigned int num_slots,
    const int* site_map,
    const jams::Real* active_field,
    jams::Real* dense_field)
{
  const unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= num_slots) return;

  const int active_site = site_map[slot];
  const unsigned int dense_base = 3 * slot;
  if (active_site >= 0) {
    const unsigned int active_base = 3 * static_cast<unsigned int>(active_site);
    dense_field[dense_base + 0] = active_field[active_base + 0];
    dense_field[dense_base + 1] = active_field[active_base + 1];
    dense_field[dense_base + 2] = active_field[active_base + 2];
  } else {
    dense_field[dense_base + 0] = static_cast<jams::Real>(0.0);
    dense_field[dense_base + 1] = static_cast<jams::Real>(0.0);
    dense_field[dense_base + 2] = static_cast<jams::Real>(0.0);
  }
}

__global__ void cuda_unpack_lattice_vector_field(
    const unsigned int num_slots,
    const int* site_map,
    const jams::Real* dense_field,
    jams::Real* active_field)
{
  const unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= num_slots) return;

  const int active_site = site_map[slot];
  if (active_site >= 0) {
    const unsigned int dense_base = 3 * slot;
    const unsigned int active_base = 3 * static_cast<unsigned int>(active_site);
    active_field[active_base + 0] = dense_field[dense_base + 0];
    active_field[active_base + 1] = dense_field[dense_base + 1];
    active_field[active_base + 2] = dense_field[dense_base + 2];
  }
}

CudaDipoleFFTHamiltonian::~CudaDipoleFFTHamiltonian() {
  if (cuda_fft_s_rspace_to_kspace) {
      cufftDestroy(cuda_fft_s_rspace_to_kspace);
  }

  if (cuda_fft_h_kspace_to_rspace) {
    cufftDestroy(cuda_fft_h_kspace_to_rspace);
  }
}

CudaDipoleFFTHamiltonian::CudaDipoleFFTHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size),
  r_cutoff_(0),
  distance_tolerance_(jams::defaults::lattice_tolerance),
  kspace_size_(0, 0, 0),
  kspace_padded_size_(0, 0, 0),
  kspace_s_(),
  kspace_h_(),
  cuda_fft_s_rspace_to_kspace(),
  cuda_fft_h_kspace_to_rspace()
{
  debug_ = jams::config_optional<bool>(settings, "debug", debug_);
  check_radius_ = jams::config_optional<bool>(settings, "check_radius", check_radius_);
  check_symmetry_ = jams::config_optional<bool>(settings, "check_symmetry", check_symmetry_);

  r_cutoff_ = jams::config_required<jams::Real>(settings, "r_cutoff");
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";
  std::cout << "  r_cutoff_max " << ::globals::lattice->max_interaction_radius() << "\n";

  if (check_radius_) {
    if (r_cutoff_ > ::globals::lattice->max_interaction_radius()) {
      throw std::runtime_error("CudaDipoleFFTHamiltonian r_cutoff is too large for the lattice size."
                                       "The cutoff must be less than the inradius of the lattice.");
    }
  }

  distance_tolerance_ = jams::config_optional<jams::Real>(
      settings, "distance_tolerance", distance_tolerance_);
  std::cout << "  distance_tolerance " << distance_tolerance_ << "\n";

  for (int n = 0; n < 3; ++n) {
      kspace_size_[n] = ::globals::lattice->size(n);
  }

  kspace_padded_size_ = kspace_size_;

  for (int n = 0; n < 3; ++n) {
      if (!::globals::lattice->is_periodic(n)) {
          kspace_padded_size_[n] = kspace_size_[n] * 2;
      }
  }

  use_dense_fft_buffers_ = globals::lattice->has_cropping() || kspace_padded_size_ != kspace_size_;
  use_full_tensor_storage_ = use_dense_fft_buffers_;

  unsigned int kspace_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2]/2 + 1) *
                             globals::lattice->num_basis_sites() * 3;

  kspace_s_.resize(kspace_size);
  kspace_h_.resize(kspace_size);

  kspace_s_.zero();
  kspace_h_.zero();

  std::cout << "    kspace size " << kspace_size_ << "\n";
  std::cout << "    kspace padded size " << kspace_padded_size_ << "\n";

  const int num_sites     = globals::lattice->num_basis_sites();

  int rank            = 3;
  int rspace_embed[3] = {
      use_dense_fft_buffers_ ? kspace_padded_size_[0] : kspace_size_[0],
      use_dense_fft_buffers_ ? kspace_padded_size_[1] : kspace_size_[1],
      use_dense_fft_buffers_ ? kspace_padded_size_[2] : kspace_size_[2]};
  int kspace_embed[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]/2 + 1};

  int fft_size[3] = {rspace_embed[0], rspace_embed[1], rspace_embed[2]};

  const int num_kpoints   = kspace_embed[0] * kspace_embed[1] * kspace_embed[2]; // Nx * Ny * (Nz/2+1)
  const int num_transforms = 3 * num_sites;                                      // unchanged

  // Input (real, r-space) layout: keep as before
  const int istride = 3 * num_sites;
  const int idist   = 1;

  // Output (complex, k-space) layout: [batch][k_idx], batch = 3*pos + comp
  const int ostride = 1;               // k dimension is contiguous
  const int odist   = num_kpoints;     // distance between batches


#if DO_MIXED_PRECISION
  CHECK_CUFFT_STATUS(
      cufftPlanMany(&cuda_fft_s_rspace_to_kspace,
                    rank,
                    fft_size,
                    rspace_embed,  // inembed
                    istride,       // istride
                    idist,         // idist
                    kspace_embed,  // onembed
                    ostride,       // ostride
                    odist,         // odist
                    CUFFT_R2C,
                    num_transforms));
#else
  CHECK_CUFFT_STATUS(
      cufftPlanMany(&cuda_fft_s_rspace_to_kspace,
                    rank,
                    fft_size,
                    rspace_embed,  // inembed
                    istride,       // istride
                    idist,         // idist
                    kspace_embed,  // onembed
                    ostride,       // ostride
                    odist,         // odist
                    CUFFT_D2Z,
                    num_transforms));
#endif

#if DO_MIXED_PRECISION
  CHECK_CUFFT_STATUS(
      cufftPlanMany(&cuda_fft_h_kspace_to_rspace,
                    rank,
                    fft_size,
                    kspace_embed,  // inembed (k-space)
                    ostride,       // istride (now complex input)
                    odist,         // idist
                    rspace_embed,  // onembed (r-space)
                    istride,       // ostride
                    idist,         // odist
                    CUFFT_C2R,
                    num_transforms));
#else
  CHECK_CUFFT_STATUS(
      cufftPlanMany(&cuda_fft_h_kspace_to_rspace,
                    rank,
                    fft_size,
                    kspace_embed,  // inembed (k-space)
                    ostride,       // istride
                    odist,         // idist
                    rspace_embed,  // onembed (r-space)
                    istride,       // ostride
                    idist,         // odist
                    CUFFT_Z2D,
                    num_transforms));
#endif

  if (use_dense_fft_buffers_) {
    const int num_dense_slots = rspace_embed[0] * rspace_embed[1] * rspace_embed[2] * num_sites;
    std::vector<int> site_map(num_dense_slots, -1);
    for (int i = 0; i < rspace_embed[0]; ++i) {
      for (int j = 0; j < rspace_embed[1]; ++j) {
        for (int k = 0; k < rspace_embed[2]; ++k) {
          for (int m = 0; m < num_sites; ++m) {
            const int slot = ((i * rspace_embed[1] + j) * rspace_embed[2] + k) * num_sites + m;
            if (i >= kspace_size_[0] || j >= kspace_size_[1] || k >= kspace_size_[2]) {
              continue;
            }
            const auto site_index = globals::lattice->site_index_by_unit_cell_optional(i, j, k, m);
            if (site_index) {
              site_map[slot] = *site_index;
            }
          }
        }
      }
    }
    fft_site_map_.resize(num_dense_slots);
    rspace_s_dense_.resize(3 * num_dense_slots);
    rspace_h_dense_.resize(3 * num_dense_slots);
    for (int i = 0; i < num_dense_slots; ++i) {
      fft_site_map_(i) = site_map[i];
    }
    rspace_s_dense_.zero();
    rspace_h_dense_.zero();
  }

  const auto num_tensor_components = kDipoleTensorComponents;

  const int num_tensor_sets = use_full_tensor_storage_
      ? num_sites * num_sites
      : num_sites * (num_sites + 1) / 2;
  kspace_tensors_.resize(num_tensor_sets, num_tensor_components, num_kpoints);
  kspace_tensors_.zero();

  if (use_full_tensor_storage_) {
    for (int pos_i = 0; pos_i < num_sites; ++pos_i) {
      std::vector<jams::Vec<double, 3>> generated_positions;
      for (int pos_j = 0; pos_j < num_sites; ++pos_j) {
        const int tensor_set = pos_i * num_sites + pos_j;
        generate_kspace_dipole_tensor(pos_i, pos_j, tensor_set, generated_positions);
      }
    }
  } else {
    for (int pos_i = 0; pos_i < num_sites; ++pos_i) {
      for (int pos_j = pos_i; pos_j < num_sites; ++pos_j) {
        std::vector<jams::Vec<double, 3>> generated_positions;
        const int pair = upper_tri_index(pos_i, pos_j, num_sites);
        generate_kspace_dipole_tensor(pos_i, pos_j, pair, generated_positions);

        if (check_symmetry_ && (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2))) {
          if (!globals::lattice->is_a_symmetry_complete_set(pos_i, generated_positions, distance_tolerance_)) {
            throw std::runtime_error(
                "The points included in the dipole tensor do not form set of all symmetric points.\n"
                "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
                "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
          }
        }
      }
    }
  }

  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_s_rspace_to_kspace, cuda_stream_.get()));
  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_h_kspace_to_rspace, cuda_stream_.get()));
}

jams::Real CudaDipoleFFTHamiltonian::calculate_total_energy(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) {
  calculate_energies(time, spins);
  return cuda_reduce_array(energy_.device_data(), globals::num_spins, cuda_stream_.get());
}

jams::Real CudaDipoleFFTHamiltonian::calculate_one_spin_energy(const int i, const jams::Vec<double, 3> &s_i, jams::Real time) {
    throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_one_spin_energy is not implemented");
}

jams::Real CudaDipoleFFTHamiltonian::calculate_energy(const int i, jams::Real time) {
    throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_energy is not implemented");
}

jams::Real CudaDipoleFFTHamiltonian::calculate_energy_difference(
    int i, const jams::Vec<double, 3> &spin_initial, const jams::Vec<double, 3> &spin_final, jams::Real time) {
  throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_energy_difference is not implemented");
}

void CudaDipoleFFTHamiltonian::calculate_energies(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) {
  calculate_fields(time, spins);
  const auto minus_half = static_cast<jams::Real>(-0.5);
  cuda_array_dot_product(globals::num_spins, minus_half, spins.device_data(), field_.device_data(), energy_.mutable_device_data(), cuda_stream_.get());
}

jams::Vec<jams::Real, 3> CudaDipoleFFTHamiltonian::calculate_field(const int i, jams::Real time) {
  throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_field is not implemented");
}

void CudaDipoleFFTHamiltonian::add_energy_current_interactions(
    jams::EnergyCurrentInteractionSink& sink) const {
  const auto offset_range_x = tensor_offset_range(kspace_size_[0], globals::lattice->is_periodic(0));
  const auto offset_range_y = tensor_offset_range(kspace_size_[1], globals::lattice->is_periodic(1));
  const auto offset_range_z = tensor_offset_range(kspace_size_[2], globals::lattice->is_periodic(2));

  for (auto cell_i_x = 0; cell_i_x < kspace_size_[0]; ++cell_i_x) {
    for (auto cell_i_y = 0; cell_i_y < kspace_size_[1]; ++cell_i_y) {
      for (auto cell_i_z = 0; cell_i_z < kspace_size_[2]; ++cell_i_z) {
        for (auto pos_i = 0; pos_i < globals::lattice->num_basis_sites(); ++pos_i) {
          const auto site_i = globals::lattice->site_index_by_unit_cell_optional(
              cell_i_x, cell_i_y, cell_i_z, pos_i);
          if (!site_i) {
            continue;
          }

          const auto r_frac_i = globals::lattice->basis_site_atom(pos_i).position_frac;
          const double mu_i = globals::lattice->material(
              globals::lattice->basis_site_atom(pos_i).material_index).moment;

          for (auto pos_j = 0; pos_j < globals::lattice->num_basis_sites(); ++pos_j) {
            const auto r_frac_j = globals::lattice->basis_site_atom(pos_j).position_frac;
            const auto r_cart_j = globals::lattice->fractional_to_cartesian(r_frac_j);
            const double mu_j = globals::lattice->material(
                globals::lattice->basis_site_atom(pos_j).material_index).moment;

            for (auto dx = offset_range_x.begin; dx < offset_range_x.end; ++dx) {
              for (auto dy = offset_range_y.begin; dy < offset_range_y.end; ++dy) {
                for (auto dz = offset_range_z.begin; dz < offset_range_z.end; ++dz) {
                  if (dx == 0 && dy == 0 && dz == 0 && pos_i == pos_j) {
                    continue;
                  }

                  auto cell_j = jams::Vec<int, 3>{cell_i_x - dx, cell_i_y - dy, cell_i_z - dz};
                  if (!globals::lattice->apply_boundary_conditions(cell_j)) {
                    continue;
                  }

                  const auto site_j = globals::lattice->site_index_by_unit_cell_optional(
                      cell_j[0], cell_j[1], cell_j[2], pos_j);
                  if (!site_j) {
                    continue;
                  }

                  const auto r_ij = globals::lattice->displacement(
                      r_cart_j,
                      globals::lattice->generate_cartesian_lattice_position_from_fractional(
                          r_frac_i, {dx, dy, dz}));
                  const auto r_abs_sq = jams::norm_squared(r_ij);

                  if (!std::isnormal(r_abs_sq)) {
                    throw std::runtime_error(
                        "fatal error in CudaDipoleFFTHamiltonian::add_energy_current_interactions: r_abs_sq is not normal");
                  }

                  if (r_abs_sq > pow2(r_cutoff_ + distance_tolerance_)) {
                    continue;
                  }

                  const auto interaction = jams::dipole::interaction_tensor(
                      r_ij, mu_i, mu_j, globals::lattice->parameter());
                  jams::dipole::insert_displacement_weighted_interaction(
                      sink, *site_i, *site_j, r_ij, interaction);
                }
              }
            }
          }
        }
      }
    }
  }
}

void CudaDipoleFFTHamiltonian::calculate_fields(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) {

  if (use_dense_fft_buffers_) {
    const unsigned int num_dense_slots = fft_site_map_.size();
    const dim3 pack_block = {128, 1, 1};
    const dim3 pack_grid = cuda_grid_size(pack_block, {num_dense_slots, 1, 1});
    cuda_pack_lattice_vector_field<<<pack_grid, pack_block, 0, cuda_stream_.get()>>>(
        num_dense_slots,
        fft_site_map_.device_data(),
        spins.device_data(),
        rspace_s_dense_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

#if DO_MIXED_PRECISION
    CHECK_CUFFT_STATUS(cufftExecR2C(cuda_fft_s_rspace_to_kspace, const_cast<cufftReal*>(reinterpret_cast<const cufftReal*>(rspace_s_dense_.device_data())), kspace_s_.mutable_device_data()));
#else
    CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, const_cast<cufftDoubleReal*>(reinterpret_cast<const cufftDoubleReal*>(rspace_s_dense_.device_data())), kspace_s_.mutable_device_data()));
#endif
  } else {
#if DO_MIXED_PRECISION
    CHECK_CUFFT_STATUS(cufftExecR2C(cuda_fft_s_rspace_to_kspace, const_cast<cufftReal*>(reinterpret_cast<const cufftReal*>(spins.device_data())), kspace_s_.mutable_device_data()));
#else
    CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, const_cast<cufftDoubleReal*>(reinterpret_cast<const cufftDoubleReal*>(spins.device_data())), kspace_s_.mutable_device_data()));
#endif
  }

  unsigned int num_pos = globals::lattice->num_basis_sites();
  const unsigned int fft_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);
  const dim3 block_size = {64, 1, 1};
  const dim3 grid_size = cuda_grid_size(block_size, {fft_size, num_pos, 1});


  cuda_dipole_convolution<<<grid_size, block_size, 0, cuda_stream_.get()>>>(
      fft_size,
      num_pos,
      use_full_tensor_storage_,
      kDipoleTensorComponents,
      0,
      false,
      kspace_s_.device_data(),
      kspace_tensors_.device_data(),
      kspace_h_.mutable_device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

#if DO_MIXED_PRECISION
  CHECK_CUFFT_STATUS(cufftExecC2R(cuda_fft_h_kspace_to_rspace, const_cast<cufftComplex*>(kspace_h_.device_data()), reinterpret_cast<cufftReal*>(use_dense_fft_buffers_ ? rspace_h_dense_.mutable_device_data() : field_.mutable_device_data())));
#else
  CHECK_CUFFT_STATUS(cufftExecZ2D(cuda_fft_h_kspace_to_rspace, const_cast<cufftDoubleComplex*>(kspace_h_.device_data()), reinterpret_cast<cufftDoubleReal*>(use_dense_fft_buffers_ ? rspace_h_dense_.mutable_device_data() : field_.mutable_device_data())));
#endif

  if (use_dense_fft_buffers_) {
    const unsigned int num_dense_slots = fft_site_map_.size();
    const dim3 unpack_block = {128, 1, 1};
    const dim3 unpack_grid = cuda_grid_size(unpack_block, {num_dense_slots, 1, 1});
    cuda_unpack_lattice_vector_field<<<unpack_grid, unpack_block, 0, cuda_stream_.get()>>>(
        num_dense_slots,
        fft_site_map_.device_data(),
        rspace_h_dense_.device_data(),
        field_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
  }

}

void CudaDipoleFFTHamiltonian::ensure_energy_current_tensors() {
  if (!kspace_energy_current_tensors_.empty()) {
    return;
  }

  const int num_sites = globals::lattice->num_basis_sites();
  const int num_kpoints = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);
  const int num_tensor_sets = use_full_tensor_storage_
      ? num_sites * num_sites
      : num_sites * (num_sites + 1) / 2;

  std::cout << "    generating dipole FFT energy current tensors\n";
  kspace_energy_current_tensors_.resize(
      num_tensor_sets,
      kEnergyCurrentTensorComponents,
      num_kpoints);
  kspace_energy_current_tensors_.zero();

  if (use_full_tensor_storage_) {
    for (int pos_i = 0; pos_i < num_sites; ++pos_i) {
      for (int pos_j = 0; pos_j < num_sites; ++pos_j) {
        const int tensor_set = pos_i * num_sites + pos_j;
        generate_kspace_energy_current_tensor(pos_i, pos_j, tensor_set);
      }
    }
  } else {
    for (int pos_i = 0; pos_i < num_sites; ++pos_i) {
      for (int pos_j = pos_i; pos_j < num_sites; ++pos_j) {
        const int tensor_set = upper_tri_index(pos_i, pos_j, num_sites);
        // The compact convolution reads upper-triangular cross-pair tensors
        // directly for the reversed motif access and reconstructs the forward
        // access by conjugation. Store the weighted kernel in that orientation.
        if (pos_i == pos_j) {
          generate_kspace_energy_current_tensor(pos_i, pos_j, tensor_set);
        } else {
          generate_kspace_energy_current_tensor(pos_j, pos_i, tensor_set);
        }
      }
    }
  }
}

std::size_t CudaDipoleFFTHamiltonian::energy_current_tensor_memory() const {
  return kspace_energy_current_tensors_.bytes();
}

void CudaDipoleFFTHamiltonian::calculate_energy_current_fields(
    const jams::MultiArray<jams::Real, 2>& spins,
    jams::MultiArray<jams::Real, 2>& energy_current_rx,
    jams::MultiArray<jams::Real, 2>& energy_current_ry,
    jams::MultiArray<jams::Real, 2>& energy_current_rz) {
  ensure_energy_current_tensors();

  energy_current_rx.resize(globals::num_spins, 3);
  energy_current_ry.resize(globals::num_spins, 3);
  energy_current_rz.resize(globals::num_spins, 3);

  if (use_dense_fft_buffers_) {
    const unsigned int num_dense_slots = fft_site_map_.size();
    const dim3 pack_block = {128, 1, 1};
    const dim3 pack_grid = cuda_grid_size(pack_block, {num_dense_slots, 1, 1});
    cuda_pack_lattice_vector_field<<<pack_grid, pack_block, 0, cuda_stream_.get()>>>(
        num_dense_slots,
        fft_site_map_.device_data(),
        spins.device_data(),
        rspace_s_dense_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

#if DO_MIXED_PRECISION
    CHECK_CUFFT_STATUS(cufftExecR2C(cuda_fft_s_rspace_to_kspace, const_cast<cufftReal*>(reinterpret_cast<const cufftReal*>(rspace_s_dense_.device_data())), kspace_s_.mutable_device_data()));
#else
    CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, const_cast<cufftDoubleReal*>(reinterpret_cast<const cufftDoubleReal*>(rspace_s_dense_.device_data())), kspace_s_.mutable_device_data()));
#endif
  } else {
#if DO_MIXED_PRECISION
    CHECK_CUFFT_STATUS(cufftExecR2C(cuda_fft_s_rspace_to_kspace, const_cast<cufftReal*>(reinterpret_cast<const cufftReal*>(spins.device_data())), kspace_s_.mutable_device_data()));
#else
    CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, const_cast<cufftDoubleReal*>(reinterpret_cast<const cufftDoubleReal*>(spins.device_data())), kspace_s_.mutable_device_data()));
#endif
  }

  unsigned int num_pos = globals::lattice->num_basis_sites();
  const unsigned int fft_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);
  const dim3 block_size = {64, 1, 1};
  const dim3 grid_size = cuda_grid_size(block_size, {fft_size, num_pos, 1});

  for (int direction = 0; direction < kEnergyCurrentDirections; ++direction) {
    cuda_dipole_convolution<<<grid_size, block_size, 0, cuda_stream_.get()>>>(
        fft_size,
        num_pos,
        use_full_tensor_storage_,
        kEnergyCurrentTensorComponents,
        direction * kDipoleTensorComponents,
        true,
        kspace_s_.device_data(),
        kspace_energy_current_tensors_.device_data(),
        kspace_h_.mutable_device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

    jams::MultiArray<jams::Real, 2>* output = &energy_current_rx;
    if (direction == 1) {
      output = &energy_current_ry;
    } else if (direction == 2) {
      output = &energy_current_rz;
    }

#if DO_MIXED_PRECISION
    CHECK_CUFFT_STATUS(cufftExecC2R(
        cuda_fft_h_kspace_to_rspace,
        const_cast<cufftComplex*>(kspace_h_.device_data()),
        reinterpret_cast<cufftReal*>(use_dense_fft_buffers_
            ? rspace_h_dense_.mutable_device_data()
            : output->mutable_device_data())));
#else
    CHECK_CUFFT_STATUS(cufftExecZ2D(
        cuda_fft_h_kspace_to_rspace,
        const_cast<cufftDoubleComplex*>(kspace_h_.device_data()),
        reinterpret_cast<cufftDoubleReal*>(use_dense_fft_buffers_
            ? rspace_h_dense_.mutable_device_data()
            : output->mutable_device_data())));
#endif

    if (use_dense_fft_buffers_) {
      const unsigned int num_dense_slots = fft_site_map_.size();
      const dim3 unpack_block = {128, 1, 1};
      const dim3 unpack_grid = cuda_grid_size(unpack_block, {num_dense_slots, 1, 1});
      cuda_unpack_lattice_vector_field<<<unpack_grid, unpack_block, 0, cuda_stream_.get()>>>(
          num_dense_slots,
          fft_site_map_.device_data(),
          rspace_h_dense_.device_data(),
          output->mutable_device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
    }
  }

  record_done();
}

// Generates the dipole tensor between unit cell positions i and j and appends
// the generated positions to a vector
void CudaDipoleFFTHamiltonian::generate_kspace_dipole_tensor(const int pos_i, const int pos_j, const int pair, std::vector<jams::Vec<double, 3>> &generated_positions) {
    const jams::Vec<double, 3> r_frac_i = globals::lattice->basis_site_atom(pos_i).position_frac;
    const jams::Vec<double, 3> r_frac_j = globals::lattice->basis_site_atom(pos_j).position_frac;

    const jams::Vec<double, 3> r_cart_j = globals::lattice->fractional_to_cartesian(r_frac_j);

    const int num_kz = kspace_padded_size_[2] / 2 + 1;
    const int num_ky = kspace_padded_size_[1];

    const double mu_i = globals::lattice->material(globals::lattice->basis_site_atom(pos_i).material_index).moment;
    const double mu_j = globals::lattice->material(globals::lattice->basis_site_atom(pos_j).material_index).moment;
    const double fft_normalization_factor = 1.0 / jams::product(kspace_padded_size_);

    const auto offset_range_x = tensor_offset_range(kspace_size_[0], globals::lattice->is_periodic(0));
    const auto offset_range_y = tensor_offset_range(kspace_size_[1], globals::lattice->is_periodic(1));
    const auto offset_range_z = tensor_offset_range(kspace_size_[2], globals::lattice->is_periodic(2));

    for (int dx = offset_range_x.begin; dx < offset_range_x.end; ++dx) {
        for (int dy = offset_range_y.begin; dy < offset_range_y.end; ++dy) {
            for (int dz = offset_range_z.begin; dz < offset_range_z.end; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0 && pos_i == pos_j) {
                    // self interaction on the same sublattice
                    continue;
                } 

                auto r_ij =
                    globals::lattice->displacement(r_cart_j,
                                                   globals::lattice->generate_cartesian_lattice_position_from_fractional(r_frac_i,
                                                                                                                         {dx, dy, dz})); // generate_cartesian_lattice_position_from_fractional requires FRACTIONAL coordinate

                const auto r_abs_sq = jams::norm_squared(r_ij);

                if (!std::isnormal(r_abs_sq)) {
                  throw std::runtime_error("fatal error in CudaDipoleFFTHamiltonian::generate_kspace_dipole_tensor: r_abs_sq is not normal");
                }

                if (r_abs_sq > pow2(r_cutoff_ + distance_tolerance_)) {
                    continue;
                }

                generated_positions.push_back(r_ij);

                const auto interaction = jams::dipole::interaction_tensor(
                    r_ij, mu_i, mu_j, globals::lattice->parameter(), fft_normalization_factor);

                const jams::ComplexHi phase_step_x = std::polar(1.0, -kTwoPi * static_cast<double>(dx) / static_cast<double>(kspace_padded_size_[0]));
                const jams::ComplexHi phase_step_y = std::polar(1.0, -kTwoPi * static_cast<double>(dy) / static_cast<double>(kspace_padded_size_[1]));
                const jams::ComplexHi phase_step_z = std::polar(1.0, -kTwoPi * static_cast<double>(dz) / static_cast<double>(kspace_padded_size_[2]));

                jams::ComplexHi phase_x = {1.0, 0.0};
                for (int h = 0; h < kspace_padded_size_[0]; ++h) {
                  jams::ComplexHi phase_y = phase_x;
                  for (int k = 0; k < kspace_padded_size_[1]; ++k) {
                    jams::ComplexHi phase = phase_y;
                    for (int l = 0; l < num_kz; ++l) {
                      const int k_idx = (h * num_ky + k) * num_kz + l;
                      const jams::ComplexHi k_xx = interaction[0][0] * phase;
                      const jams::ComplexHi k_xy = interaction[0][1] * phase;
                      const jams::ComplexHi k_xz = interaction[0][2] * phase;
                      const jams::ComplexHi k_yy = interaction[1][1] * phase;
                      const jams::ComplexHi k_yz = interaction[1][2] * phase;
                      const jams::ComplexHi k_zz = interaction[2][2] * phase;
#if DO_MIXED_PRECISION
                      kspace_tensors_(pair, 0, k_idx) += make_cuComplex(static_cast<float>(k_xx.real()), static_cast<float>(k_xx.imag()));
                      kspace_tensors_(pair, 1, k_idx) += make_cuComplex(static_cast<float>(k_xy.real()), static_cast<float>(k_xy.imag()));
                      kspace_tensors_(pair, 2, k_idx) += make_cuComplex(static_cast<float>(k_xz.real()), static_cast<float>(k_xz.imag()));
                      kspace_tensors_(pair, 3, k_idx) += make_cuComplex(static_cast<float>(k_yy.real()), static_cast<float>(k_yy.imag()));
                      kspace_tensors_(pair, 4, k_idx) += make_cuComplex(static_cast<float>(k_yz.real()), static_cast<float>(k_yz.imag()));
                      kspace_tensors_(pair, 5, k_idx) += make_cuComplex(static_cast<float>(k_zz.real()), static_cast<float>(k_zz.imag()));
#else
                      kspace_tensors_(pair, 0, k_idx) += make_cuDoubleComplex(k_xx.real(), k_xx.imag());
                      kspace_tensors_(pair, 1, k_idx) += make_cuDoubleComplex(k_xy.real(), k_xy.imag());
                      kspace_tensors_(pair, 2, k_idx) += make_cuDoubleComplex(k_xz.real(), k_xz.imag());
                      kspace_tensors_(pair, 3, k_idx) += make_cuDoubleComplex(k_yy.real(), k_yy.imag());
                      kspace_tensors_(pair, 4, k_idx) += make_cuDoubleComplex(k_yz.real(), k_yz.imag());
                      kspace_tensors_(pair, 5, k_idx) += make_cuDoubleComplex(k_zz.real(), k_zz.imag());
#endif
                      phase *= phase_step_z;
                    }
                    phase_y *= phase_step_y;
                  }
                  phase_x *= phase_step_x;
                }
            }
        }
    }
  
    if (debug_) {
      std::ofstream debugfile(jams::output::hamiltonian_filename(
          name(),
          "DEBUG_" + std::to_string(pos_i) + "_" + std::to_string(pos_j) + "_rij",
          "tsv"));

      for (const auto& r : generated_positions) {
        debugfile << r << "\n";
      }
    }
}

void CudaDipoleFFTHamiltonian::generate_kspace_energy_current_tensor(
    const int pos_i,
    const int pos_j,
    const int pair) {
    const jams::Vec<double, 3> r_frac_i = globals::lattice->basis_site_atom(pos_i).position_frac;
    const jams::Vec<double, 3> r_frac_j = globals::lattice->basis_site_atom(pos_j).position_frac;

    const jams::Vec<double, 3> r_cart_j = globals::lattice->fractional_to_cartesian(r_frac_j);

    const int num_kz = kspace_padded_size_[2] / 2 + 1;
    const int num_ky = kspace_padded_size_[1];

    const double mu_i = globals::lattice->material(globals::lattice->basis_site_atom(pos_i).material_index).moment;
    const double mu_j = globals::lattice->material(globals::lattice->basis_site_atom(pos_j).material_index).moment;
    const double fft_normalization_factor = 1.0 / jams::product(kspace_padded_size_);

    const auto offset_range_x = tensor_offset_range(kspace_size_[0], globals::lattice->is_periodic(0));
    const auto offset_range_y = tensor_offset_range(kspace_size_[1], globals::lattice->is_periodic(1));
    const auto offset_range_z = tensor_offset_range(kspace_size_[2], globals::lattice->is_periodic(2));

    for (int dx = offset_range_x.begin; dx < offset_range_x.end; ++dx) {
        for (int dy = offset_range_y.begin; dy < offset_range_y.end; ++dy) {
            for (int dz = offset_range_z.begin; dz < offset_range_z.end; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0 && pos_i == pos_j) {
                    continue;
                }

                const auto r_ji =
                    globals::lattice->displacement(
                        r_cart_j,
                        globals::lattice->generate_cartesian_lattice_position_from_fractional(
                            r_frac_i, {dx, dy, dz}));

                const auto r_abs_sq = jams::norm_squared(r_ji);

                if (!std::isnormal(r_abs_sq)) {
                  throw std::runtime_error("fatal error in CudaDipoleFFTHamiltonian::generate_kspace_energy_current_tensor: r_abs_sq is not normal");
                }

                if (r_abs_sq > pow2(r_cutoff_ + distance_tolerance_)) {
                    continue;
                }

                const auto interaction = jams::dipole::interaction_tensor(
                    r_ji, mu_i, mu_j, globals::lattice->parameter(), fft_normalization_factor);

                const jams::ComplexHi phase_step_x = std::polar(1.0, -kTwoPi * static_cast<double>(dx) / static_cast<double>(kspace_padded_size_[0]));
                const jams::ComplexHi phase_step_y = std::polar(1.0, -kTwoPi * static_cast<double>(dy) / static_cast<double>(kspace_padded_size_[1]));
                const jams::ComplexHi phase_step_z = std::polar(1.0, -kTwoPi * static_cast<double>(dz) / static_cast<double>(kspace_padded_size_[2]));

                jams::ComplexHi phase_x = {1.0, 0.0};
                for (int h = 0; h < kspace_padded_size_[0]; ++h) {
                  jams::ComplexHi phase_y = phase_x;
                  for (int k = 0; k < kspace_padded_size_[1]; ++k) {
                    jams::ComplexHi phase = phase_y;
                    for (int l = 0; l < num_kz; ++l) {
                      const int k_idx = (h * num_ky + k) * num_kz + l;
                      const jams::ComplexHi components[kDipoleTensorComponents] = {
                          interaction[0][0] * phase,
                          interaction[0][1] * phase,
                          interaction[0][2] * phase,
                          interaction[1][1] * phase,
                          interaction[1][2] * phase,
                          interaction[2][2] * phase
                      };

                      for (int direction = 0; direction < kEnergyCurrentDirections; ++direction) {
                        const double displacement_weight = r_ji[direction];
                        const int component_offset = direction * kDipoleTensorComponents;
                        for (int component = 0; component < kDipoleTensorComponents; ++component) {
                          const auto weighted_component = displacement_weight * components[component];
#if DO_MIXED_PRECISION
                          kspace_energy_current_tensors_(pair, component_offset + component, k_idx) +=
                              make_cuComplex(
                                  static_cast<float>(weighted_component.real()),
                                  static_cast<float>(weighted_component.imag()));
#else
                          kspace_energy_current_tensors_(pair, component_offset + component, k_idx) +=
                              make_cuDoubleComplex(weighted_component.real(), weighted_component.imag());
#endif
                        }
                      }

                      phase *= phase_step_z;
                    }
                    phase_y *= phase_step_y;
                  }
                  phase_x *= phase_step_x;
                }
            }
        }
    }
}
