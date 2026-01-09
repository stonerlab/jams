#include <fstream>

#include <libconfig.h++>
#include <cufft.h>

#include "jams/interface/fft.h"
#include "jams/helpers/exception.h"

#include "jams/helpers/output.h"
#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_device_complex_ops.h"
#include "jams/hamiltonian/cuda_dipole_fft.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_array_kernels.h"
#include <jams/helpers/mixed_precision.h>


__constant__ jams::Real mu_const[128];

// Pack upper-triangular (i<=j) pairs into a 1D index.
// Number of pairs = n*(n+1)/2.
__host__ __device__ __forceinline__ int upper_tri_index(const int i, const int j, const int n) {
  // Requires: 0 <= i <= j < n
  return i * n - (i * (i - 1)) / 2 + (j - i);
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

template<typename ComplexType>
__global__ void cuda_dipole_convolution(
  const unsigned int num_kpoints,
  const unsigned int num_pos,
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
    const jams::Real mu_j = mu_const[pos_j];

    int batch_base_j = 3 * pos_j;
    int idx0 = (batch_base_j + 0) * num_kpoints + k_idx;
    int idx1 = (batch_base_j + 1) * num_kpoints + k_idx;
    int idx2 = (batch_base_j + 2) * num_kpoints + k_idx;

    const ComplexType sq0 = sk[idx0];
    const ComplexType sq1 = sk[idx1];
    const ComplexType sq2 = sk[idx2];

    // wk is stored only for i<=j (upper-triangular in (pos_i,pos_j))
    const int a = (pos_i <= pos_j) ? pos_i : pos_j;
    const int b = (pos_i <= pos_j) ? pos_j : pos_i;
    const bool swapped = (pos_i > pos_j);

    const int pair = upper_tri_index(a, b, (int)num_pos);

    int base0 = ((pair * 6 + 0) * (int)num_kpoints) + (int)k_idx;
    int base1 = base0 + (int)num_kpoints;
    int base2 = base1 + (int)num_kpoints;

    // Hermitian symmetry: W_{ji}(k) = conj(W_{ij}(k))
    // Staggering the accessing of w components gives about 1 us improvement on A30.
    ComplexType w0 = swapped ? wk[base0] : complex_conj(wk[base0]);
    ComplexType w1 = swapped ? wk[base1] : complex_conj(wk[base1]);
    ComplexType w2 = swapped ? wk[base2] : complex_conj(wk[base2]);

    hk_sum[0] +=  mu_j * (w0 * sq0 + w1 * sq1 + w2 * sq2);

    int base3 = base2 + (int)num_kpoints;
    int base4 = base3 + (int)num_kpoints;

    ComplexType w3 = swapped ? wk[base3] : complex_conj(wk[base3]);
    ComplexType w4 = swapped ? wk[base4] : complex_conj(wk[base4]);

    hk_sum[1] +=  mu_j * (w1 * sq0 + w3 * sq1 + w4 * sq2);

    int base5 = base4 + (int)num_kpoints;

    ComplexType w5 = swapped ? wk[base5] : complex_conj(wk[base5]);

    hk_sum[2] +=  mu_j * (w2 * sq0 + w4 * sq1 + w5 * sq2);
  }
  const jams::Real mu_i = mu_const[pos_i];

  int batch_base_i = 3 * pos_i;
  int out0 = (batch_base_i + 0) * num_kpoints + k_idx;
  int out1 = (batch_base_i + 1) * num_kpoints + k_idx;
  int out2 = (batch_base_i + 2) * num_kpoints + k_idx;

  hk[out0] = mu_i * hk_sum[0];
  hk[out1] = mu_i * hk_sum[1];
  hk[out2] = mu_i * hk_sum[2];
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
  kspace_size_({0, 0, 0}),
  kspace_padded_size_({0, 0, 0}),
  kspace_s_(),
  kspace_h_(),
  cuda_fft_s_rspace_to_kspace(),
  cuda_fft_h_kspace_to_rspace()
{
  settings.lookupValue("debug", debug_);
  settings.lookupValue("check_radius", check_radius_);
  settings.lookupValue("check_symmetry", check_symmetry_);

  r_cutoff_ = double(settings["r_cutoff"]);
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";
  std::cout << "  r_cutoff_max " << ::globals::lattice->max_interaction_radius() << "\n";

  if (check_radius_) {
    if (r_cutoff_ > ::globals::lattice->max_interaction_radius()) {
      throw std::runtime_error("CudaDipoleFFTHamiltonian r_cutoff is too large for the lattice size."
                                       "The cutoff must be less than the inradius of the lattice.");
    }
  }

  settings.lookupValue("distance_tolerance", distance_tolerance_);
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
  int stride          = 3 * num_sites;
  int dist            = 1;
  int rspace_embed[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
  int kspace_embed[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]/2 + 1};

  int fft_size[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
  int fft_padded_size[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]};

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

  s_float_.resize(globals::s.size(0), globals::s.size(1));

  const auto num_tensor_components = 6;

  const int num_pairs = num_sites * (num_sites + 1) / 2;
  kspace_tensors_.resize(num_pairs, num_tensor_components, num_kpoints);
  for (int pos_i = 0; pos_i < num_sites; ++pos_i) {
    for (int pos_j = pos_i; pos_j < num_sites; ++pos_j) {
      std::vector<Vec3> generated_positions;
      auto wq = generate_kspace_dipole_tensor(pos_i, pos_j, generated_positions);

      assert(wq.size(1) * wq.size(2) * wq.size(3) == kspace_tensors_.size(2));

      const int pair = upper_tri_index(pos_i, pos_j, num_sites);
      for (auto m = 0; m < num_tensor_components; ++m) {
        for (auto h = 0; h < wq.size(1); ++h) {
          for (auto k = 0; k < wq.size(2); ++k) {
            for (auto l = 0; l < wq.size(3); ++l) {
              auto k_idx = (h * kspace_embed[1] + k) * kspace_embed[2] + l;
#if DO_MIXED_PRECISION
              kspace_tensors_(pair, m, k_idx) = make_cuComplex(wq(m, h, k, l).real(), wq(m, h, k, l).imag());
#else
              kspace_tensors_(pair, m, k_idx) = make_cuDoubleComplex(wq(m, h, k, l).real(), wq(m, h, k, l).imag());
#endif
            }
          }
        }
      }

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

  mus_unitcell_.resize(num_sites);
  for (auto i = 0; i < num_sites; ++i) {
    mus_unitcell_(i) = globals::lattice->material(globals::lattice->basis_site_atom(i).material_index).moment;
  }

  cudaMemcpyToSymbol(mu_const, mus_unitcell_.device_data(), mus_unitcell_.bytes(), 0, cudaMemcpyHostToDevice);

  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_s_rspace_to_kspace, cuda_stream_.get()));
  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_h_kspace_to_rspace, cuda_stream_.get()));
}

jams::Real CudaDipoleFFTHamiltonian::calculate_total_energy(jams::Real time) {
  calculate_energies(time);
  return cuda_reduce_array(energy_.device_data(), globals::num_spins, cuda_stream_.get());
}

jams::Real CudaDipoleFFTHamiltonian::calculate_one_spin_energy(const int i, const Vec3 &s_i, jams::Real time) {
    throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_one_spin_energy is not implemented");
}

jams::Real CudaDipoleFFTHamiltonian::calculate_energy(const int i, jams::Real time) {
    throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_energy is not implemented");
}

jams::Real CudaDipoleFFTHamiltonian::calculate_energy_difference(
    int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) {
  throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_energy_difference is not implemented");
}

void CudaDipoleFFTHamiltonian::calculate_energies(jams::Real time) {
  calculate_fields(time);
  const auto minus_half = static_cast<jams::Real>(-0.5);
  cuda_array_dot_product(globals::num_spins, minus_half, globals::s.device_data(), field_.device_data(), energy_.device_data(), cuda_stream_.get());
}

Vec3R CudaDipoleFFTHamiltonian::calculate_field(const int i, jams::Real time) {
  throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_field is not implemented");
}

void CudaDipoleFFTHamiltonian::calculate_fields(jams::Real time) {

#if DO_MIXED_PRECISION
  cuda_array_double_to_float(globals::s.elements(), globals::s.device_data(), s_float_.device_data(), cuda_stream_.get());
  CHECK_CUFFT_STATUS(cufftExecR2C(cuda_fft_s_rspace_to_kspace, reinterpret_cast<cufftReal*>(s_float_.device_data()), kspace_s_.device_data()));
#else
  CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, reinterpret_cast<cufftDoubleReal*>(globals::s.device_data()), kspace_s_.device_data()));
#endif

  unsigned int num_pos = globals::lattice->num_basis_sites();
const unsigned int fft_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);
const dim3 block_size = {64, 1, 1};
const dim3 grid_size = cuda_grid_size(block_size, {fft_size, num_pos, 1});


cuda_dipole_convolution<<<grid_size, block_size, 0, cuda_stream_.get()>>>(fft_size, num_pos, kspace_s_.device_data(), kspace_tensors_.device_data(), kspace_h_.device_data());
DEBUG_CHECK_CUDA_ASYNC_STATUS;

#ifdef DO_MIXED_PRECISION
  CHECK_CUFFT_STATUS(cufftExecC2R(cuda_fft_h_kspace_to_rspace, kspace_h_.device_data(), reinterpret_cast<cufftReal*>(field_.device_data())));
#else
  CHECK_CUFFT_STATUS(cufftExecZ2D(cuda_fft_h_kspace_to_rspace, kspace_h_.device_data(), reinterpret_cast<cufftDoubleReal*>(field_.device_data())));
#endif

}

// Generates the dipole tensor between unit cell positions i and j and appends
// the generated positions to a vector
jams::MultiArray<jams::ComplexLo, 4>
CudaDipoleFFTHamiltonian::generate_kspace_dipole_tensor(const int pos_i, const int pos_j, std::vector<Vec3> &generated_positions) {
    using std::pow;
  
    const Vec3 r_frac_i = globals::lattice->basis_site_atom(pos_i).position_frac;
    const Vec3 r_frac_j = globals::lattice->basis_site_atom(pos_j).position_frac;

    const Vec3 r_cart_i = globals::lattice->fractional_to_cartesian(r_frac_i);
    const Vec3 r_cart_j = globals::lattice->fractional_to_cartesian(r_frac_j);

  jams::MultiArray<double, 4> rspace_tensor(
        6,
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2]);

  jams::MultiArray<jams::ComplexHi, 4> kspace_tensor_hi(
        6,
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2]/2 + 1);


    rspace_tensor.zero();
    kspace_tensor_hi.zero();

    const double fft_normalization_factor = 1.0 / product(kspace_size_);
    const double v = pow(globals::lattice->parameter(), 3);
    const double w0 = fft_normalization_factor * kVacuumPermeabilityIU / (4.0 * kPi * v);

    for (int nx = 0; nx < kspace_size_[0]; ++nx) {
        for (int ny = 0; ny < kspace_size_[1]; ++ny) {
            for (int nz = 0; nz < kspace_size_[2]; ++nz) {
                if (nx == 0 && ny == 0 && nz == 0 && pos_i == pos_j) {
                    // self interaction on the same sublattice
                    continue;
                } 

                auto r_ij =
                    globals::lattice->displacement(r_cart_j,
                                                   globals::lattice->generate_cartesian_lattice_position_from_fractional(r_frac_i,
                                                                                                                         {nx, ny, nz})); // generate_cartesian_lattice_position_from_fractional requires FRACTIONAL coordinate

                const auto r_abs_sq = norm_squared(r_ij);

                if (!std::isnormal(r_abs_sq)) {
                  throw std::runtime_error("fatal error in CudaDipoleFFTHamiltonian::generate_kspace_dipole_tensor: r_abs_sq is not normal");
                }

                if (r_abs_sq > pow2(r_cutoff_ + distance_tolerance_)) {
                    continue;
                }

                const double r_pow_5_2 = r_abs_sq * r_abs_sq * std::sqrt(r_abs_sq);

                generated_positions.push_back(r_ij);

                // xx
                rspace_tensor(0, nx, ny, nz) =  w0 * (3 * r_ij[0] * r_ij[0] - r_abs_sq) / r_pow_5_2;
                // xy
                rspace_tensor(1, nx, ny, nz) =  w0 * (3 * r_ij[0] * r_ij[1]) / r_pow_5_2;
                // xz
                rspace_tensor(2, nx, ny, nz) =  w0 * (3 * r_ij[0] * r_ij[2]) / r_pow_5_2;
                // yy
                rspace_tensor(3, nx, ny, nz) =  w0 * (3 * r_ij[1] * r_ij[1] - r_abs_sq) / r_pow_5_2;
                // yz
                rspace_tensor(4, nx, ny, nz) =  w0 * (3 * r_ij[1] * r_ij[2]) / r_pow_5_2;
                // zz
                rspace_tensor(5, nx, ny, nz) =  w0 * (3 * r_ij[2] * r_ij[2] - r_abs_sq) / r_pow_5_2;
            }
        }
    }
  
    if (debug_) {
      std::ofstream debugfile(jams::output::full_path_filename(
          "DEBUG_dipole_fft_" + std::to_string(pos_i) + "_" + std::to_string(pos_j) + "_rij.tsv"));

      for (const auto& r : generated_positions) {
        debugfile << r << "\n";
      }
    }

  int rank           = 3;
  int istride        = 1;
  int ostride        = 1;
  int idist          = kspace_padded_size_[0]*kspace_padded_size_[1]*kspace_padded_size_[2];         // Nx*Ny*Nz
  int odist          = kspace_padded_size_[0]*kspace_padded_size_[1]*(kspace_padded_size_[2]/2 + 1); // Nx*Ny*(Nz/2+1)
  int num_transforms = 6;
  int *inembed       = nullptr;
  int *outembed      = nullptr;
  int transform_size[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]};

  fftw_plan fft_dipole_tensor_rspace_to_kspace =
      fftw_plan_many_dft_r2c(
          rank,
          transform_size,
          num_transforms,
          rspace_tensor.data(),
          inembed,
          istride,
          idist,
          FFTW_COMPLEX_CAST(kspace_tensor_hi.data()),
          outembed,
          ostride,
          odist,
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);

  jams::MultiArray<jams::ComplexLo, 4> kspace_tensor_lo(
    6,
    kspace_padded_size_[0],
    kspace_padded_size_[1],
    kspace_padded_size_[2]/2 + 1);

    for (auto i = 0; i < kspace_tensor_hi.size(0); ++i) {
      for (auto j = 0; j < kspace_tensor_hi.size(1); ++j) {
        for (auto k = 0; k < kspace_tensor_hi.size(2); ++k) {
          for (auto l = 0; l < kspace_tensor_hi.size(3); ++l) {
            kspace_tensor_lo(i, j, k, l) = kspace_tensor_hi(i, j, k, l);
          }
        }
      }
    }

    return kspace_tensor_lo;
}

