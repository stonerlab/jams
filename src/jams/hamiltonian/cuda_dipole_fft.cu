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

template<typename RealType, typename ComplexType>
__global__ void cuda_dipole_convolution(
  const unsigned int num_kpoints,
  const unsigned int num_pos,
  const RealType* mu,
  const ComplexType* sk,
  const ComplexType* wk,
  ComplexType* hk
)
{
  unsigned int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int pos_i = blockIdx.y;

  if (k_idx >= num_kpoints || pos_i >= num_pos) return;

  ComplexType hk_sum[3] = {0.0, 0.0, 0.0};

  int offset_i = 3 * (num_pos * k_idx + pos_i);
  for (int pos_j = 0; pos_j < num_pos; ++pos_j) {
    int offset_j = 3 * (num_pos * k_idx + pos_j);
    int offset_w = ((pos_i*num_pos + pos_j)*num_kpoints + k_idx)*6;
    const ComplexType sq[3] = {sk[offset_j + 0], sk[offset_j + 1], sk[offset_j + 2]};

    const ComplexType w[6] = {
      wk[offset_w + 0], wk[offset_w + 1], wk[offset_w + 2], wk[offset_w + 3], wk[offset_w + 4], wk[offset_w + 5]
    };

    RealType mu_j = __ldg(mu + pos_j);
    hk_sum[0] +=  mu_j * (w[0] * sq[0] + w[1] * sq[1] + w[2] * sq[2]);
    hk_sum[1] +=  mu_j * (w[1] * sq[0] + w[3] * sq[1] + w[4] * sq[2]);
    hk_sum[2] +=  mu_j * (w[2] * sq[0] + w[4] * sq[1] + w[5] * sq[2]);
  }

  RealType mu_i = __ldg(mu + pos_i);
  hk[offset_i + 0] = mu_i * hk_sum[0];
  hk[offset_i + 1] = mu_i * hk_sum[1];
  hk[offset_i + 2] = mu_i * hk_sum[2];
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
  dev_stream_(),
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

  int rank            = 3;           
  int stride          = 3 * globals::lattice->num_basis_sites();
  int dist            = 1;
  int num_transforms  = 3 * globals::lattice->num_basis_sites();
  int rspace_embed[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
  int kspace_embed[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]/2 + 1};

  int fft_size[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
  int fft_padded_size[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]};


#ifdef DO_MIXED_PRECISION
  CHECK_CUFFT_STATUS(cufftPlanMany(&cuda_fft_s_rspace_to_kspace, rank, fft_size, rspace_embed, stride, dist,
      kspace_embed, stride, dist, CUFFT_R2C, num_transforms));

  CHECK_CUFFT_STATUS(cufftPlanMany(&cuda_fft_h_kspace_to_rspace, rank, fft_size, kspace_embed, stride, dist,
          rspace_embed, stride, dist, CUFFT_C2R, num_transforms));

  s_float_.resize(globals::s.size(0), globals::s.size(1));
  h_float_.resize(globals::h.size(0), globals::h.size(1));

#else
  CHECK_CUFFT_STATUS(cufftPlanMany(&cuda_fft_s_rspace_to_kspace, rank, fft_size, rspace_embed, stride, dist,
        kspace_embed, stride, dist, CUFFT_D2Z, num_transforms));

  CHECK_CUFFT_STATUS(cufftPlanMany(&cuda_fft_h_kspace_to_rspace, rank, fft_size, kspace_embed, stride, dist,
          rspace_embed, stride, dist, CUFFT_Z2D, num_transforms));
#endif

  const auto num_sites = globals::lattice->num_basis_sites();
  const auto num_kpoints = kspace_embed[0] * kspace_embed[1] * kspace_embed[2];
  const auto num_tensor_components = 6;

  kspace_tensors_.resize(num_sites, num_sites, num_kpoints, num_tensor_components);
  for (int pos_i = 0; pos_i < num_sites; ++pos_i) {
    std::vector<Vec3> generated_positions;
    for (int pos_j = 0; pos_j < num_sites; ++pos_j) {
        auto wq = generate_kspace_dipole_tensor(pos_i, pos_j, generated_positions);

        assert(wq.size(0)*wq.size(1)*wq.size(2) == kspace_tensors_.size(2));
        for (auto h = 0; h < wq.size(0); ++h) {
          for (auto k = 0; k < wq.size(1); ++k) {
            for (auto l = 0; l < wq.size(2); ++l) {
              for (auto m = 0; m < num_tensor_components; ++m) {
                auto k_idx = (h*kspace_embed[1] + k)*kspace_embed[2] + l;
#ifdef DO_MIXED_PRECISION
                kspace_tensors_(pos_i, pos_j, k_idx, m) = make_cuComplex(wq(h, k, l, m).real(), wq(h, k, l, m).imag());
#else
                kspace_tensors_(pos_i, pos_j, k_idx, m) = make_cuDoubleComplex(wq(h, k, l, m).real(), wq(h, k, l, m).imag());
#endif
              }
            }
          }
        }

    }
      if (check_symmetry_ && (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2))) {
        if (!globals::lattice->is_a_symmetry_complete_set(pos_i, generated_positions, distance_tolerance_)) {
          throw std::runtime_error("The points included in the dipole tensor do not form set of all symmetric points.\n"
                                   "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
                                   "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
        }
      }
  }

  mus_float_.resize(num_sites);
  for (auto i = 0; i < num_sites; ++i) {
    mus_float_(i) = globals::lattice->material(globals::lattice->basis_site_atom(i).material_index).moment;
  }



  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_s_rspace_to_kspace, dev_stream_.get()));
  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_h_kspace_to_rspace, dev_stream_.get()));
}

double CudaDipoleFFTHamiltonian::calculate_total_energy(double time) {
  calculate_fields(time);

  double e_total = 0.0;
  for (auto i = 0; i < globals::num_spins; ++i) {
      e_total += (  globals::s(i,0)*field_(i, 0)
                  + globals::s(i,1)*field_(i, 1)
                  + globals::s(i,2)*field_(i, 2) );
  }

  return -0.5*e_total;
}

double CudaDipoleFFTHamiltonian::calculate_one_spin_energy(const int i, const Vec3 &s_i, double time) {
    return 0.0;
}

double CudaDipoleFFTHamiltonian::calculate_energy(const int i, double time) {
    return 0.0;
}

double CudaDipoleFFTHamiltonian::calculate_energy_difference(
    int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) {

    return 0.0;
}

void CudaDipoleFFTHamiltonian::calculate_energies(double time) {
  cuda_array_elementwise_scale(globals::num_spins, 3, globals::mus.device_data(), 1.0, field_.device_data(), 1, field_.device_data(), 1, dev_stream_.get());
}

Vec3 CudaDipoleFFTHamiltonian::calculate_field(const int i, double time) {
  throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_field");
}

void CudaDipoleFFTHamiltonian::calculate_fields(double time) {

#ifdef DO_MIXED_PRECISION
  cuda_array_double_to_float(globals::s.elements(), globals::s.device_data(), s_float_.device_data(), dev_stream_.get());
  CHECK_CUFFT_STATUS(cufftExecR2C(cuda_fft_s_rspace_to_kspace, reinterpret_cast<cufftReal*>(s_float_.device_data()), kspace_s_.device_data()));
#else
  CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, reinterpret_cast<cufftDoubleReal*>(globals::s.device_data()), kspace_s_.device_data()));
#endif

  unsigned int num_pos = globals::lattice->num_basis_sites();
const unsigned int fft_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);
const dim3 block_size = {256, 1, 1};
const dim3 grid_size = cuda_grid_size(block_size, {fft_size, num_pos, 1});


cuda_dipole_convolution<<<grid_size, block_size, 0, dev_stream_.get()>>>(fft_size, num_pos, mus_float_.device_data(), kspace_s_.device_data(), kspace_tensors_.device_data(), kspace_h_.device_data());
DEBUG_CHECK_CUDA_ASYNC_STATUS;

#ifdef DO_MIXED_PRECISION
  CHECK_CUFFT_STATUS(cufftExecC2R(cuda_fft_h_kspace_to_rspace, kspace_h_.device_data(), reinterpret_cast<cufftReal*>(h_float_.device_data())));
  cuda_array_float_to_double(h_float_.elements(), h_float_.device_data(), field_.device_data(), dev_stream_.get());
#else
  CHECK_CUFFT_STATUS(cufftExecZ2D(cuda_fft_h_kspace_to_rspace, kspace_h_.device_data(), reinterpret_cast<cufftDoubleReal*>(field_.device_data())));
#endif

  // cuda_array_elementwise_scale(globals::num_spins, 3, globals::mus.device_data(), 1.0, field_.device_data(), 1, field_.device_data(), 1, dev_stream_.get());
}

// Generates the dipole tensor between unit cell positions i and j and appends
// the generated positions to a vector
jams::MultiArray<CudaDipoleFFTHamiltonian::ComplexLo, 4>
CudaDipoleFFTHamiltonian::generate_kspace_dipole_tensor(const int pos_i, const int pos_j, std::vector<Vec3> &generated_positions) {
    using std::pow;
  
    const Vec3 r_frac_i = globals::lattice->basis_site_atom(pos_i).position_frac;
    const Vec3 r_frac_j = globals::lattice->basis_site_atom(pos_j).position_frac;

    const Vec3 r_cart_i = globals::lattice->fractional_to_cartesian(r_frac_i);
    const Vec3 r_cart_j = globals::lattice->fractional_to_cartesian(r_frac_j);

  jams::MultiArray<double, 4> rspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2],
        6);

  jams::MultiArray<Complex, 4> kspace_tensor_hi(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2]/2 + 1,
        6);


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
                rspace_tensor(nx, ny, nz, 0) =  w0 * (3 * r_ij[0] * r_ij[0] - r_abs_sq) / r_pow_5_2;
                // xy
                rspace_tensor(nx, ny, nz, 1) =  w0 * (3 * r_ij[0] * r_ij[1]) / r_pow_5_2;
                // xz
                rspace_tensor(nx, ny, nz, 2) =  w0 * (3 * r_ij[0] * r_ij[2]) / r_pow_5_2;
                // yy
                rspace_tensor(nx, ny, nz, 3) =  w0 * (3 * r_ij[1] * r_ij[1] - r_abs_sq) / r_pow_5_2;
                // yz
                rspace_tensor(nx, ny, nz, 4) =  w0 * (3 * r_ij[1] * r_ij[2]) / r_pow_5_2;
                // zz
                rspace_tensor(nx, ny, nz, 5) =  w0 * (3 * r_ij[2] * r_ij[2] - r_abs_sq) / r_pow_5_2;
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

    int rank            = 3;
    int stride          = 6;
    int dist            = 1;
    int num_transforms  = 6;
    int * nembed        = nullptr;
    int transform_size[3]  = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]};

    fftw_plan fft_dipole_tensor_rspace_to_kspace
        = fftw_plan_many_dft_r2c(
            rank,                       // dimensionality
            transform_size,    // array of sizes of each dimension
            num_transforms,             // number of transforms
            rspace_tensor.data(),       // input: real data
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_COMPLEX_CAST(kspace_tensor_hi.data()),       // output: real dat
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);

  jams::MultiArray<ComplexLo, 4> kspace_tensor_lo(
    kspace_padded_size_[0],
    kspace_padded_size_[1],
    kspace_padded_size_[2]/2 + 1,
    6);

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

