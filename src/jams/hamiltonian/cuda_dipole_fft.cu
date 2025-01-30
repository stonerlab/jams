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

__global__ void cuda_dipole_convolution(
  const unsigned int size,
  const unsigned int pos_i, 
  const unsigned int pos_j,
  const unsigned int num_pos,
  const double alpha,
  const cufftDoubleComplex* gpu_sq,
  const cufftDoubleComplex* gpu_wq,
  cufftDoubleComplex* gpu_hq
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {

    const cuDoubleComplex sq[3] = 
      { gpu_sq[3 * (num_pos * idx + pos_j) + 0],
        gpu_sq[3 * (num_pos * idx + pos_j) + 1],
        gpu_sq[3 * (num_pos * idx + pos_j) + 2]
      };

      gpu_hq[3 * (num_pos * idx + pos_i) + 0] +=  alpha * (gpu_wq[6 * idx + 0] * sq[0] + gpu_wq[6 * idx + 1] * sq[1] + gpu_wq[6 * idx + 2] * sq[2]);
      gpu_hq[3 * (num_pos * idx + pos_i) + 1] +=  alpha * (gpu_wq[6 * idx + 1] * sq[0] + gpu_wq[6 * idx + 3] * sq[1] + gpu_wq[6 * idx + 4] * sq[2]);
      gpu_hq[3 * (num_pos * idx + pos_i) + 2] +=  alpha * (gpu_wq[6 * idx + 2] * sq[0] + gpu_wq[6 * idx + 4] * sq[1] + gpu_wq[6 * idx + 5] * sq[2]);
  }

}


namespace {
    const Mat3 Id = {1, 0, 0, 0, 1, 0, 0, 0, 1};
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

  CHECK_CUFFT_STATUS(cufftPlanMany(&cuda_fft_s_rspace_to_kspace, rank, fft_size, rspace_embed, stride, dist,
          kspace_embed, stride, dist, CUFFT_D2Z, num_transforms));

  CHECK_CUFFT_STATUS(cufftPlanMany(&cuda_fft_h_kspace_to_rspace, rank, fft_size, kspace_embed, stride, dist,
          rspace_embed, stride, dist, CUFFT_Z2D, num_transforms));

  kspace_tensors_.resize(globals::lattice->num_basis_sites());
  for (int pos_i = 0; pos_i < globals::lattice->num_basis_sites(); ++pos_i) {
    std::vector<Vec3> generated_positions;
    for (int pos_j = 0; pos_j < globals::lattice->num_basis_sites(); ++pos_j) {
        auto wq = generate_kspace_dipole_tensor(pos_i, pos_j, generated_positions);

        jams::MultiArray<cufftDoubleComplex, 1> gpu_wq(wq.elements());
        kspace_tensors_[pos_i].push_back(gpu_wq);

        CHECK_CUDA_STATUS(cudaMemcpy(kspace_tensors_[pos_i].back().device_data(), wq.data(), wq.elements() * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    }
      if (check_symmetry_ && (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2))) {
        if (!globals::lattice->is_a_symmetry_complete_set(pos_i, generated_positions, distance_tolerance_)) {
          throw std::runtime_error("The points included in the dipole tensor do not form set of all symmetric points.\n"
                                   "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
                                   "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
        }
      }
  }

  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_s_rspace_to_kspace, dev_stream_[0].get()));
  CHECK_CUFFT_STATUS(cufftSetStream(cuda_fft_h_kspace_to_rspace, dev_stream_[0].get()));
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
  cuda_array_elementwise_scale(globals::num_spins, 3, globals::mus.device_data(), 1.0, field_.device_data(), 1, field_.device_data(), 1, dev_stream_[0].get());
}

Vec3 CudaDipoleFFTHamiltonian::calculate_field(const int i, double time) {
  throw jams::unimplemented_error("CudaDipoleFFTHamiltonian::calculate_field");
}

void CudaDipoleFFTHamiltonian::calculate_fields(double time) {

  kspace_h_.zero();

  CHECK_CUFFT_STATUS(cufftExecD2Z(cuda_fft_s_rspace_to_kspace, reinterpret_cast<cufftDoubleReal*>(globals::s.device_data()), kspace_s_.device_data()));
  cudaStreamSynchronize(dev_stream_[0].get());

  for (int pos_j = 0; pos_j < globals::lattice->num_basis_sites(); ++pos_j) {
    const double mus_j = globals::lattice->material(
        globals::lattice->basis_site_atom(pos_j).material_index).moment;

    for (int pos_i = 0; pos_i < globals::lattice->num_basis_sites(); ++pos_i) {

      const unsigned int fft_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);

      dim3 block_size = {32, 1, 1};
      dim3 grid_size = cuda_grid_size(block_size, {fft_size, 1, 1});

      cuda_dipole_convolution<<<grid_size, block_size, 0, dev_stream_[pos_i%4].get()>>>(fft_size, pos_i, pos_j, globals::lattice->num_basis_sites(), mus_j, kspace_s_.device_data(), kspace_tensors_[pos_i][pos_j].device_data(), kspace_h_.device_data());
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
    }
    cudaDeviceSynchronize();
  }

  CHECK_CUFFT_STATUS(cufftExecZ2D(cuda_fft_h_kspace_to_rspace, kspace_h_.device_data(), reinterpret_cast<cufftDoubleReal*>(field_.device_data())));

  cuda_array_elementwise_scale(globals::num_spins, 3, globals::mus.device_data(), 1.0, field_.device_data(), 1, field_.device_data(), 1, dev_stream_[0].get());

}

// Generates the dipole tensor between unit cell positions i and j and appends
// the generated positions to a vector
jams::MultiArray<Complex, 4>
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

  jams::MultiArray<Complex, 4> kspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2]/2 + 1,
        6);


    rspace_tensor.zero();
    kspace_tensor.zero();

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

                generated_positions.push_back(r_ij);

                // xx
                rspace_tensor(nx, ny, nz, 0) =  w0 * (3 * r_ij[0] * r_ij[0] - r_abs_sq) / pow(sqrt(r_abs_sq), 5);
                // xy
                rspace_tensor(nx, ny, nz, 1) =  w0 * (3 * r_ij[0] * r_ij[1]) / pow(sqrt(r_abs_sq), 5);
                // xz
                rspace_tensor(nx, ny, nz, 2) =  w0 * (3 * r_ij[0] * r_ij[2]) / pow(sqrt(r_abs_sq), 5);
                // yy
                rspace_tensor(nx, ny, nz, 3) =  w0 * (3 * r_ij[1] * r_ij[1] - r_abs_sq) / pow(sqrt(r_abs_sq), 5);
                // yz
                rspace_tensor(nx, ny, nz, 4) =  w0 * (3 * r_ij[1] * r_ij[2]) / pow(sqrt(r_abs_sq), 5);
                // zz
                rspace_tensor(nx, ny, nz, 5) =  w0 * (3 * r_ij[2] * r_ij[2] - r_abs_sq) / pow(sqrt(r_abs_sq), 5);

//                for (int m = 0; m < 3; ++m) {
//                    for (int n = m; n < 3; ++n) {
//                        auto value = w0 * (3 * r_ij[m] * r_ij[n] - r_abs_sq * Id[m][n]) / pow(sqrt(r_abs_sq), 5);
//
//                        std::cout << m << " " << n << " " << value << std::endl;
//                        if (!std::isfinite(value)) {
//                          throw std::runtime_error("fatal error in CudaDipoleFFTHamiltonian::generate_kspace_dipole_tensor: tensor Szz is not finite");
//                        }
//                        rspace_tensor(nx, ny, nz, m, n) = value;
//                    }
//                }
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
            FFTW_COMPLEX_CAST(kspace_tensor.data()),       // output: real dat
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);

    return kspace_tensor;
}

