
#include <cufft.h>
#include <fstream>
#include <libconfig.h++>

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"

#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"

#include "cuda_dipole_fft.h"
#include "../cuda/cuda-complex-operators.h"

using namespace std;

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

      gpu_hq[3 * (num_pos * idx + pos_i) + 0] +=  alpha * (gpu_wq[9 * idx + 0] * sq[0] + gpu_wq[9 * idx + 1] * sq[1] + gpu_wq[9 * idx + 2] * sq[2]);
      gpu_hq[3 * (num_pos * idx + pos_i) + 1] +=  alpha * (gpu_wq[9 * idx + 3] * sq[0] + gpu_wq[9 * idx + 4] * sq[1] + gpu_wq[9 * idx + 5] * sq[2]);
      gpu_hq[3 * (num_pos * idx + pos_i) + 2] +=  alpha * (gpu_wq[9 * idx + 6] * sq[0] + gpu_wq[9 * idx + 7] * sq[1] + gpu_wq[9 * idx + 8] * sq[2]);
  }

}


namespace {
    const Mat3 Id = {1, 0, 0, 0, 1, 0, 0, 0, 1};
}

CudaDipoleHamiltonianFFT::~CudaDipoleHamiltonianFFT() {
  if (cuda_fft_s_rspace_to_kspace) {
      cufftDestroy(cuda_fft_s_rspace_to_kspace);
  }

  if (cuda_fft_h_kspace_to_rspace) {
      cufftDestroy(cuda_fft_h_kspace_to_rspace);
  }
}

CudaDipoleHamiltonianFFT::CudaDipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size),
  dev_stream_(),
  r_cutoff_(0),
  distance_tolerance_(1e-6),
  h_(3*globals::num_spins),
  kspace_size_(0, 0, 0),
  kspace_padded_size_(0, 0, 0),
  kspace_s_(),
  kspace_h_(),
  cuda_fft_s_rspace_to_kspace(),
  cuda_fft_h_kspace_to_rspace()
{
  settings.lookupValue("debug", debug_);
  settings.lookupValue("check_radius", check_radius_);
  settings.lookupValue("check_symmetry", check_symmetry_);

  r_cutoff_ = double(settings["r_cutoff"]);
  cout << "  r_cutoff " << r_cutoff_ << "\n";
  cout << "  r_cutoff_max " << ::lattice->max_interaction_radius() << "\n";

  if (check_radius_) {
    if (r_cutoff_ > ::lattice->max_interaction_radius()) {
      throw std::runtime_error("DipoleHamiltonianFFT r_cutoff is too large for the lattice size."
                                       "The cutoff must be less than the inradius of the lattice.");
    }
  }

  settings.lookupValue("distance_tolerance", distance_tolerance_);
  cout << "  distance_tolerance " << distance_tolerance_ << "\n";

  for (int n = 0; n < 3; ++n) {
      kspace_size_[n] = ::lattice->size(n);
  }

  kspace_padded_size_ = kspace_size_;

  for (int n = 0; n < 3; ++n) {
      if (!::lattice->is_periodic(n)) {
          kspace_padded_size_[n] = kspace_size_[n] * 2;
      }
  }

  unsigned int kspace_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2]/2 + 1) *
          lattice->num_motif_positions() * 3;

  kspace_s_.resize(kspace_size);
  kspace_h_.resize(kspace_size);

  kspace_s_.zero();
  kspace_h_.zero();
  h_.zero();

  cout << "    kspace size " << kspace_size_ << "\n";
  cout << "    kspace padded size " << kspace_padded_size_ << "\n";

  int rank            = 3;           
  int stride          = 3 * lattice->num_motif_positions();
  int dist            = 1;
  int num_transforms  = 3 * lattice->num_motif_positions();
  int rspace_embed[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
  int kspace_embed[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]/2 + 1};

  int fft_size[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
  int fft_padded_size[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]};

  cufftResult ret = cufftPlanMany(&cuda_fft_s_rspace_to_kspace, rank, fft_size, rspace_embed, stride, dist, 
          kspace_embed, stride, dist, CUFFT_D2Z, num_transforms);

  if (ret != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT failure");
  }

  ret = cufftPlanMany(&cuda_fft_h_kspace_to_rspace, rank, fft_size, kspace_embed, stride, dist, 
          rspace_embed, stride, dist, CUFFT_Z2D, num_transforms);

  if (ret != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT failure");
  }

  kspace_tensors_.resize(lattice->num_motif_positions());
  for (int pos_i = 0; pos_i < lattice->num_motif_positions(); ++pos_i) {
      for (int pos_j = 0; pos_j < lattice->num_motif_positions(); ++pos_j) {
        auto wq = generate_kspace_dipole_tensor(pos_i, pos_j);

        jblib::CudaArray<cufftDoubleComplex, 1> gpu_wq(wq.elements());
        kspace_tensors_[pos_i].push_back(gpu_wq);
          cudaMemcpy(kspace_tensors_[pos_i].back().data(), wq.data(), wq.elements() * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
          
      }
  }

  cufftSetStream(cuda_fft_s_rspace_to_kspace, dev_stream_[0].get());
  cufftSetStream(cuda_fft_h_kspace_to_rspace, dev_stream_[0].get());


}

double CudaDipoleHamiltonianFFT::calculate_total_energy() {
  calculate_fields(h_);

  jblib::Array<double, 2> hd(globals::num_spins, 3);

  h_.copy_to_host_array(hd);

  double e_total = 0.0;
  for (int i = 0; i < globals::num_spins; ++i) {
      e_total += (  globals::s(i,0)*hd(i, 0)
                  + globals::s(i,1)*hd(i, 1)
                  + globals::s(i,2)*hd(i, 2) ) * globals::mus(i);
  }

  return -0.5*e_total;
}

double CudaDipoleHamiltonianFFT::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    return 0.0;
}

double CudaDipoleHamiltonianFFT::calculate_one_spin_energy(const int i) {
    return 0.0;
}

double CudaDipoleHamiltonianFFT::calculate_one_spin_energy_difference(
    const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {

    return 0.0;
}

void CudaDipoleHamiltonianFFT::calculate_energies(jblib::Array<double, 1>& energies) {

}

void CudaDipoleHamiltonianFFT::calculate_one_spin_field(const int i, double h[3]) {

}

void CudaDipoleHamiltonianFFT::calculate_fields(jblib::Array<double, 2>& fields) {

}

void CudaDipoleHamiltonianFFT::calculate_fields(jblib::CudaArray<double, 1>& gpu_h) {
  cufftResult result;

  kspace_h_.zero(dev_stream_[0].get());

  // TODO: change these to macros to avoid blocking in production code
  result = cufftExecD2Z(cuda_fft_s_rspace_to_kspace, reinterpret_cast<cufftDoubleReal*>(solver->dev_ptr_spin()), kspace_s_.data());
  if (result != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT failure");
  }

  cudaDeviceSynchronize();

  for (int pos_j = 0; pos_j < lattice->num_motif_positions(); ++pos_j) {
    for (int pos_i = 0; pos_i < lattice->num_motif_positions(); ++pos_i) {
      const double mus_j = lattice->motif_material(pos_j).moment;

      const unsigned int fft_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);

      dim3 block_size = {128, 1, 1};
      dim3 grid_size = cuda_grid_size(block_size, {fft_size, 1, 1});

      cuda_dipole_convolution<<<grid_size, block_size, 0, dev_stream_[pos_i%4].get()>>>(fft_size, pos_i, pos_j, lattice->num_motif_positions(), mus_j, kspace_s_.data(),  kspace_tensors_[pos_i][pos_j].data(), kspace_h_.data());
    }
    cudaDeviceSynchronize();
  }

  result = cufftExecZ2D(cuda_fft_h_kspace_to_rspace, kspace_h_.data(), reinterpret_cast<cufftDoubleReal*>(gpu_h.data()));
  
  if (result != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT failure");
  }




}

jblib::Array<fftw_complex, 5> 
CudaDipoleHamiltonianFFT::generate_kspace_dipole_tensor(const int pos_i, const int pos_j) {
    using std::pow;

    const Vec3 r_frac_i = lattice->motif_position_frac(pos_i);
    const Vec3 r_frac_j = lattice->motif_position_frac(pos_j);

    const Vec3 r_cart_i = lattice->motif_position_cart(pos_i);
    const Vec3 r_cart_j = lattice->motif_position_cart(pos_j);

    jblib::Array<double, 5> rspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2],
        3, 3);

    jblib::Array<fftw_complex, 5> kspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2]/2 + 1,
        3, 3);


    rspace_tensor.zero();
    kspace_tensor.zero();

    const double fft_normalization_factor = 1.0 / product(kspace_padded_size_);
    const double v = pow(lattice->parameter(), 3);
    const double w0 = fft_normalization_factor * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * v);

    std::vector<Vec3> positions;

    for (int nx = 0; nx < kspace_size_[0]; ++nx) {
        for (int ny = 0; ny < kspace_size_[1]; ++ny) {
            for (int nz = 0; nz < kspace_size_[2]; ++nz) {

                if (nx == 0 && ny == 0 && nz == 0 && pos_i == pos_j) {
                    // self interaction on the same sublattice
                    continue;
                } 

                const Vec3 r_ij = 
                    lattice->displacement(r_cart_j,
                        lattice->generate_position(r_frac_i, {nx, ny, nz})); // generate_position requires FRACTIONAL coordinate

                const auto r_abs_sq = abs_sq(r_ij);

                if (r_abs_sq > pow(r_cutoff_ + distance_tolerance_, 2)) {
                    // outside of cutoff radius
                    continue;
                }

                positions.push_back(r_ij);

                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        rspace_tensor(nx, ny, nz, m, n)
                            = w0 * (3 * r_ij[m] * r_ij[n] - r_abs_sq * Id[m][n]) / pow(sqrt(r_abs_sq), 5);
                    }
                }
            }
        }
    }
  
    if (debug_) {
      std::string filename = "debug_dipole_fft_" + std::to_string(pos_i) + "_" + std::to_string(pos_j) + "_rij.tsv";
      std::ofstream debugfile(filename);

      for (const auto& r : positions) {
        debugfile << r << "\n";
      }
    }

    if (check_symmetry_) {
      if (lattice->is_a_symmetry_complete_set(positions, distance_tolerance_) == false) {
        throw std::runtime_error("The points included in the dipole tensor do not form set of all symmetric points.\n"
                                         "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
                                         "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
      }
    }
    int rank            = 3;
    int stride          = 9;
    int dist            = 1;
    int num_transforms  = 9;
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
            kspace_tensor.data(),       // output: real dat
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);

    return kspace_tensor;
}

