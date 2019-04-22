#include <cassert>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <jams/helpers/maths.h>

#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"

#include "dipole_fft.h"

using std::pow;
using std::abs;
using std::min;
using namespace std;

namespace {
    const Mat3 Id = {1, 0, 0, 0, 1, 0, 0, 0, 1};
}

//---------------------------------------------------------------------

DipoleHamiltonianFFT::~DipoleHamiltonianFFT() {
    if (fft_s_rspace_to_kspace) {
        fftw_destroy_plan(fft_s_rspace_to_kspace);
        fft_s_rspace_to_kspace = nullptr;
    }

    if (fft_h_kspace_to_rspace) {
        fftw_destroy_plan(fft_h_kspace_to_rspace);
        fft_h_kspace_to_rspace = nullptr;
    }
}

//---------------------------------------------------------------------

DipoleHamiltonianFFT::DipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size)
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

    settings.lookupValue("distance_tolerance", r_distance_tolerance_);
    cout << "  distance_tolerance " << r_distance_tolerance_ << "\n";

    h_.resize(globals::num_spins, 3);

    for (auto n = 0; n < 3; ++n) {
        kspace_size_[n] = ::lattice->size(n);
    }

    kspace_padded_size_ = kspace_size_;

    for (auto n = 0; n < 3; ++n) {
        if (!::lattice->is_periodic(n)) {
            kspace_padded_size_[n] = kspace_size_[n] * 2;
        }
    }

    rspace_s_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    rspace_h_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    kspace_s_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    kspace_h_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);

    cout << "    kspace size " << kspace_size_ << "\n";
    cout << "    kspace padded size " << kspace_padded_size_ << "\n";
    cout << "    generating tensors\n";

  kspace_tensors_.resize(lattice->num_motif_atoms());

    for (auto pos_i = 0; pos_i < lattice->num_motif_atoms(); ++pos_i) {
      for (auto pos_j = 0; pos_j < lattice->num_motif_atoms(); ++pos_j) {
        kspace_tensors_[pos_i].push_back(generate_kspace_dipole_tensor(pos_i, pos_j));
      }
    }

    cout << "    planning FFTs\n";

    int rank            = 3;           
    int stride          = 3;
    int dist            = 1;
    int num_transforms  = 3;
    int transform_size[3]  = {
            static_cast<int>(kspace_padded_size_[0]),
            static_cast<int>(kspace_padded_size_[1]),
            static_cast<int>(kspace_padded_size_[2])
    };

    int * nembed = nullptr;

    fft_s_rspace_to_kspace
        = fftw_plan_many_dft_r2c(
            rank,                    // dimensionality
            transform_size,          // array of sizes of each dimension
            num_transforms,          // number of transforms
            rspace_s_.data(),        // input: real data
            nembed,                  // number of embedded dimensions 
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            kspace_s_.data(),        // output: complex data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT | FFTW_PRESERVE_INPUT);

    fft_h_kspace_to_rspace
        = fftw_plan_many_dft_c2r(
            rank,                    // dimensionality
            transform_size,          // array of sizes of each dimension
            num_transforms,          // number of transforms
            kspace_h_.data(),        // input: complex data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            rspace_h_.data(),        // output: real data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT | FFTW_PRESERVE_INPUT);
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_total_energy() {
    double e_total = 0.0;

    calculate_fields(h_);
    for (auto i = 0; i < globals::num_spins; ++i) {
        e_total += (  globals::s(i,0)*h_(i, 0)
                    + globals::s(i,1)*h_(i, 1)
                    + globals::s(i,2)*h_(i, 2) )*globals::mus(i);
    }

    return -0.5*e_total;
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0] * h[0] + s_i[1] * h[1] + s_i[2] * h[2]) * globals::mus(i);
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i) {
    return calculate_one_spin_energy(i, {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)});
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy_difference(
    const int i, const Vec3 &spin_initial, const Vec3 &spin_final)
{
    double h[3] = {0, 0, 0};

    calculate_fields(h_);
    for (auto m = 0; m < 3; ++m) {
        Vec3i pos = ::lattice->supercell_index(i);
        h[m] += rspace_h_(pos[0], pos[1], pos[2], m);
    }

    return -( (spin_final[0] * h[0] + spin_final[1] * h[1] + spin_final[2] * h[2])
          - (spin_initial[0] * h[0] + spin_initial[1] * h[1] + spin_initial[2] * h[2])) * globals::mus(i);
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_energies(jams::MultiArray<double, 1>& energies) {
    assert(energies.elements() == globals::num_spins);
    for (auto i = 0; i < globals::num_spins; ++i) {
        energies(i) = calculate_one_spin_energy(i);
    }
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_one_spin_field(const int i, double h[3]) {
    for (auto m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }

    calculate_fields(h_);
    for (auto m = 0; m < 3; ++m) {
        Vec3i pos = ::lattice->supercell_index(i);
        h[m] += rspace_h_(pos[0], pos[1], pos[2], m);
    }
}

//---------------------------------------------------------------------

//---------------------------------------------------------------------

jblib::Array<fftw_complex, 5> 
DipoleHamiltonianFFT::generate_kspace_dipole_tensor(const int pos_i, const int pos_j) {
  using std::pow;

  const Vec3 r_frac_i = lattice->motif_atom(pos_i).pos;
  const Vec3 r_frac_j = lattice->motif_atom(pos_j).pos;

  const Vec3 r_cart_i = lattice->fractional_to_cartesian(r_frac_i);
  const Vec3 r_cart_j = lattice->fractional_to_cartesian(r_frac_j);

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
  const double a3 = pow3(::lattice->parameter());
  const double w0 = fft_normalization_factor * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * a3);

  std::vector<Vec3> positions;
  positions.reserve(product(kspace_size_));

  for (auto nx = 0; nx < kspace_size_[0]; ++nx) {
    for (auto ny = 0; ny < kspace_size_[1]; ++ny) {
      for (auto nz = 0; nz < kspace_size_[2]; ++nz) {
        // self interaction on the same sublattice
        if (nx == 0 && ny == 0 && nz == 0 && pos_i == pos_j) {
          continue;
        }

        const auto r_ij = lattice->displacement(r_cart_j, lattice->generate_position(r_frac_i, {nx, ny, nz}));
        const auto r_abs_sq = abs_sq(r_ij);

        if (r_abs_sq > pow2(r_cutoff_ + r_distance_tolerance_)) {
          continue;
        }

        positions.push_back(r_ij);

        for (auto m = 0; m < 3; ++m) {
          for (auto n = 0; n < 3; ++n) {
            rspace_tensor(nx, ny, nz, m, n) = w0 * (3 * r_ij[m] * r_ij[n] - r_abs_sq * Id[m][n]) / pow5(sqrt(r_abs_sq));
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

  if (check_symmetry_ && (lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2))) {
    if (lattice->is_a_symmetry_complete_set(positions, r_distance_tolerance_) == false) {
      throw std::runtime_error("The points included in the dipole tensor do not form set of all symmetric points.\n"
              "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
              "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
    }
  }

  {
    int rank = 3;
    int stride = 9;
    int dist = 1;
    int num_transforms = 9;
    int *nembed = nullptr;
    int transform_size[3]  = {
            static_cast<int>(kspace_padded_size_[0]),
            static_cast<int>(kspace_padded_size_[1]),
            static_cast<int>(kspace_padded_size_[2])
    };

    fftw_plan fft_dipole_tensor_rspace_to_kspace
            = fftw_plan_many_dft_r2c(
                    rank,                       // dimensionality
                    transform_size,             // array of sizes of each dimension
                    num_transforms,             // number of transforms
                    rspace_tensor.data(),       // input: real data
                    nembed,                     // number of embedded dimensions
                    stride,                     // memory stride between elements of one fft dataset
                    dist,                       // memory distance between fft datasets
                    kspace_tensor.data(),       // output: real dat
                    nembed,                     // number of embedded dimensions
                    stride,                     // memory stride between elements of one fft dataset
                    dist,                       // memory distance between fft datasets
                    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);
  }

    return kspace_tensor;
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_fields(jams::MultiArray<double, 2> &fields) {
  using std::min;
  using std::pow;

  h_.zero();

  for (auto pos_i = 0; pos_i < ::lattice->num_motif_atoms(); ++pos_i) {
    kspace_h_.zero();
    for (auto pos_j = 0; pos_j < ::lattice->num_motif_atoms(); ++pos_j) {
      rspace_s_.zero();
      for (auto kx = 0; kx < kspace_size_[0]; ++kx) {
        for (auto ky = 0; ky < kspace_size_[1]; ++ky) {
          for (auto kz = 0; kz < kspace_size_[2]; ++kz) {
            const auto index = ::lattice->site_index_by_unit_cell(kx, ky, kz, pos_j);
            for (auto m = 0; m < 3; ++m) {
              rspace_s_(kx, ky, kz, m) = globals::s(index, m);
            }
          }
        }
      }

      fftw_execute(fft_s_rspace_to_kspace);

      const double mus_j = ::lattice->material(lattice->motif_atom(pos_j).material).moment;

      // perform convolution as multiplication in fourier space
      for (auto i = 0; i < kspace_padded_size_[0]; ++i) {
        for (auto j = 0; j < kspace_padded_size_[1]; ++j) {
          for (auto k = 0; k < (kspace_padded_size_[2]/2)+1; ++k) {
            for (auto m = 0; m < 3; ++m) {
              for (auto n = 0; n < 3; ++n) {
                std::complex<double> wq(
                        kspace_tensors_[pos_i][pos_j](i,j,k,m,n)[0],
                        kspace_tensors_[pos_i][pos_j](i,j,k,m,n)[1]);

                std::complex<double> sq(kspace_s_(i,j,k,n)[0], kspace_s_(i,j,k,n)[1]);

                std::complex<double> hq = wq * sq;

                kspace_h_(i,j,k,m)[0] += mus_j * hq.real();
                kspace_h_(i,j,k,m)[1] += mus_j * hq.imag();
              }
            }
          }
        }
      }
    }  // unit cell pos_j

    rspace_h_.zero();
    fftw_execute(fft_h_kspace_to_rspace);

    for (auto i = 0; i < kspace_size_[0]; ++i) {
      for (auto j = 0; j < kspace_size_[1]; ++j) {
        for (auto k = 0; k < kspace_size_[2]; ++k) {
          const auto index = lattice->site_index_by_unit_cell(i, j, k, pos_i);
          for (auto m = 0; m < 3; ++ m) {
            fields(index, m) += rspace_h_(i, j, k, m);
          }
        }
      }
    }
  }  // unit cell pos_i
}

