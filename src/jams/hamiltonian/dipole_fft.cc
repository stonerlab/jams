#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>

#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/output.h"
#include "jams/interface/fft.h"

#include "jams/hamiltonian/dipole_fft.h"

using std::pow;
using std::abs;
using std::min;
using namespace std;

namespace {
    const Mat3 Id = {1, 0, 0, 0, 1, 0, 0, 0, 1};
}


DipoleFFTHamiltonian::~DipoleFFTHamiltonian() {
    if (fft_s_rspace_to_kspace) {
        fftw_destroy_plan(fft_s_rspace_to_kspace);
        fft_s_rspace_to_kspace = nullptr;
    }

    if (fft_h_kspace_to_rspace) {
        fftw_destroy_plan(fft_h_kspace_to_rspace);
        fft_h_kspace_to_rspace = nullptr;
    }
}


DipoleFFTHamiltonian::DipoleFFTHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size)
{
  settings.lookupValue("debug", debug_);
  settings.lookupValue("check_radius", check_radius_);
  settings.lookupValue("check_symmetry", check_symmetry_);

    r_cutoff_ = double(settings["r_cutoff"]);
    cout << "  r_cutoff " << r_cutoff_ << "\n";
    cout << "  r_cutoff_max " << ::lattice->max_interaction_radius() << "\n";

    if (check_radius_) {
      if (r_cutoff_ > ::lattice->max_interaction_radius()) {
        throw std::runtime_error("DipoleFFTHamiltonian r_cutoff is too large for the lattice size."
                                         "The cutoff must be less than the inradius of the lattice.");
      }
    }

    settings.lookupValue("distance_tolerance", r_distance_tolerance_);
    cout << "  distance_tolerance " << r_distance_tolerance_ << "\n";

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
      std::vector<Vec3> generated_positions;
      for (auto pos_j = 0; pos_j < lattice->num_motif_atoms(); ++pos_j) {
        kspace_tensors_[pos_i].push_back(generate_kspace_dipole_tensor(pos_i, pos_j, generated_positions));
      }
      if (check_symmetry_ && (lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2))) {
        if (!lattice->is_a_symmetry_complete_set(generated_positions, r_distance_tolerance_)) {
          throw std::runtime_error("The points included in the dipole tensor do not form set of all symmetric points.\n"
                                   "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
                                   "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
        }
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
            FFTW_COMPLEX_CAST(kspace_s_.data()),        // output: complex data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT);
    assert(fft_s_rspace_to_kspace != NULL);

    fft_h_kspace_to_rspace
        = fftw_plan_many_dft_c2r(
            rank,                    // dimensionality
            transform_size,          // array of sizes of each dimension
            num_transforms,          // number of transforms
            FFTW_COMPLEX_CAST(kspace_h_.data()),        // input: complex data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            rspace_h_.data(),        // output: real data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT);
    assert(fft_h_kspace_to_rspace != NULL);

}


double DipoleFFTHamiltonian::calculate_total_energy() {
    double e_total = 0.0;

    calculate_fields();
    for (auto i = 0; i < globals::num_spins; ++i) {
        e_total += (  globals::s(i,0) * field_(i, 0)
                    + globals::s(i,1) * field_(i, 1)
                    + globals::s(i,2) * field_(i, 2) )*globals::mus(i);
    }

    return -0.5*e_total;
}

double DipoleFFTHamiltonian::calculate_energy(const int i) {
  const Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
  const auto field = calculate_field(i);
  return -globals::mus(i) * dot(s_i, field);
}


double DipoleFFTHamiltonian::calculate_energy_difference(
    int i, const Vec3 &spin_initial, const Vec3 &spin_final)
{
    double h[3] = {0, 0, 0};

    calculate_fields();
    for (auto m = 0; m < 3; ++m) {
        Vec3i pos = ::lattice->cell_offset(i);
        h[m] += rspace_h_(pos[0], pos[1], pos[2], m);
    }

    return -( (spin_final[0] * h[0] + spin_final[1] * h[1] + spin_final[2] * h[2])
          - (spin_initial[0] * h[0] + spin_initial[1] * h[1] + spin_initial[2] * h[2])) * globals::mus(i);
}


void DipoleFFTHamiltonian::calculate_energies() {
    assert(energy_.elements() == globals::num_spins);
    for (auto i = 0; i < globals::num_spins; ++i) {
      energy_(i) = calculate_energy(i);
    }
}


Vec3 DipoleFFTHamiltonian::calculate_field(const int i) {
    Vec3 field = {0.0, 0.0, 0.0};
    calculate_fields();
    for (auto m = 0; m < 3; ++m) {
        Vec3i pos = ::lattice->cell_offset(i);
        field[m] += rspace_h_(pos[0], pos[1], pos[2], m);
    }
    return field;
}


// Generates the dipole tensor between unit cell positions i and j and appends
// the generated positions to a vector
jams::MultiArray<Complex, 5>
DipoleFFTHamiltonian::generate_kspace_dipole_tensor(const int pos_i, const int pos_j, std::vector<Vec3> &generated_positions) {
  using std::pow;

  const Vec3 r_frac_i = lattice->motif_atom(pos_i).position;
  const Vec3 r_frac_j = lattice->motif_atom(pos_j).position;

  const Vec3 r_cart_i = lattice->fractional_to_cartesian(r_frac_i);
  const Vec3 r_cart_j = lattice->fractional_to_cartesian(r_frac_j);

  jams::MultiArray<double, 5> rspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2],
        3, 3);

  jams::MultiArray<Complex, 5> kspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2]/2 + 1,
        3, 3);


  rspace_tensor.zero();
  kspace_tensor.zero();

  const double fft_normalization_factor = 1.0 / product(kspace_padded_size_);
  const double a3 = pow3(::lattice->parameter());
  const double w0 = fft_normalization_factor * kVacuumPermeabilityIU / (4.0 * kPi * a3);

  for (auto nx = 0; nx < kspace_size_[0]; ++nx) {
    for (auto ny = 0; ny < kspace_size_[1]; ++ny) {
      for (auto nz = 0; nz < kspace_size_[2]; ++nz) {
        // self interaction on the same sublattice
        if (nx == 0 && ny == 0 && nz == 0 && pos_i == pos_j) {
          continue;
        }

        const auto r_ij = lattice->displacement(r_cart_j,
                                                lattice->generate_cartesian_lattice_position_from_fractional(r_frac_i,
                                                                                                             {nx, ny,
                                                                                                              nz}));
        const auto r_abs_sq = norm_sq(r_ij);

        if (r_abs_sq > pow2(r_cutoff_ + r_distance_tolerance_)) {
          continue;
        }

        generated_positions.push_back(r_ij);

        for (auto m = 0; m < 3; ++m) {
          for (auto n = 0; n < 3; ++n) {
            rspace_tensor(nx, ny, nz, m, n) = w0 * (3 * r_ij[m] * r_ij[n] - r_abs_sq * Id[m][n]) / pow5(sqrt(r_abs_sq));
          }
        }
      }
    }
  }

  if (debug_) {
    std::ofstream debugfile(jams::output::full_path_filename("DEBUG_dipole_fft_" + std::to_string(pos_i) + "_" + std::to_string(pos_j) + "_rij.tsv"));

    for (const auto& r : generated_positions) {
      debugfile << r << "\n";
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
                    FFTW_COMPLEX_CAST(kspace_tensor.data()),       // output: real dat
                    nembed,                     // number of embedded dimensions
                    stride,                     // memory stride between elements of one fft dataset
                    dist,                       // memory distance between fft datasets
                    FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);
  }

    return kspace_tensor;
}


void DipoleFFTHamiltonian::calculate_fields() {
  using std::min;
  using std::pow;

  zero(field_);

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

      const double mus_j = ::lattice->material(lattice->motif_atom(pos_j).material_index).moment;

      // perform convolution as multiplication in fourier space
      for (auto i = 0; i < kspace_padded_size_[0]; ++i) {
        for (auto j = 0; j < kspace_padded_size_[1]; ++j) {
          for (auto k = 0; k < (kspace_padded_size_[2]/2)+1; ++k) {
            for (auto m = 0; m < 3; ++m) {
              for (auto n = 0; n < 3; ++n) {
                kspace_h_(i,j,k,m) += mus_j * kspace_tensors_[pos_i][pos_j](i,j,k,m,n) * kspace_s_(i,j,k,n);
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
            field_(index, m) += rspace_h_(i, j, k, m);
          }
        }
      }
    }
  }  // unit cell pos_i
}

