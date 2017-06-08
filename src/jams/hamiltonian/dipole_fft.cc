#include <cassert>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <complex>

#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/utils.h"
#include "jams/core/output.h"

#include "jams/hamiltonian/dipole_fft.h"

using std::pow;
using std::abs;
using std::min;

namespace {
    const jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );
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
: HamiltonianStrategy(settings, size),
  r_cutoff_(0),
  distance_tolerance_(1e-6),
  h_(globals::num_spins, 3),
  fftw_h_(globals::num_spins, 3),
  kspace_size_(0, 0, 0),
  kspace_padded_size_(0, 0, 0),
  kspace_s_(),
  kspace_h_(),
  kspace_tensors_(),
  fft_s_rspace_to_kspace(nullptr),
  fft_h_kspace_to_rspace(nullptr)
{
    r_cutoff_ = double(settings["r_cutoff"]);
    output->write("  r_cutoff: %e\n", r_cutoff_);

    if (r_cutoff_ > ::lattice->maximum_interaction_radius()) {
        throw std::runtime_error("DipoleHamiltonianFFT r_cutoff is too large for the lattice size."
        "The cutoff must be less than half of the shortest supercell lattice vector.");
    }

    settings.lookupValue("distance_tolerance", distance_tolerance_);
    output->write("  distance_tolerance: %e\n", distance_tolerance_);

    for (int n = 0; n < 3; ++n) {
        kspace_size_[n] = ::lattice->num_unit_cells(n);
    }

    kspace_padded_size_ = kspace_size_;

    for (int n = 0; n < 3; ++n) {
        if (!::lattice->is_periodic(n)) {
            kspace_padded_size_[n] = kspace_size_[n] * 2;
        }
    }

    unsigned int kspace_size = kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2]/2 + 1) * lattice->num_unit_cell_positions() * 3;

    std::cerr << kspace_size_ << "\t" << kspace_padded_size_ << std::endl;

    kspace_s_.resize(kspace_size);
    kspace_h_.resize(kspace_size);

    kspace_s_.zero();
    kspace_h_.zero();
    h_.zero();
    fftw_h_.zero();

    output->write("    kspace size: %d %d %d\n", kspace_size_[0], kspace_size_[1], kspace_size_[2]);
    output->write("    kspace padded size: %d %d %d\n", kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]);

    output->write("    planning FFTs\n");

    int rank            = 3;
    int stride          = 3 * lattice->num_unit_cell_positions();
    int dist            = 1;
    int num_transforms  = 3 * lattice->num_unit_cell_positions();
    int rspace_embed[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};
    int kspace_embed[3] = {kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]/2 + 1};
    int fft_size[3] = {kspace_size_[0], kspace_size_[1], kspace_size_[2]};

    fft_s_rspace_to_kspace
        = fftw_plan_many_dft_r2c(
            rank,                    // dimensionality
            fft_size,                // array of sizes of each dimension
            num_transforms,          // number of transforms
            globals::s.data(),        // input: real data
            rspace_embed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            kspace_s_.data(),        // output: complex data
            kspace_embed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT|FFTW_PRESERVE_INPUT);

    fft_h_kspace_to_rspace
        = fftw_plan_many_dft_c2r(
            rank,                    // dimensionality
            fft_size, // array of sizes of each dimension
            num_transforms,          // number of transforms
            kspace_h_.data(),        // input: complex data
            kspace_embed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            fftw_h_.data(),        // output: real data
            rspace_embed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT|FFTW_PRESERVE_INPUT);

    kspace_tensors_.resize(lattice->num_unit_cell_positions());

    for (int pos_i = 0; pos_i < lattice->num_unit_cell_positions(); ++pos_i) {
        for (int pos_j = 0; pos_j < lattice->num_unit_cell_positions(); ++pos_j) {
            kspace_tensors_[pos_i].push_back(generate_kspace_dipole_tensor(pos_i, pos_j));
        }
    }
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_total_energy() {
    double e_total = 0.0;

    calculate_fields(h_);
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += (  globals::s(i,0)*h_(i, 0)
                    + globals::s(i,1)*h_(i, 1)
                    + globals::s(i,2)*h_(i, 2) )*globals::mus(i);
    }

    return -0.5*e_total;
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0] * h[0] + s_i[1] * h[1] + s_i[2] * h[2]) * globals::mus(i);
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy_difference(
    const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final)
{
    jblib::Vec3<int> pos;

    double h[3] = {0, 0, 0};

    calculate_fields(h_);

    return -( (spin_final[0] * h_(i,0) + spin_final[1] * h_(i,1) + spin_final[2] * h_(i,2))
          - (spin_initial[0] * h_(i,0) + spin_initial[1] * h_(i,1) + spin_initial[2] * h_(i,2))) * globals::mus(i);
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.elements() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_one_spin_field(const int i, double h[3]) {
    jblib::Vec3<int> pos;

    for (int m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }

    calculate_fields(h_);
    for (int m = 0; m < 3; ++m) {
        h[m] += h_(i,m);
    }
}

//---------------------------------------------------------------------

//---------------------------------------------------------------------

jblib::Array<fftw_complex, 5> 
DipoleHamiltonianFFT::generate_kspace_dipole_tensor(const int pos_i, const int pos_j) {
    using std::pow;

    const Vec3 r_frac_i = lattice->unit_cell_position(pos_i);
    const Vec3 r_frac_j = lattice->unit_cell_position(pos_j);

    const Vec3 r_cart_i = lattice->unit_cell_position_cart(pos_i);
    const Vec3 r_cart_j = lattice->unit_cell_position_cart(pos_j);

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

    const double fft_normalization_factor = 1.0 / product(kspace_size_);
    const double v = pow(lattice->parameter(), 3);
    double w0 = fft_normalization_factor * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * v);

    std::vector<Vec3> positions;

    for (int nx = 0; nx < kspace_size_[0]; ++nx) {
        for (int ny = 0; ny < kspace_size_[1]; ++ny) {
            for (int nz = 0; nz < kspace_size_[2]; ++nz) {

                if (nx == 0 && ny == 0 && nz == 0 && pos_i == pos_j) {
                    // self interaction on the same sublattice
                    continue;
                } 

                const Vec3 r_ij = 
                    lattice->minimum_image(r_cart_j,
                        lattice->generate_position(r_frac_i, {nx, ny, nz})); // generate_position requires FRACTIONAL coordinate

                const auto r_abs_sq = r_ij.norm_sq();

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

    if(lattice->is_a_symmetry_complete_set(positions, distance_tolerance_) == false) {
      throw std::runtime_error("The points included in the dipole tensor do not form set of all symmetric points.\n"
                               "This can happen if the r_cutoff just misses a point because of floating point arithmetic"
                               "Check that the lattice vectors are specified to enough precision or increase r_cutoff by a very small amount.");
    }

    int rank            = 3;
    int stride          = 9;
    int dist            = 1;
    int num_transforms  = 9;
    int * nembed        = NULL;
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

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_fields(jblib::Array<double, 2> &fields) {
    // The copy-swap idiom in the array classes makes it dangerous to use the pointer from h_ for FFTW when passing h_
    // as the fields argument. Therefore fftw_h_ is the only one which should be used.
    using std::min;
    using std::pow;

//    kspace_s_.zero();
    fftw_execute(fft_s_rspace_to_kspace);

    unsigned num_pos = lattice->num_unit_cell_positions();
    unsigned int fft_size =
            kspace_padded_size_[0] * kspace_padded_size_[1] * (kspace_padded_size_[2] / 2 + 1);
    kspace_h_.zero();

    for (unsigned i = 0; i < fft_size; ++i) {

        for (int pos_i = 0; pos_i < num_pos; ++pos_i) {
            for (int pos_j = 0; pos_j < num_pos; ++pos_j) {

                const double mus_j = lattice->unit_cell_material(pos_j).moment;
                
                jblib::Vec3<std::complex<double>> sq = {
                        {kspace_s_[3 * (num_pos * i + pos_j) + 0][0], kspace_s_[3 * (num_pos * i + pos_j) + 0][1]},
                        {kspace_s_[3 * (num_pos * i + pos_j) + 1][0], kspace_s_[3 * (num_pos * i + pos_j) + 1][1]},
                        {kspace_s_[3 * (num_pos * i + pos_j) + 2][0], kspace_s_[3 * (num_pos * i + pos_j) + 2][1]}};

                jblib::Matrix<std::complex<double>, 3, 3> wq(
                        {kspace_tensors_[pos_i][pos_j][9 * i + 0][0], kspace_tensors_[pos_i][pos_j][9 * i + 0][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 1][0], kspace_tensors_[pos_i][pos_j][9 * i + 1][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 2][0], kspace_tensors_[pos_i][pos_j][9 * i + 2][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 3][0], kspace_tensors_[pos_i][pos_j][9 * i + 3][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 4][0], kspace_tensors_[pos_i][pos_j][9 * i + 4][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 5][0], kspace_tensors_[pos_i][pos_j][9 * i + 5][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 6][0], kspace_tensors_[pos_i][pos_j][9 * i + 6][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 7][0], kspace_tensors_[pos_i][pos_j][9 * i + 7][1]},
                        {kspace_tensors_[pos_i][pos_j][9 * i + 8][0], kspace_tensors_[pos_i][pos_j][9 * i + 8][1]}
                );

                jblib::Vec3<std::complex<double>> hq = wq * sq;

                for (int n = 0; n < 3; ++n) {
                    kspace_h_[3 * (num_pos * i + pos_i) + n][0] += mus_j * hq[n].real();
                    kspace_h_[3 * (num_pos * i + pos_i) + n][1] += mus_j * hq[n].imag();
                }
            }
        }
    }

//    h_.zero();
    fftw_execute(fft_h_kspace_to_rspace);

    fields = fftw_h_;
//    for (int i = 0; i < h_.elements(); ++i) {
//        fields[i] = fftw_h_[i];
//    }
}

