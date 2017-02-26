#include "core/globals.h"
#include "core/utils.h"
#include "core/maths.h"
#include "core/consts.h"
#include "core/cuda_defs.h"

#include "hamiltonian/anisotropy-cubic.h"
#include "hamiltonian/anisotropy-cubic-kernel.h"

AnisotropyCubicHamiltonian::AnisotropyCubicHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings),
  K1_value_(),
  K2_value_()
{
    ::output.write("initialising cubic anisotropy Hamiltonian\n");
    // output in default format for now
    outformat_ = TEXT;

    // resize member arrays
    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);
    field_.zero();

    K1_value_.resize(globals::num_spins);
    K2_value_.resize(globals::num_spins);

    K1_value_.zero();
    K2_value_.zero();

    if (settings["K1"].getLength() != lattice.num_materials()) {
        jams_error("AnisotropyCubicHamiltonian: K1 must be specified for every material");
    }
    for (int i = 0; i < globals::num_spins; ++i) {
        K1_value_(i) = double(settings["K1"][lattice.atom_material(i)])/kBohrMagneton;
    }

    if (settings["K2"].getLength() != lattice.num_materials()) {
        jams_error("AnisotropyCubicHamiltonian: K2 must be specified for every material");
    }
    for (int i = 0; i < globals::num_spins; ++i) {
        K2_value_(i) = double(settings["K2"][lattice.atom_material(i)])/kBohrMagneton;
    }

    // transfer arrays to cuda device if needed
#ifdef CUDA
    if (solver->is_cuda_solver()) {
        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_ = jblib::CudaArray<double, 1>(field_);
        dev_K1_value_ = jblib::CudaArray<double, 1>(K1_value_);
        dev_K2_value_ = jblib::CudaArray<double, 1>(K2_value_);
    }

    cudaStreamCreate(&dev_stream_);

    dev_blocksize_ = 128;
#endif

}

// --------------------------------------------------------------------------

double AnisotropyCubicHamiltonian::calculate_total_energy() {
    double e_total = 0.0;

    calculate_energies();
    if (solver->is_cuda_solver()) {
#ifdef CUDA
        dev_energy_.copy_to_host_array(energy_);
#endif // CUDA
    }

    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += energy_[i];
    }

    return e_total;
}

// --------------------------------------------------------------------------

double AnisotropyCubicHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;

    return cubic_K1_K2_anisotropy_energy(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
}

// --------------------------------------------------------------------------

double AnisotropyCubicHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    using std::pow;

    const double e_initial = cubic_K1_K2_anisotropy_energy(K1_value_(i), K2_value_(i), spin_initial.x, spin_initial.y, spin_initial.z);

    const double e_final = cubic_K1_K2_anisotropy_energy(K1_value_(i), K2_value_(i), spin_final.x, spin_final.y, spin_final.z);

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::calculate_energies() {
    if (solver->is_cuda_solver()) {
#ifdef CUDA
         cuda_anisotropy_cubic_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_>>>
            (globals::num_spins, dev_K1_value_.data(), dev_K2_value_.data(), solver->dev_ptr_spin(), dev_energy_.data());
#endif  // CUDA   
    } else {
        for (int i = 0; i < globals::num_spins; ++i) {
            energy_[i] = calculate_one_spin_energy(i);
        }
  }
}

// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    using std::pow;
    local_field[0] = cubic_K1_K2_anisotropy_field_x(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
    local_field[1] = cubic_K1_K2_anisotropy_field_y(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
    local_field[2] = cubic_K1_K2_anisotropy_field_z(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
}

// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::calculate_fields() {
    using namespace globals;

    // dev_s needs to be found from the solver

    if (solver->is_cuda_solver()) {
#ifdef CUDA
        cuda_anisotropy_cubic_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_>>>
            (globals::num_spins, dev_K1_value_.data(), dev_K2_value_.data(), solver->dev_ptr_spin(), dev_field_.data());
#endif  // CUDA
    } else {
        field_.zero();
        for (int i = 0; i < globals::num_spins; ++i) {
            field_(i, 0) = cubic_K1_K2_anisotropy_field_x(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
            field_(i, 1) = cubic_K1_K2_anisotropy_field_y(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
            field_(i, 2) = cubic_K1_K2_anisotropy_field_z(K1_value_(i), K2_value_(i), s(i, 0), s(i, 1), s(i, 2));
        }
    }
}
// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::output_energies(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_energies_text();
        case HDF5:
            jams_error("Uniaxial energy output: HDF5 not yet implemented");
        default:
            jams_error("Uniaxial energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::output_fields(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_fields_text();
        case HDF5:
            jams_error("Uniaxial energy output: HDF5 not yet implemented");
        default:
            jams_error("Uniaxial energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::output_energies_text() {

}

// --------------------------------------------------------------------------

void AnisotropyCubicHamiltonian::output_fields_text() {

}

double AnisotropyCubicHamiltonian::calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final) {
  if (i != j) {
    return 0.0;
    } else {
  return calculate_one_spin_energy_difference(i, sj_initial, sj_final);
    }
}
