#include <cublas_v2.h>

#include "core/globals.h"
#include "core/utils.h"
#include "core/maths.h"
#include "core/consts.h"
#include "core/cuda_defs.h"
#include "core/cuda_array_kernels.h"

#include "hamiltonian/demag.h"
#include "hamiltonian/demag_kernel.h"

DemagHamiltonian::DemagHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings)
{
    ::output.write("initialising Demag Hamiltonian\n");
    outformat_ = TEXT;

    // resize member arrays
    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);
    field_.zero();

    for (int i = 0; i < 3; ++i) {
        factors_[i] = double(settings["factors"][i]);
    }

    if (!floats_are_equal(1.0, sum(factors_))) {
        jams_warning("dipole factor array is not normalized - normalizing now");
        factors_ = factors_ / abs(factors_);
    }

    factors_ = factors_ * kTwoPi / double(globals::num_spins);

    // transfer arrays to cuda device if needed
#ifdef CUDA
    ::output.write("  initialising CUDA device\n");

    if (solver->is_cuda_solver()) {

        cuda_api_error_check(cudaStreamCreate(&dev_stream_));
        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_  = jblib::CudaArray<double, 1>(field_);
        
        ::output.write("    copying mus\n");

        dev_mus_ = jblib::CudaArray<double, 1>(globals::mus);

        ::output.write("    resizing dev_moments_\n");

        dev_moments_ = jblib::CudaArray<double, 1>(globals::s);
    }




    dev_blocksize_ = 128;
#endif

}

// --------------------------------------------------------------------------

double DemagHamiltonian::calculate_total_energy() {
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_one_spin_energy(i);
    }
     return e_total;
}

// --------------------------------------------------------------------------

double DemagHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    double energy = 0.0;


    return energy;
}

// --------------------------------------------------------------------------

double DemagHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    using std::pow;

    double e_initial = 0.0;
    double e_final = 0.0;

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void DemagHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DemagHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    using std::pow;
    local_field[0] = 0.0;
    local_field[1] = 0.0;
    local_field[2] = 0.0;

}



// --------------------------------------------------------------------------

void DemagHamiltonian::calculate_fields() {

    if (solver->is_cuda_solver()) {
#ifdef CUDA

        // calculate mu_s * (sx, sy, sz)
        cuda_array_elementwise_scale(globals::num_spins, 3, dev_mus_.data(), 1.0, solver->dev_ptr_spin(), 1, dev_moments_.data(), 1, dev_stream_);

        // calculate magentization in each direction
        // Mx

        jblib::Vec3<double> mag;

        mag.x = cuda_array_sum(globals::num_spins, dev_moments_.data(), 3, dev_stream_);

        mag.y = cuda_array_sum(globals::num_spins, dev_moments_.data()+1, 3, dev_stream_);

        mag.z = cuda_array_sum(globals::num_spins, dev_moments_.data()+2, 3, dev_stream_);


        // std::cerr << mag.x << "\t" << mag.y << "\t" << mag.z << std::endl;

        // assign demag field to each field element

        //H = 2*pi*(1/N) * D * m

        cuda_demag_field_kernel<<<(globals::num_spins + dev_blocksize_ - 1) / dev_blocksize_, dev_blocksize_, 0, dev_stream_>>>
        (globals::num_spins, factors_.x, factors_.y, factors_.z, mag[0], mag[1], mag[2], dev_mus_.data(), dev_field_.data());


#endif  // CUDA
    } else {
        field_.zero();
    }
}
// --------------------------------------------------------------------------

void DemagHamiltonian::output_energies(OutputFormat format) {
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

void DemagHamiltonian::output_fields(OutputFormat format) {
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

void DemagHamiltonian::output_energies_text() {

}

// --------------------------------------------------------------------------

void DemagHamiltonian::output_fields_text() {

}

double DemagHamiltonian::calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final) {
  if (i != j) {
    return 0.0;
    } else {
  return calculate_one_spin_energy_difference(i, sj_initial, sj_final);
    }
}
