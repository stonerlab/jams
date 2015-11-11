#include "core/globals.h"
#include "core/utils.h"
#include "core/maths.h"
#include "core/consts.h"
#include "core/cuda_defs.h"

#include "hamiltonian/zeeman.h"
#include "hamiltonian/zeeman_kernel.h"

ZeemanHamiltonian::ZeemanHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings)
{
    ::output.write("initialising Zeeman Hamiltonian\n");
    // output in default format for now
    outformat_ = TEXT;

    // resize member arrays
    energy_.resize(globals::num_spins);
    energy_.zero();
    field_.resize(globals::num_spins, 3);
    field_.zero();

    dc_local_field_.resize(globals::num_spins, 3);
    dc_local_field_.zero();


    ac_local_field_.resize(globals::num_spins, 3);
    ac_local_frequency_.resize(globals::num_spins);

    ac_local_field_.zero();
    ac_local_frequency_.zero();


    if(settings.exists("dc_local_field")) {
        if (settings["dc_local_field"].getLength() != lattice.num_materials()) {
            jams_error("ZeemanHamiltonian: dc_local_field must be specified for every material");
        }


        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                dc_local_field_(i, j) = settings["dc_local_field"][lattice.material(i)][j];
                dc_local_field_(i, j) *= globals::mus(i);
            }
        }
    }

    if(settings.exists("ac_local")) {
        if (settings["ac_local"].getLength() != lattice.num_materials()) {
            jams_error("ZeemanHamiltonian: ac_local must be specified for every material");
        }
    }

    if(settings.exists("ac_local_field") || settings.exists("ac_local_frequency")) {
        if(!(settings.exists("ac_local_field") && settings.exists("ac_local_frequency"))) {
            jams_error("ZeemanHamiltonian: ac_local must have a field and a frequency");
        }
        if (settings["ac_local_frequency"].getLength() != lattice.num_materials()) {
            jams_error("ZeemanHamiltonian: ac_local_frequency must be specified for every material");
        }
        if (settings["ac_local_field"].getLength() != lattice.num_materials()) {
            jams_error("ZeemanHamiltonian: ac_local_field must be specified for every material");
        }

        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                ac_local_field_(i, j) = settings["ac_local_field"][lattice.material(i)][j];
                ac_local_field_(i, j) *= globals::mus(i);
            }
        }

        for (int i = 0; i < globals::num_spins; ++i) {
            ac_local_frequency_(i) = settings["ac_local_frequency"][lattice.material(i)];
            ac_local_frequency_(i) = kTwoPi*ac_local_frequency_(i);
        }
    }

    // transfer arrays to cuda device if needed
#ifdef CUDA
    if (solver->is_cuda_solver()) {
        cudaStreamCreate(&dev_stream_);

        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_  = jblib::CudaArray<double, 1>(field_);

        dev_dc_local_field_ = jblib::CudaArray<double, 1>(dc_local_field_);

        dev_ac_local_field_ = jblib::CudaArray<double, 1>(ac_local_field_);
        dev_ac_local_frequency_ = jblib::CudaArray<double, 1>(ac_local_frequency_);
    }
#endif

}

// --------------------------------------------------------------------------

double ZeemanHamiltonian::calculate_total_energy() {
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_one_spin_energy(i);
    }
     return e_total;
}

// --------------------------------------------------------------------------

double ZeemanHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    double one_spin_field[3];

    calculate_one_spin_field(i, one_spin_field);

    return -(s(i, 0)*one_spin_field[0] + s(i, 1)*one_spin_field[1] + s(i, 2)*one_spin_field[2]);
}

// --------------------------------------------------------------------------

double ZeemanHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    using std::pow;

    double e_initial = 0.0;
    double e_final = 0.0;

    double h_local[3];

    calculate_one_spin_field(i, h_local);

    for (int n = 0; n < 3; ++n) {
        e_initial += -spin_initial[n]*h_local[n];
    }

    for (int n = 0; n < 3; ++n) {
        e_final += -spin_final[n]*h_local[n];
    }

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void ZeemanHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void ZeemanHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    using std::pow;

    for (int j = 0; j < 3; ++j) {
        local_field[j] = dc_local_field_(i, j) + ac_local_field_(i, j) * cos(ac_local_frequency_(i) * solver->time());
    }
}



// --------------------------------------------------------------------------

void ZeemanHamiltonian::calculate_fields() {
    if (solver->is_cuda_solver()) {
#ifdef CUDA
        dim3 block_size;
        block_size.x = 32;
        block_size.y = 4;

        dim3 grid_size;
        grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;
        grid_size.y = (3 + block_size.y - 1) / block_size.y;


        cuda_zeeman_field_kernel<<<grid_size, block_size, 0, dev_stream_>>>
            (globals::num_spins, solver->time(), dev_dc_local_field_.data(),
                dev_ac_local_field_.data(), dev_ac_local_frequency_.data(),
                solver->dev_ptr_spin(), dev_field_.data());
        cuda_kernel_error_check();
#endif  // CUDA
    } else {
        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                field_(i, j) = dc_local_field_(i, j) + ac_local_field_(i, j) * cos(ac_local_frequency_(i) * solver->time());
            }
        }
    }
}
// --------------------------------------------------------------------------

void ZeemanHamiltonian::output_energies(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_energies_text();
        case HDF5:
            jams_error("Zeeman energy output: HDF5 not yet implemented");
        default:
            jams_error("Zeeman energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void ZeemanHamiltonian::output_fields(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_fields_text();
        case HDF5:
            jams_error("Zeeman energy output: HDF5 not yet implemented");
        default:
            jams_error("Zeeman energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void ZeemanHamiltonian::output_energies_text() {

}

// --------------------------------------------------------------------------

void ZeemanHamiltonian::output_fields_text() {

}
