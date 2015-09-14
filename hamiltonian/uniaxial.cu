#include "core/globals.h"
#include "core/utils.h"
#include "core/maths.h"
#include "core/consts.h"
#include "core/cuda_defs.h"


#include "hamiltonian/uniaxial.h"
#include "hamiltonian/uniaxial_kernel.h"

UniaxialHamiltonian::UniaxialHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings) {

    // output in default format for now
    outformat_ = TEXT;

    // resize member arrays
    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);

    has_d2z_ = false;
    has_d4z_ = false;
    has_d6z_ = false;

    d2z_.resize(globals::num_spins);
    d4z_.resize(globals::num_spins);
    d6z_.resize(globals::num_spins);

    for (int i = 0; i < globals::num_spins; ++i) {
        energy_(i) = 0.0;
        d2z_(i) = 0.0;
        d4z_(i) = 0.0;
        d6z_(i) = 0.0;
        for (int j = 0; j < 3; ++j) {
            field_(i, j) = 0.0;
        }
    }

    // don't allow mixed specification of anisotropy
    if ( (settings.exists("K1") || settings.exists("K2") || settings.exists("K3")) &&
         (settings.exists("d2z") || settings.exists("d4z") || settings.exists("d6z")) ) {
      jams_error("UniaxialHamiltonian: anisotropy should only be specified in terms of K1, K2, K3 or d2z, d4z, d6z in the config file");
    }

    // deal with magnetic anisotropy constants
    jblib::Array<double, 1> K1(globals::num_spins, 0.0);
    jblib::Array<double, 1> K2(globals::num_spins, 0.0);
    jblib::Array<double, 1> K3(globals::num_spins, 0.0);

    if(settings.exists("K1")) {
        if (settings["K1"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: K1 must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            K1(i) = double(settings["K1"][lattice.material(i)])/kBohrMagneton;
        }
        has_d2z_ = true;
    }

    if(settings.exists("K2")) {
        if (settings["K2"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: K2 must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            K2(i) = double(settings["K2"][lattice.material(i)])/kBohrMagneton;
        }
        has_d2z_ = true;
        has_d4z_ = true;
    }

    if(settings.exists("K3")) {
        if (settings["K3"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: K3 must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            K3(i) = double(settings["K3"][lattice.material(i)])/kBohrMagneton;
        }
        has_d2z_ = true;
        has_d4z_ = true;
        has_d6z_ = true;
    }

    for (int i = 0; i < globals::num_spins; ++i) {
        d2z_(i) = -(2.0/3.0)*(K1(i) + (8.0/7.0)*K2(i) + (8.0/7.0)*K3(i));
        d4z_(i) = ((8.0/35.0)*K2(i) + (144.0/385.0)*K3(i));
        d6z_(i) = -((16.0/231.0)*K3(i));
    }

    // deal with magnetocrystalline anisotropy coefficients
    if(settings.exists("d2z")) {
        if (settings["d2z"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: d2z must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            d2z_(i) = double(settings["d2z"][lattice.material(i)])/kBohrMagneton;
        }
        has_d2z_ = true;
    }

    if(settings.exists("d4z")) {
        if (settings["d4z"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: d4z must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            d4z_(i) = double(settings["d4z"][lattice.material(i)])/kBohrMagneton;
        }
        has_d4z_ = true;
    }

    if(settings.exists("d6z")) {
        if (settings["d6z"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: d6z must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            d6z_(i) = double(settings["d6z"][lattice.material(i)])/kBohrMagneton;
        }
        has_d6z_ = true;
    }

    // transfer arrays to cuda device if needed
#ifdef CUDA
    if (solver->is_cuda_solver()) {
        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_ = jblib::CudaArray<double, 1>(field_);

        dev_d2z_ = jblib::CudaArray<double, 1>(d2z_);
        dev_d4z_ = jblib::CudaArray<double, 1>(d4z_);
        dev_d6z_ = jblib::CudaArray<double, 1>(d6z_);
    }
#endif

}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_total_energy() {
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_one_spin_energy(i);
    }
     return e_total;
}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    return d2z_(i)*0.5*(3.0*s(i, 2)*s(i, 2) - 1.0)
         + d4z_(i)*0.125*(35.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)-30.0*s(i, 2)*s(i, 2) + 3.0)
         + d6z_(i)*0.0625*(231.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) - 315.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) + 105.0*s(i, 2)*s(i, 2) - 5.0);
}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    using std::pow;

    double e_initial = 0.0;
    double e_final = 0.0;

    if (has_d2z_) {
        e_initial += d2z_(i) * legendre_poly(spin_initial.z, 2);
        e_final += d2z_(i) * legendre_poly(spin_final.z, 2);
    }

    if (has_d4z_) {
        e_initial += d4z_(i) * legendre_poly(spin_initial.z, 4);
        e_final += d4z_(i) * legendre_poly(spin_final.z, 4);
    }

    if (has_d6z_) {
        e_initial += d6z_(i) * legendre_poly(spin_initial.z, 6);
        e_final += d6z_(i) * legendre_poly(spin_final.z, 6);
    }

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    using std::pow;
    const double sz = s(i, 2);
    local_field[0] = 0.0;
    local_field[1] = 0.0;
    local_field[2] = 0.0;

    if (has_d2z_) {
        local_field[2] += -d2z_(i) * legendre_dpoly(sz, 2);
    }

    if (has_d4z_) {
        local_field[2] += -d4z_(i) * legendre_dpoly(sz, 4);
    }

    if (has_d6z_) {
        local_field[2] += -d6z_(i) * legendre_dpoly(sz, 6);
    }
}



// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_fields() {

    // dev_s needs to be found from the solver

    if (solver->is_cuda_solver()) {
#ifdef CUDA
        cuda_uniaxial_field_kernel<<<(globals::num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE >>>
            (globals::num_spins, dev_d2z_.data(), dev_d4z_.data(), dev_d6z_.data(), solver->dev_ptr_spin(), dev_field_.data());
#endif  // CUDA
    } else {
        for (int i = 0; i < globals::num_spins; ++i) {
            field_(i, 0) = 0.0;
            field_(i, 1) = 0.0;
            field_(i, 0) = 0.0;
            const double sz = globals::s(i, 2);

            if (has_d2z_) {
                field_(i, 2) += -d2z_(i) * legendre_dpoly(sz, 2);
            }

            if (has_d4z_) {
                field_(i, 2) += -d4z_(i) * legendre_dpoly(sz, 4);
            }

            if (has_d6z_) {
                field_(i, 2) += -d6z_(i) * legendre_dpoly(sz, 6);
            }
        }
    }
}
// --------------------------------------------------------------------------

void UniaxialHamiltonian::output_energies(OutputFormat format) {
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

void UniaxialHamiltonian::output_fields(OutputFormat format) {
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

void UniaxialHamiltonian::output_energies_text() {
    using namespace globals;

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_energy_.copy_to_host_array(energy_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_eng_uniaxial_"+zero_pad_number(outcount)+".dat");

    std::ofstream outfile(filename.c_str());

    outfile << "# type | rx (nm) | ry (nm) | rz (nm) | d2z | d4z | d6z" << std::endl;

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << lattice.material(i);

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile <<  lattice.parameter()*lattice.position(i)[j];
        }

        // energy
        outfile << d2z_(i)*0.5*(3.0*s(i, 2)*s(i, 2) - 1.0);
        outfile << d4z_(i)*(17.5*s(i, 2)*s(i, 2)*s(i, 2)-7.5*s(i, 2));
        outfile << d6z_(i)*(86.625*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) - 78.75*s(i, 2)*s(i, 2)*s(i, 2) + 13.125*s(i, 2));
    }
    outfile.close();
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::output_fields_text() {

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_field_.copy_to_host_array(field_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_field_uniaxial_"+zero_pad_number(outcount)+".dat");

    // using direct file access for performance
    std::ofstream outfile(filename.c_str());
    outfile.setf(std::ios::right);

    outfile << "#";
    outfile << std::setw(16) << "type";
    outfile << std::setw(16) << "rx (nm)";
    outfile << std::setw(16) << "ry (nm)";
    outfile << std::setw(16) << "rz (nm)";
    outfile << std::setw(16) << "hx (nm)";
    outfile << std::setw(16) << "hy (nm)";
    outfile << std::setw(16) << "hz (nm)";
    outfile << "\n";

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << std::setw(16) << lattice.material(i);

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::fixed << lattice.parameter()*lattice.position(i)[j];
        }

        // fields
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::scientific << field_(i,j);
        }
        outfile << "\n";
    }
    outfile.close();
}
