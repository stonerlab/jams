#include "core/globals.h"
#include "core/utils.h"
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

    // deal with magnetocrystalline anisotropy coefficients
    if(settings.exists("d2z")) {
        if (settings["d2z"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: d2z must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            d2z_(i) = double(settings["d2z"][lattice.get_material_number(i)])/mu_bohr_si;
        }
    }

    if(settings.exists("d4z")) {
        if (settings["d4z"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: d4z must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            d4z_(i) = double(settings["d4z"][lattice.get_material_number(i)])/mu_bohr_si;
        }
    }

    if(settings.exists("d6z")) {
        if (settings["d6z"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: d6z must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            d6z_(i) = double(settings["d6z"][lattice.get_material_number(i)])/mu_bohr_si;
        }
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
            K1(i) = double(settings["K1"][lattice.get_material_number(i)])/mu_bohr_si;
        }
    }

    if(settings.exists("K2")) {
        if (settings["K2"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: K2 must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            K2(i) = double(settings["K2"][lattice.get_material_number(i)])/mu_bohr_si;
        }
    }

    if(settings.exists("K3")) {
        if (settings["K3"].getLength() != lattice.num_materials()) {
            jams_error("UniaxialHamiltonian: K3 must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            K3(i) = double(settings["K3"][lattice.get_material_number(i)])/mu_bohr_si;
        }
    }

    for (int i = 0; i < globals::num_spins; ++i) {
        d2z_(i) = (2.0/3.0)*(K1(i) + (8.0/7.0)*K2(i) + (8.0/7.0)*K3(i));
        d4z_(i) = -((8.0/35.0)*K2(i) + (144.0/385.0)*K3(i));
        d6z_(i) = ((16.0/231.0)*K3(i));
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
    return 0.0;
}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    return d2z_(i)*0.5*(3.0*s(i, 2)*s(i, 2) - 1.0)
         + d4z_(i)*0.125*(35.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)-30.0*s(i, 2)*s(i, 2) + 3.0)
         + d6z_(i)*0.0625*(231.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) - 315.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) + 105.0*s(i, 2)*s(i, 2) - 5.0);
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_one_spin_fields(const int i, double h[3]) {
    using namespace globals;
    h[0] = 0.0; h[1] = 0.0;
    h[2] = d2z_(i)*3.0*s(i, 2)
         + d4z_(i)*(17.5*s(i, 2)*s(i, 2)*s(i, 2)-7.5*s(i, 2))
         + d6z_(i)*(86.625*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) - 78.75*s(i, 2)*s(i, 2)*s(i, 2) + 13.125*s(i, 2));
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_fields() {

    // dev_s needs to be found from the solver

    if (solver->is_cuda_solver()) {
#ifdef CUDA
        cuda_uniaxial_field_kernel<<<(globals::num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE >>>
            (globals::num_spins, dev_d2z_.data(), dev_d4z_.data(), dev_d6z_.data(), solver->dev_ptr_spin(), dev_field_.data());
#endif  //
    } else {
        for (int i = 0; i < globals::num_spins; ++i) {
            double h[3];
            calculate_one_spin_fields(i, h);
            for(int j = 0; j < 3; ++j) {
                field_(i,j) = h[j];
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
        outfile << lattice.lattice_material_num_[i];

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile <<  lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
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
        outfile << std::setw(16) << lattice.lattice_material_num_[i];

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::fixed << lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
        }

        // fields
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::scientific << field_(i,j);
        }
        outfile << "\n";
    }
    outfile.close();
}