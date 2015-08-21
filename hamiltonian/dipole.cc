#include "core/globals.h"
#include "core/utils.h"

#include "hamiltonian/dipole.h"
#include "hamiltonian/dipole_bruteforce.h"
#include "hamiltonian/dipole_tensor.h"
#include "hamiltonian/dipole_ewald.h"

DipoleHamiltonian::DipoleHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings) {

    // output in default format for now
    outformat_ = TEXT;

    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_energy_.resize(globals::num_spins);
        dev_field_.resize(globals::num_spins3);
    }
#endif

    dipole_strategy = new DipoleHamiltonianEwald(settings);

}

// --------------------------------------------------------------------------

double DipoleHamiltonian::calculate_total_energy() {
    return dipole_strategy->calculate_total_energy();
}

// --------------------------------------------------------------------------

double DipoleHamiltonian::calculate_one_spin_energy(const int i) {
    return dipole_strategy->calculate_one_spin_energy(i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    return dipole_strategy->calculate_one_spin_energy_difference(i, spin_initial, spin_final);
}
// --------------------------------------------------------------------------

void DipoleHamiltonian::calculate_energies() {
    dipole_strategy->calculate_energies(energy_);
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::calculate_one_spin_field(const int i, double h[3]) {
    dipole_strategy->calculate_one_spin_field(i, h);
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::calculate_fields() {
    dipole_strategy->calculate_fields(field_);
}
// --------------------------------------------------------------------------

void DipoleHamiltonian::output_energies(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_energies_text();
        case HDF5:
            jams_error("Dipole energy output: HDF5 not yet implemented");
        default:
            jams_error("Dipole energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::output_fields(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_fields_text();
        case HDF5:
            jams_error("Dipole energy output: HDF5 not yet implemented");
        default:
            jams_error("Dipole energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::output_energies_text() {

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_energy_.copy_to_host_array(energy_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_eng_dipole_"+zero_pad_number(outcount)+".dat");

    // using direct file access for performance
    std::ofstream outfile(filename.c_str());

    outfile << "# type | rx (nm) | ry (nm) | rz (nm) | energy" << std::endl;

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << lattice.lattice_material_num_[i];

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile <<  lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
        }

        // energy
        outfile << energy_[i];
    }
    outfile.close();
}

// --------------------------------------------------------------------------

void DipoleHamiltonian::output_fields_text() {

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_field_.copy_to_host_array(field_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_field_dipole_"+zero_pad_number(outcount)+".dat");

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
// --------------------------------------------------------------------------
