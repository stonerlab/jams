#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"

#include "anisotropy_uniaxial.h"
#include "anisotropy_uniaxial_kernel.h"

UniaxialHamiltonian::UniaxialHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
: Hamiltonian(settings, num_spins),
  mca_order_(),
  mca_value_()
{
    bool has_d2z = false;
    bool has_d4z = false;
    bool has_d6z = false;


    // don't allow mixed specification of anisotropy
    if ( (settings.exists("K1") || settings.exists("K2") || settings.exists("K3")) &&
         (settings.exists("d2z") || settings.exists("d4z") || settings.exists("d6z")) ) {
      jams_error("UniaxialHamiltonian: anisotropy should only be specified in terms of K1, K2, K3 or d2z, d4z, d6z in the config file");
    }

    // deal with magnetic anisotropy constants
    jblib::Array<double, 1> K1(num_spins, 0.0);
    jblib::Array<double, 1> K2(num_spins, 0.0);
    jblib::Array<double, 1> K3(num_spins, 0.0);

    if(settings.exists("K1")) {
        if (settings["K1"].getLength() != lattice->num_materials()) {
            jams_error("UniaxialHamiltonian: K1 must be specified for every material");
        }
        for (int i = 0; i < globals::num_spins; ++i) {
            K1(i) = double(settings["K1"][lattice->atom_material_id(i)])/kBohrMagneton;
        }
        has_d2z = true;
    }


    if(settings.exists("K2")) {
        if (settings["K2"].getLength() != lattice->num_materials()) {
            jams_error("UniaxialHamiltonian: K2 must be specified for every material");
        }
        for (int i = 0; i < num_spins; ++i) {
            K2(i) = double(settings["K2"][lattice->atom_material_id(i)])/kBohrMagneton;
        }
        has_d2z = true;
        has_d4z = true;
    }

    if(settings.exists("K3")) {
        if (settings["K3"].getLength() != lattice->num_materials()) {
            jams_error("UniaxialHamiltonian: K3 must be specified for every material");
        }
        for (int i = 0; i < num_spins; ++i) {
            K3(i) = double(settings["K3"][lattice->atom_material_id(i)])/kBohrMagneton;
        }
        has_d2z = true;
        has_d4z = true;
        has_d6z = true;
    }

    if (has_d2z) {
        mca_order_.push_back(2);
        jblib::Array<double, 1> mca(num_spins, 0.0);
        for (int i = 0; i < num_spins; ++i) {
            mca(i) = -(2.0/3.0)*(K1(i) + (8.0/7.0)*K2(i) + (8.0/7.0)*K3(i));
        }
        mca_value_.push_back(mca);
    }


    if (has_d4z) {
        mca_order_.push_back(4);
        jblib::Array<double, 1> mca(num_spins, 0.0);
        for (int i = 0; i < num_spins; ++i) {
            mca(i) = ((8.0/35.0)*K2(i) + (144.0/385.0)*K3(i));
        }
        mca_value_.push_back(mca);
    }
    if (has_d6z) {
        mca_order_.push_back(6);
        jblib::Array<double, 1> mca(num_spins, 0.0);
        for (int i = 0; i < num_spins; ++i) {
            mca(i) = -((16.0/231.0)*K3(i));
        }
        mca_value_.push_back(mca);
    }


    // deal with magnetocrystalline anisotropy coefficients
    if(settings.exists("d2z")) {
        if (settings["d2z"].getLength() != lattice->num_materials()) {
            jams_error("UniaxialHamiltonian: d2z must be specified for every material");
        }
        mca_order_.push_back(2);

        jblib::Array<double, 1> mca(num_spins, 0.0);
        for (int i = 0; i < num_spins; ++i) {
            mca(i) = double(settings["d2z"][lattice->atom_material_id(i)])/kBohrMagneton;
        }
        mca_value_.push_back(mca);
    }



    if(settings.exists("d4z")) {
        if (settings["d4z"].getLength() != lattice->num_materials()) {
            jams_error("UniaxialHamiltonian: d4z must be specified for every material");
        }
        mca_order_.push_back(4);
        jblib::Array<double, 1> mca(num_spins, 0.0);
        for (int i = 0; i < num_spins; ++i) {
            mca(i) = double(settings["d4z"][lattice->atom_material_id(i)])/kBohrMagneton;
        }
        mca_value_.push_back(mca);
    }

    if(settings.exists("d6z")) {
        if (settings["d6z"].getLength() != lattice->num_materials()) {
            jams_error("UniaxialHamiltonian: d6z must be specified for every material");
        }
        mca_order_.push_back(6);
        jblib::Array<double, 1> mca(num_spins, 0.0);
        for (int i = 0; i < num_spins; ++i) {
            mca(i) = double(settings["d6z"][lattice->atom_material_id(i)])/kBohrMagneton;
        }
        mca_value_.push_back(mca);
    }


    // transfer arrays to cuda device if needed
#if HAS_CUDA
    if (solver->is_cuda_solver()) {
        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_ = jblib::CudaArray<double, 1>(field_);

        jblib::Array<int, 1> tmp_mca_order(mca_order_.size());
        for (int i = 0; i < mca_order_.size(); ++i) {
            tmp_mca_order[i] = mca_order_[i];
        }

        dev_mca_order_ = jblib::CudaArray<int, 1>(tmp_mca_order);

        jblib::Array<double, 1> tmp_mca_value(mca_order_.size() * num_spins);

        for (int i = 0; i < num_spins; ++i) {
            for (int j = 0; j < mca_order_.size(); ++j) {
                tmp_mca_value[ mca_order_.size() * i + j] = mca_value_[j](i);
            }
        }
        dev_mca_value_ = tmp_mca_value;
    }

    cudaStreamCreate(&dev_stream_);

    dev_blocksize_ = 128;
#endif

}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_total_energy() {
    double e_total = 0.0;
    for (int i = 0; i < energy_.size(); ++i) {
        e_total += calculate_one_spin_energy(i);
    }
     return e_total;
}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_one_spin_energy(const int i) {
    double energy = 0.0;

    for (int n = 0; n < mca_order_.size(); ++n) {
        energy += mca_value_[n](i) * legendre_poly(globals::s(i, 2), mca_order_[n]);
    }

    return energy;
}

// --------------------------------------------------------------------------

double UniaxialHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    double e_initial = 0.0;
    double e_final = 0.0;

    for (int n = 0; n < mca_order_.size(); ++n) {
        e_initial += mca_value_[n](i) * legendre_poly(spin_initial[2], mca_order_[n]);
    }

    for (int n = 0; n < mca_order_.size(); ++n) {
        e_final += mca_value_[n](i) * legendre_poly(spin_final[2], mca_order_[n]);
    }

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_energies() {
    for (int i = 0; i < energy_.size(); ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    const double sz = globals::s(i, 2);
    local_field[0] = 0.0;
    local_field[1] = 0.0;
    local_field[2] = 0.0;

    for (int n = 0; n < mca_order_.size(); ++n) {
        local_field[2] += -mca_value_[n](i) * legendre_dpoly(sz, mca_order_[n]);
    }
}

// --------------------------------------------------------------------------

void UniaxialHamiltonian::calculate_fields() {
    if (solver->is_cuda_solver()) {
#if HAS_CUDA
        cuda_uniaxial_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_>>>
            (globals::num_spins, mca_order_.size(), dev_mca_order_.data(), dev_mca_value_.data(), solver->dev_ptr_spin(), dev_field_.data());
#endif  // CUDA
    } else {
        field_.zero();
        for (int n = 0; n < mca_order_.size(); ++n) {
            for (int i = 0; i < field_.size(0); ++i) {
                field_(i, 2) += -mca_value_[n](i) * legendre_dpoly(globals::s(i, 2), mca_order_[n]);
            }
        }
    }
}
