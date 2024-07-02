#include <jams/hamiltonian/cpu_crystal_field.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>

CPUCrystalFieldHamiltonian::CPUCrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size)
: Hamiltonian(settings, size) {
    auto& cf_coeff_settings = settings["crystal_field_coefficients"];

    // validate input
    for (auto i = 0; i < cf_coeff_settings.getLength(); ++i) {
        if ( !(cf_coeff_settings[i][0].isNumber() || cf_coeff_settings[i][0].isString())) {
            // TODO: raise error that first number must be a number or letter
        }

            // TODO: check if the strings are valid materials and the numbers are valid positions

        if (cf_coeff_settings[i][0].getType() != cf_coeff_settings[0][0].getType()) {
            // TODO: raise error that the first indices must all have the same types
        }

        if ( !cf_coeff_settings[i][1].isArray() ||  !cf_coeff_settings[i][1][0].isNumber()) {
            // TODO: raise error that second entry must be an array of numbers
        }

        if ( cf_coeff_settings[i][1].getLength() != kCrystalFieldNumCoeff_ ) {
            // TODO: raise error that second entry must be of length kMaxCrystalFieldCoeff_
        }
    }

    // TODO: need to support reading in complex values
    // Read crystal field coefficients into a buffer
    jams::MultiArray<double, 2> cf_coeff_buffer(kCrystalFieldNumCoeff_, size);
    cf_coeff_buffer.zero();

    for (auto setting_idx = 0; setting_idx < cf_coeff_settings.getLength(); ++setting_idx) {
        if (cf_coeff_settings[setting_idx][0].isNumber()) {
            unsigned motif_position = cf_coeff_settings[setting_idx][0];
            for (auto i = 0; i < globals::num_spins; i++) {
                if (globals::lattice->atom_motif_position(i) == motif_position - 1) {
                    for (auto n = 0; n < kCrystalFieldNumCoeff_; ++n) {
                        cf_coeff_buffer(i, n) = cf_coeff_settings[setting_idx][1][n];
                    }
                }
            }
        }

        if (cf_coeff_settings[setting_idx][0].isString()) {
            std::string material_name(cf_coeff_settings[setting_idx][0].c_str());
            for (auto i = 0; i < globals::num_spins; i++) {
                if (globals::lattice->atom_material_name(i) == material_name) {
                    for (auto n = 0; n < kCrystalFieldNumCoeff_; ++n) {
                        cf_coeff_buffer(i, n) = cf_coeff_settings[setting_idx][1][n];
                    }
                }
            }
        }
    }

    // Count the number of spins with non-zero crystal field
    std::vector<int> spin_index_buffer;
    for (auto i = 0; i < globals::num_spins; i++) {
        for (auto n = 0; n < kCrystalFieldNumCoeff_; ++n) {
            if (cf_coeff_buffer(i, n) != 0.0) {
                spin_index_buffer.push_back(i);
                break; // we don't need to check any more coefficients for this spin
            }
        }
    }

    // convert crystal field
    crystal_field_spin_indices_ = jams::MultiArray<int,1>(spin_index_buffer.begin(), spin_index_buffer.end());
    crystal_field_tesseral_coeff_.resize(kCrystalFieldNumCoeff_, spin_index_buffer.size());

    for (auto spin_index : spin_index_buffer) {
        for (auto n = 0; n < kCrystalFieldNumCoeff_; ++n) {
            auto B_lm = cf_coeff_buffer(spin_index_buffer[spin_index], n);
            // need to map l,m indices to the linear indices
//            crystal_field_tesseral_coeff_(spin_index, n) =
        }
    }


}