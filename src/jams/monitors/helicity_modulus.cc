//
// Created by Sean Stansill [ll14s26s] on 26/09/2022.
//

#include <string>
#include <iomanip>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/output.h"
#include "jams/containers/sparse_matrix_builder.h"
#include "jams/hamiltonian/exchange.h"

#include "helicity_modulus.h"

using namespace std;

HelicityModulusMonitor::HelicityModulusMonitor(const libconfig::Setting &settings)
        : Monitor(settings), tsv_file(jams::output::full_path_filename("free_eng.tsv")) {
    tsv_file.setf(std::ios::right);
    tsv_file << tsv_header();

    const auto& exchange_hamiltonian = find_hamiltonian<ExchangeHamiltonian>(::solver->hamiltonians());

    jams::SparseMatrix<double>::Builder sparse_matrix_builder_J(globals::num_spins, globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_x(globals::num_spins, globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_y(globals::num_spins, globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_z(globals::num_spins, globals::num_spins);

    const auto& nbr_list = exchange_hamiltonian.neighbour_list();
    for (auto n = 0; n < nbr_list.size(); ++n) {
        auto i = nbr_list[n].first[0];
        auto j = nbr_list[n].first[1];
        auto Jij = nbr_list[n].second[0][0];
        sparse_matrix_builder_J.insert(i, j, Jij);
        sparse_matrix_builder_x.insert(i, j, (lattice->displacement(i, j))[0]);
        sparse_matrix_builder_y.insert(i, j, (lattice->displacement(i, j))[1]);
        sparse_matrix_builder_z.insert(i, j, (lattice->displacement(i, j))[2]);
    }
    cout << "    dipole sparse matrix builder memory " << (sparse_matrix_builder_x.memory() + sparse_matrix_builder_y.memory() + sparse_matrix_builder_z.memory()) / kBytesToMegaBytes << "(MB)\n";
    cout << "    building CSR matrix\n";
    interaction_Jij_ = sparse_matrix_builder_J
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_rij_x_ = sparse_matrix_builder_x
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_rij_y_ = sparse_matrix_builder_y
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_rij_z_ = sparse_matrix_builder_z
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    cout << "    exchange sparse matrix memory (CSR): " << (interaction_rij_x_.memory() + interaction_rij_y_.memory() + interaction_rij_z_.memory()) / kBytesToMegaBytes << " (MB)\n";


    helicity_field_rxx_.resize(globals::num_spins, 3);
    helicity_field_rxy_.resize(globals::num_spins, 3);
    helicity_field_rxz_.resize(globals::num_spins, 3);
    helicity_field_ryy_.resize(globals::num_spins, 3);
    helicity_field_ryz_.resize(globals::num_spins, 3);
    helicity_field_rzz_.resize(globals::num_spins, 3);

    entropy_field_x_.resize(globals::num_spins, 3);
    entropy_field_y_.resize(globals::num_spins, 3);
    entropy_field_z_.resize(globals::num_spins, 3);

}

void HelicityModulusMonitor::update(Solver * solver) {
    using namespace globals;

    const double beta = 1.0 / (kBoltzmannIU * solver->physics()->temperature());
    tsv_file.width(12);

    tsv_file << std::scientific << solver->time() << "\t";

    for (auto &hamiltonian : solver->hamiltonians()) {
        if (hamiltonian->name() == "exchange") {

            calculate_helicity_fields();
            calculate_entropy_fields();

            Mat3 energy_difference = exchange_total_internal_energy_difference();
            Mat3 entropy = exchange_total_entropy();

            entropy *= beta;

            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    tsv_file << std::scientific << std::setprecision(15) << energy_difference[i][j] << "\t";
                    tsv_file << std::scientific << std::setprecision(15) << entropy[i][j] << "\t";
                }
            }
        }

        else {
            auto energy_difference = hamiltonian->calculate_total_internal_energy_difference();
            auto entropy = hamiltonian->calculate_total_entropy();
            entropy *= beta;

            tsv_file << std::scientific << std::setprecision(15) << energy_difference << "\t";
            tsv_file << std::scientific << std::setprecision(15) << entropy << "\t";
        }
    }

    tsv_file << std::endl;
}

std::string HelicityModulusMonitor::tsv_header() {
    std::stringstream ss;
    ss.width(12);

    ss << "time\t";
    for (auto &hamiltonian : solver->hamiltonians()) {
        if (hamiltonian->name() == "exchange") {
            string unit_vecs[3] = {"x", "y", "z"};
            for (auto i = 0; i < 3; ++i){
                for (auto j = 0; j < 3; ++j){
                    ss << hamiltonian->name() << "_dU_" << unit_vecs[i] << unit_vecs[j] <<"_E_meV\t";
                    ss << hamiltonian->name() << "_TS_" << unit_vecs[i] << unit_vecs[j] <<"_E_meV\t";
                }
            }
        }

        else {
            ss << hamiltonian->name() << "_dU_E_meV\t";
            ss << hamiltonian->name() << "_TS_E_mev\t";

        }
    }

    ss << std::endl;

    return ss.str();
}

void HelicityModulusMonitor::calculate_helicity_fields() {
#if HAS_CUDA
    if (jams::instance().mode() == jams::Mode::GPU) {
        interaction_rij_x_.multiply_gpu(globals::s, helicity_field_rxx_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_x_.multiply_gpu(globals::s, helicity_field_rxx_, jams::instance().cusparse_handle(), cusparse_stream_.get());

        interaction_rij_x_.multiply_gpu(globals::s, helicity_field_rxy_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_y_.multiply_gpu(globals::s, helicity_field_rxy_, jams::instance().cusparse_handle(), cusparse_stream_.get());

        interaction_rij_x_.multiply_gpu(globals::s, helicity_field_rxz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_z_.multiply_gpu(globals::s, helicity_field_rxz_, jams::instance().cusparse_handle(), cusparse_stream_.get());


        interaction_rij_y_.multiply_gpu(globals::s, helicity_field_ryy_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_y_.multiply_gpu(globals::s, helicity_field_ryy_, jams::instance().cusparse_handle(), cusparse_stream_.get());

        interaction_rij_y_.multiply_gpu(globals::s, helicity_field_ryz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_z_.multiply_gpu(globals::s, helicity_field_ryz_, jams::instance().cusparse_handle(), cusparse_stream_.get());

        interaction_rij_z_.multiply_gpu(globals::s, helicity_field_rzz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_z_.multiply_gpu(globals::s, helicity_field_rzz_, jams::instance().cusparse_handle(), cusparse_stream_.get());

        interaction_Jij_.multiply_gpu(helicity_field_rxx_, helicity_field_rxx_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(helicity_field_rxy_, helicity_field_rxy_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(helicity_field_rxz_, helicity_field_rxz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(helicity_field_ryy_, helicity_field_ryy_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(helicity_field_ryz_, helicity_field_ryz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(helicity_field_rzz_, helicity_field_rzz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
      return;
    }
#endif

    interaction_rij_x_.multiply(globals::s, helicity_field_rxx_);
    interaction_rij_x_.multiply(globals::s, helicity_field_rxx_);

    interaction_rij_x_.multiply(globals::s, helicity_field_rxy_);
    interaction_rij_y_.multiply(globals::s, helicity_field_rxy_);

    interaction_rij_x_.multiply(globals::s, helicity_field_rxz_);
    interaction_rij_z_.multiply(globals::s, helicity_field_rxz_);


    interaction_rij_y_.multiply(globals::s, helicity_field_ryy_);
    interaction_rij_y_.multiply(globals::s, helicity_field_ryy_);

    interaction_rij_y_.multiply(globals::s, helicity_field_ryz_);
    interaction_rij_z_.multiply(globals::s, helicity_field_ryz_);

    interaction_rij_z_.multiply(globals::s, helicity_field_rzz_);
    interaction_rij_z_.multiply(globals::s, helicity_field_rzz_);

    interaction_Jij_.multiply(helicity_field_rxx_, helicity_field_rxx_);
    interaction_Jij_.multiply(helicity_field_rxy_, helicity_field_rxy_);
    interaction_Jij_.multiply(helicity_field_rxz_, helicity_field_rxz_);
    interaction_Jij_.multiply(helicity_field_ryy_, helicity_field_ryy_);
    interaction_Jij_.multiply(helicity_field_ryz_, helicity_field_ryz_);
    interaction_Jij_.multiply(helicity_field_rzz_, helicity_field_rzz_);

}

void HelicityModulusMonitor::calculate_entropy_fields() {
#if HAS_CUDA
    if (jams::instance().mode() == jams::Mode::GPU) {
        interaction_rij_x_.multiply_gpu(globals::s, entropy_field_x_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_y_.multiply_gpu(globals::s, entropy_field_y_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_rij_z_.multiply_gpu(globals::s, entropy_field_z_, jams::instance().cusparse_handle(), cusparse_stream_.get());

        interaction_Jij_.multiply_gpu(entropy_field_x_, entropy_field_x_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(entropy_field_y_, entropy_field_y_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_Jij_.multiply_gpu(entropy_field_z_, entropy_field_z_, jams::instance().cusparse_handle(), cusparse_stream_.get());
      return;
    }
#endif
    interaction_rij_x_.multiply(globals::s, entropy_field_x_);
    interaction_rij_y_.multiply(globals::s, entropy_field_y_);
    interaction_rij_z_.multiply(globals::s, entropy_field_z_);

    interaction_Jij_.multiply(entropy_field_x_, entropy_field_x_);
    interaction_Jij_.multiply(entropy_field_y_, entropy_field_y_);
    interaction_Jij_.multiply(entropy_field_z_, entropy_field_z_);
}

Mat3 HelicityModulusMonitor::exchange_total_internal_energy_difference() {
    using namespace globals;
    Mat3 dU = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (auto i = 0; i < globals::num_spins; ++i) {

        Vec3 s_i = {s(i,0), s(i,1), s(i,2)};

        // TO DO: ONLY CALCULATE UPPER TRIANGLE OF THE STIFFNESS TENSOR

        // COULD ADD ACCURACY IF CALCULATE THE MEAN OF THE UPPER AND LOWER
        // TRIANGLE

        Vec3 h_i_xx = {helicity_field_rxx_(i,0), helicity_field_rxx_(i, 1), helicity_field_rxx_(i, 2)};
        Vec3 h_i_xy = {helicity_field_rxy_(i,0), helicity_field_rxy_(i, 1), helicity_field_rxy_(i, 2)};
        Vec3 h_i_xz = {helicity_field_rxz_(i,0), helicity_field_rxz_(i, 1), helicity_field_rxz_(i, 2)};

        Vec3 h_i_yx = h_i_xy;
        Vec3 h_i_yy = {helicity_field_ryy_(i,0), helicity_field_ryy_(i, 1), helicity_field_ryy_(i, 2)};
        Vec3 h_i_yz = {helicity_field_ryz_(i,0), helicity_field_ryz_(i, 1), helicity_field_ryz_(i, 2)};

        Vec3 h_i_zx = h_i_xz;
        Vec3 h_i_zy = h_i_yz;
        Vec3 h_i_zz = {helicity_field_rzz_(i,0), helicity_field_rzz_(i, 1), helicity_field_rzz_(i, 2)};

        dU[0][0] += dot_sq(s_i, h_i_xx);
        dU[1][1] += dot_sq(s_i, h_i_yy);
        dU[2][2] += dot_sq(s_i, h_i_zz);

        dU[0][1] += dot_sq(s_i, h_i_xy);
        dU[0][2] += dot_sq(s_i, h_i_xz);
        dU[1][2] += dot_sq(s_i, h_i_yz);

        dU[1][0] += dot_sq(s_i, h_i_yx);
        dU[2][0] += dot_sq(s_i, h_i_zx);
        dU[2][1] += dot_sq(s_i, h_i_zy);

    }

    return dU;
}

Mat3 HelicityModulusMonitor::exchange_total_entropy() {
    using namespace globals;
    Mat3 TS = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (auto i = 0; i < globals::num_spins; ++i) {

        Vec3 s_i = {s(i,0), s(i,1), s(i,2)};

        Vec3 h_i_x = {entropy_field_x_(i,0), entropy_field_x_(i, 1), entropy_field_x_(i, 2)};
        Vec3 h_i_y = {entropy_field_y_(i,0), entropy_field_y_(i, 1), entropy_field_y_(i, 2)};
        Vec3 h_i_z = {entropy_field_z_(i,0), entropy_field_z_(i, 1), entropy_field_z_(i, 2)};

        Vec3 Jij_Rij_x = cross(s_i, h_i_x);
        Vec3 Jij_Rij_y = cross(s_i, h_i_y);
        Vec3 Jij_Rij_z = cross(s_i, h_i_z);

        // TO DO: ONLY CALCULATE UPPER TRIANGLE OF THE STIFFNESS TENSOR

        // COULD ADD ACCURACY IF CALCULATE THE MEAN OF THE UPPER AND LOWER
        // TRIANGLE

        TS[0][0] += dot(Jij_Rij_x, Jij_Rij_x);
        TS[1][1] += dot(Jij_Rij_y, Jij_Rij_y);
        TS[2][2] += dot(Jij_Rij_z, Jij_Rij_z);

        TS[0][1] += dot(Jij_Rij_x, Jij_Rij_y);
        TS[0][2] += dot(Jij_Rij_x, Jij_Rij_z);

        TS[1][0] += dot(Jij_Rij_x, Jij_Rij_y);
        TS[2][0] += dot(Jij_Rij_x, Jij_Rij_z);

        TS[2][1] += dot(Jij_Rij_z, Jij_Rij_y);
        TS[1][2] += dot(Jij_Rij_y, Jij_Rij_z);
    }

    return TS;
}
