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

    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JRR_xx(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JRR_xy(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JRR_xz(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JRR_yy(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JRR_yz(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JRR_zz(3 *globals::num_spins, 3 * globals::num_spins);

    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JR_x(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JR_y(3 *globals::num_spins, 3 * globals::num_spins);
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_JR_z(3 *globals::num_spins, 3 * globals::num_spins);

    const auto& input_energy_conversion = exchange_hamiltonian.get_input_conversion();
    const auto& nbr_list = exchange_hamiltonian.neighbour_list();
    for (auto n = 0; n < nbr_list.size(); ++n) {
        auto i = nbr_list[n].first[0];
        auto j = nbr_list[n].first[1];
        auto Jij = input_energy_conversion * nbr_list[n].second;
        auto rij = lattice->displacement(i, j);

        Mat3 JijRij_x = rij[0] * Jij;
        Mat3 JijRij_y = rij[1] * Jij;
        Mat3 JijRij_z = rij[2] * Jij;

        Mat3 JijRijRij_xx = rij[0]*rij[0] * Jij;
        Mat3 JijRijRij_xy = rij[0]*rij[1] * Jij;
        Mat3 JijRijRij_xz = rij[0]*rij[2] * Jij;
        Mat3 JijRijRij_yy = rij[1]*rij[1] * Jij;
        Mat3 JijRijRij_yz = rij[1]*rij[2] * Jij;
        Mat3 JijRijRij_zz = rij[2]*rij[2] * Jij;

        for (auto m = 0; m < 3; ++m) {
            for (auto l = 0; l < 3; ++l) {
                if (JijRijRij_xx[m][l] != 0.0) {
                    sparse_matrix_builder_JRR_xx.insert(3 * i + m, 3 * j + l, JijRijRij_xx[m][l]);
                }
                if (JijRijRij_xy[m][l] != 0.0) {
                    sparse_matrix_builder_JRR_xy.insert(3 * i + m, 3 * j + l, JijRijRij_xy[m][l]);
                }
                if (JijRijRij_xz[m][l] != 0.0) {
                    sparse_matrix_builder_JRR_xz.insert(3 * i + m, 3 * j + l, JijRijRij_xz[m][l]);
                }
                if (JijRijRij_yy[m][l] != 0.0) {
                    sparse_matrix_builder_JRR_yy.insert(3 * i + m, 3 * j + l, JijRijRij_yy[m][l]);
                }
                if (JijRijRij_yz[m][l] != 0.0) {
                    sparse_matrix_builder_JRR_yz.insert(3 * i + m, 3 * j + l, JijRijRij_yz[m][l]);
                }
                if (JijRijRij_zz[m][l] != 0.0) {
                    sparse_matrix_builder_JRR_zz.insert(3 * i + m, 3 * j + l, JijRijRij_zz[m][l]);
                }

                if (JijRij_x[m][l] != 0.0) {
                    sparse_matrix_builder_JR_x.insert(3 * i + m, 3 * j + l, JijRij_x[m][l]);
                }
                if (JijRij_y[m][l] != 0.0) {
                    sparse_matrix_builder_JR_y.insert(3 * i + m, 3 * j + l, JijRij_y[m][l]);
                }
                if (JijRij_z[m][l] != 0.0) {
                    sparse_matrix_builder_JR_z.insert(3 * i + m, 3 * j + l, JijRij_z[m][l]);
                }
            }
        }
    }
    cout << "    dipole sparse matrix builder memory " << sparse_matrix_builder_JRR_xx.memory() / kBytesToMegaBytes << "(MB)\n";
    cout << "    building CSR matrix\n";
    interaction_JRR_xx_ = sparse_matrix_builder_JRR_xx
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JRR_xy_ = sparse_matrix_builder_JRR_xy
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JRR_xz_ = sparse_matrix_builder_JRR_xz
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JRR_yy_ = sparse_matrix_builder_JRR_yy
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JRR_yz_ = sparse_matrix_builder_JRR_yz
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JRR_zz_ = sparse_matrix_builder_JRR_zz
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JR_x_ = sparse_matrix_builder_JR_x
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JR_y_ = sparse_matrix_builder_JR_y
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    interaction_JR_z_ = sparse_matrix_builder_JR_z
            .set_format(jams::SparseMatrixFormat::CSR)
            .build();
    cout << "    exchange sparse matrix memory (CSR): " << sparse_matrix_builder_JRR_xx.memory() / kBytesToMegaBytes << " (MB)\n";


    helicity_field_rxrx_.resize(globals::num_spins, 3);
    helicity_field_rxry_.resize(globals::num_spins, 3);
    helicity_field_rxrz_.resize(globals::num_spins, 3);
    helicity_field_ryry_.resize(globals::num_spins, 3);
    helicity_field_ryrz_.resize(globals::num_spins, 3);
    helicity_field_rzrz_.resize(globals::num_spins, 3);

    entropy_field_rx_.resize(globals::num_spins, 3);
    entropy_field_ry_.resize(globals::num_spins, 3);
    entropy_field_rz_.resize(globals::num_spins, 3);

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

            entropy *= -beta;

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

            entropy *= -beta;

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
            interaction_JRR_xx_.multiply_gpu(globals::s, helicity_field_rxrx_, jams::instance().cusparse_handle(), cusparse_stream_.get());
            interaction_JRR_xy_.multiply_gpu(globals::s, helicity_field_rxry_, jams::instance().cusparse_handle(), cusparse_stream_.get());
            interaction_JRR_xz_.multiply_gpu(globals::s, helicity_field_rxrz_, jams::instance().cusparse_handle(), cusparse_stream_.get());

            interaction_JRR_yy_.multiply_gpu(globals::s, helicity_field_ryry_, jams::instance().cusparse_handle(), cusparse_stream_.get());
            interaction_JRR_yz_.multiply_gpu(globals::s, helicity_field_ryrz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
            interaction_JRR_zz_.multiply_gpu(globals::s, helicity_field_rzrz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        return;
    }
#endif

    interaction_JRR_xx_.multiply(globals::s, helicity_field_rxrx_);
    interaction_JRR_xy_.multiply(globals::s, helicity_field_rxry_);
    interaction_JRR_xz_.multiply(globals::s, helicity_field_rxrz_);

    interaction_JRR_yy_.multiply(globals::s, helicity_field_ryry_);
    interaction_JRR_yz_.multiply(globals::s, helicity_field_ryrz_);
    interaction_JRR_zz_.multiply(globals::s, helicity_field_rzrz_);

}

void HelicityModulusMonitor::calculate_entropy_fields() {
#if HAS_CUDA
    if (jams::instance().mode() == jams::Mode::GPU) {
        interaction_JR_x_.multiply_gpu(globals::s, entropy_field_rx_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_JR_y_.multiply_gpu(globals::s, entropy_field_ry_, jams::instance().cusparse_handle(), cusparse_stream_.get());
        interaction_JR_z_.multiply_gpu(globals::s, entropy_field_rz_, jams::instance().cusparse_handle(), cusparse_stream_.get());
      return;
    }
#endif

    interaction_JR_x_.multiply(globals::s, entropy_field_rx_);
    interaction_JR_y_.multiply(globals::s, entropy_field_ry_);
    interaction_JR_z_.multiply(globals::s, entropy_field_rz_);
}

Mat3 HelicityModulusMonitor::exchange_total_internal_energy_difference() {
    using namespace globals;
    Mat3 dU = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (auto i = 0; i < globals::num_spins; ++i) {

        Vec3 s_i = {s(i,0), s(i,1), s(i,2)};

        // TO DO: ONLY CALCULATE UPPER TRIANGLE OF THE STIFFNESS TENSOR

        // COULD ADD ACCURACY IF CALCULATE THE MEAN OF THE UPPER AND LOWER
        // TRIANGLE

        Vec3 h_i_xx = {helicity_field_rxrx_(i,0), helicity_field_rxrx_(i, 1), helicity_field_rxrx_(i, 2)};
        Vec3 h_i_xy = {helicity_field_rxry_(i,0), helicity_field_rxry_(i, 1), helicity_field_rxry_(i, 2)};
        Vec3 h_i_xz = {helicity_field_rxrz_(i,0), helicity_field_rxrz_(i, 1), helicity_field_rxrz_(i, 2)};

        Vec3 h_i_yx = h_i_xy;
        Vec3 h_i_yy = {helicity_field_ryry_(i,0), helicity_field_ryry_(i, 1), helicity_field_ryry_(i, 2)};
        Vec3 h_i_yz = {helicity_field_ryrz_(i,0), helicity_field_ryrz_(i, 1), helicity_field_ryrz_(i, 2)};

        Vec3 h_i_zx = h_i_xz;
        Vec3 h_i_zy = h_i_yz;
        Vec3 h_i_zz = {helicity_field_rzrz_(i,0), helicity_field_rzrz_(i, 1), helicity_field_rzrz_(i, 2)};

        dU[0][0] += dot(s_i, h_i_xx);
        dU[1][1] += dot(s_i, h_i_yy);
        dU[2][2] += dot(s_i, h_i_zz);

        dU[0][1] += dot(s_i, h_i_xy);
        dU[0][2] += dot(s_i, h_i_xz);
        dU[1][2] += dot(s_i, h_i_yz);

        dU[1][0] += dot(s_i, h_i_yx);
        dU[2][0] += dot(s_i, h_i_zx);
        dU[2][1] += dot(s_i, h_i_zy);

    }

    return dU;
}

Mat3 HelicityModulusMonitor::exchange_total_entropy() {
    using namespace globals;
    Mat3 TS = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Vec3 Jij_Rij_x = {0.0, 0.0, 0.0};
    Vec3 Jij_Rij_y = {0.0, 0.0, 0.0};
    Vec3 Jij_Rij_z = {0.0, 0.0, 0.0};

    for (auto i = 0; i < globals::num_spins; ++i) {

        Vec3 s_i = {s(i,0), s(i,1), s(i,2)};

        Vec3 h_i_x = {entropy_field_rx_(i,0), entropy_field_rx_(i, 1), entropy_field_rx_(i, 2)};
        Vec3 h_i_y = {entropy_field_ry_(i,0), entropy_field_ry_(i, 1), entropy_field_ry_(i, 2)};
        Vec3 h_i_z = {entropy_field_rz_(i,0), entropy_field_rz_(i, 1), entropy_field_rz_(i, 2)};

        Jij_Rij_x = cross(s_i, h_i_x); // = \sum_j r_ij^x J_ij S_i x S_j
        Jij_Rij_y = cross(s_i, h_i_y); // = \sum_j r_ij^y J_ij S_i x S_j
        Jij_Rij_z = cross(s_i, h_i_z); // = \sum_j r_ij^z J_ij S_i x S_j

        TS[0][0] += dot(Jij_Rij_x, Jij_Rij_x);
        TS[1][1] += dot(Jij_Rij_y, Jij_Rij_y);
        TS[2][2] += dot(Jij_Rij_z, Jij_Rij_z);

        TS[0][1] += dot(Jij_Rij_x, Jij_Rij_y);
        TS[0][2] += dot(Jij_Rij_x, Jij_Rij_z);

        TS[1][0] += dot(Jij_Rij_y, Jij_Rij_x);
        TS[2][0] += dot(Jij_Rij_z, Jij_Rij_x);

        TS[2][1] += dot(Jij_Rij_z, Jij_Rij_y);
        TS[1][2] += dot(Jij_Rij_y, Jij_Rij_z);
    }

    return TS;
}
