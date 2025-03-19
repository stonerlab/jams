#include <fstream>
#include <jams/hamiltonian/cuda_pisd_exchange_kernel.cuh>

#include <jams/core/globals.h>
#include <jams/core/interactions.h>
#include <jams/core/solver.h>
#include <jams/hamiltonian/cuda_pisd_exchange.h>
#include <jams/helpers/output.h>

CudaPisdExchangeHamiltonian::CudaPisdExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size)
: NeighbourListInteractionHamiltonian(settings, size),
  bz_field_(jams::config_required<double>(settings, "bz_field"))
{

    for (auto i = 0; i < globals::num_spins; ++i) {
    // This Hamiltonian is for spin {1/2, 1, 3/2, 2}, so our mus should be g*{1/2, 1, 3/2, 2} bohr magnetons
        if (!(approximately_equal(globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU), 0.5, 1e-5) or
               approximately_equal(globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU), 1.0, 1e-5) or
               approximately_equal(globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU), 1.5, 1e-5) or
               approximately_equal(globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU), 2.0, 1e-5)
        )) {
            std::cout << globals::mus(i) << ", " << globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU) << std::endl;
            throw std::runtime_error("The pisd-exchange hamiltonian is only for S={1/2, 1, 3/2, 2} systems");
        }
    }

    auto sparse_matrix_symmetry_check = read_sparse_matrix_symmetry_check_from_settings(settings, jams::SparseMatrixSymmetryCheck::Symmetric);

    neighbour_list_ = create_neighbour_list_from_settings(settings);

    print_neighbour_list_info(std::cout, neighbour_list_);

    if (debug_is_enabled()) {
        write_neighbour_list(jams::output::full_path_ofstream("DEBUG_pisd_exchange_nbr_list.tsv"), neighbour_list_);
    }

    for (auto n = 0; n < neighbour_list_.size(); ++n) {
        auto i = neighbour_list_[n].first[0];
        auto j = neighbour_list_[n].first[1];
        auto value = input_energy_unit_conversion_ * neighbour_list_[n].second[0][0];

        insert_interaction_scalar(i, j, value);
    }
    select_kernel(static_cast<int>(std::round(2*globals::mus(0) / (kElectronGFactor * kBohrMagnetonIU))));

    finalize(sparse_matrix_symmetry_check);
}

void CudaPisdExchangeHamiltonian::select_kernel(const int double_spin) {
    switch (double_spin) {
        case 1:
            cuda_pisd_exchange_field_kernel = cuda_pisd_exchange_field_kernel_spin_one_half;
            break;
        case 2:
            cuda_pisd_exchange_field_kernel = cuda_pisd_exchange_field_kernel_spin_one;
            break;
        case 3:
            cuda_pisd_exchange_field_kernel = cuda_pisd_exchange_field_kernel_spin_three_half;
            break;
        case 4:
            cuda_pisd_exchange_field_kernel = cuda_pisd_exchange_field_kernel_spin_two;
            break;
        default:
            std::cout << "Unsupported spin value: " << double_spin << std::endl;
            throw std::runtime_error("Unknown spin value encountered in kernel launcher.");
    }
}

void CudaPisdExchangeHamiltonian::calculate_fields(double time) {
  assert(is_finalized_);

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  const double beta = 1.0/(kBoltzmannIU * globals::solver->thermostat()->temperature());

  cuda_pisd_exchange_field_kernel<<<grid_size, block_size>>>
      (bz_field_, beta,
       globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
       interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
       field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

Vec3 CudaPisdExchangeHamiltonian::calculate_field(int i, double time) {
  return Vec3({0, 0, 0});
}

