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
    int double_spin = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
        double_spin = static_cast<int> 2* std::round(globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU));
    // This Hamiltonian is for spin {1/2, 1, 3/2, 2}, so our mus should be g*{1/2, 1, 3/2, 2} bohr magnetons
        if (!(double_spin==1 or double_spin==2 or double_spin==3 or double_spin==4)) {
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
    select_kernel(double_spin);

    finalize(sparse_matrix_symmetry_check);
}

void CudaPisdExchangeHamiltonian::select_kernel(double_spin) {
    switch (double_spin) {
        case 1:
            kernel_launcher = cuda_pisd_exchange_field_kernel_spin_one_half;
            break;

        case 2:
            kernel_launcher = cuda_pisd_exchange_field_kernel_spin_one;
            break;

        case 3:
            kernel_launcher = cuda_pisd_exchange_field_kernel_spin_three_half;
            break;

        case 4:
            kernel_launcher = cuda_pisd_exchange_field_kernel_spin_two;
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
  switch(rint(2*globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU))) {
      case 1:
          kernel_launcher<<<grid_size, block_size>>>
              (bz_field_, beta,
               globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
               interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
               field_.device_data());
      case 2:
          cuda_pisd_exchange_field_kernel_spin_one<<<grid_size, block_size>>>
                  (bz_field_, beta,
                   globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
                   interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
                   field_.device_data());
      case 3:
          cuda_pisd_exchange_field_kernel_spin_three_half<<<grid_size, block_size>>>
                  (bz_field_, beta,
                   globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
                   interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
                   field_.device_data());
      case 4:
          cuda_pisd_exchange_field_kernel_spin_two<<<grid_size, block_size>>>
                  (bz_field_, beta,
                   globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
                   interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
                   field_.device_data());
      default:
          std::cerr << "Unsupported spin value: " << rint(2*globals::mus(i) / (kElectronGFactor * kBohrMagnetonIU)) << std::endl;
          throw std::runtime_error("Unknown spin value encountered in kernel launcher.");
  }
    DEBUG_CHECK_CUDA_ASYNC_STATUS

}

Vec3 CudaPisdExchangeHamiltonian::calculate_field(int i, double time) {
  return Vec3({0, 0, 0});
}

