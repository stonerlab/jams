//
// Created by Joe Barker on 2018/04/20.
//

#include <jams/helpers/exception.h>
#include <array>
#include <utility>
#include <vector>

#include <jams/common.h>
#include <jams/helpers/error.h>
#include <jams/helpers/consts.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/helpers/output.h"
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/monitors/cuda_thermal_current.h"
#include "jams/cuda/cuda_common.h"
#include "cuda_thermal_current.h"

CudaThermalCurrentMonitor::CudaThermalCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {
  if (jams::instance().mode() != jams::Mode::GPU) {
    throw std::runtime_error("thermal-current monitor requires GPU mode");
  }

  std::array<jams::SparseMatrix<double>::Builder, 3> energy_current_operator_builders = {
      jams::SparseMatrix<double>::Builder(globals::num_spins3, globals::num_spins3),
      jams::SparseMatrix<double>::Builder(globals::num_spins3, globals::num_spins3),
      jams::SparseMatrix<double>::Builder(globals::num_spins3, globals::num_spins3)
  };

  for (auto& builder : energy_current_operator_builders) {
    builder.set_format(jams::SparseMatrixFormat::CSR);
  }

  for (const auto& hamiltonian : globals::solver->hamiltonians()) {
    switch (hamiltonian->energy_current_interaction_support()) {
      case Hamiltonian::EnergyCurrentInteractionSupport::None:
        continue;
      case Hamiltonian::EnergyCurrentInteractionSupport::Supported:
        hamiltonian->add_energy_current_interactions(
            energy_current_operator_builders[0],
            energy_current_operator_builders[1],
            energy_current_operator_builders[2]);
        break;
      case Hamiltonian::EnergyCurrentInteractionSupport::Unsupported:
        throw std::runtime_error(
            "thermal-current monitor does not support energy-current interactions for Hamiltonian '"
            + hamiltonian->name() + "' (module '" + hamiltonian->module_name() + "')");
    }
  }

  energy_current_operator_rx_ = energy_current_operator_builders[0].build();
  energy_current_operator_ry_ = energy_current_operator_builders[1].build();
  energy_current_operator_rz_ = energy_current_operator_builders[2].build();

  volume_ = volume(globals::lattice->get_supercell());
  if (volume_ <= 0.0) {
    throw std::runtime_error("thermal-current monitor requires a positive simulation volume");
  }

  std::cout << "    energy current operator rx non-zero: " << energy_current_operator_rx_.num_non_zero() << "\n";
  std::cout << "    energy current operator ry non-zero: " << energy_current_operator_ry_.num_non_zero() << "\n";
  std::cout << "    energy current operator rz non-zero: " << energy_current_operator_rz_.num_non_zero() << "\n";
  std::cout << "    energy current operator memory: "
            << (energy_current_operator_rx_.memory()
                + energy_current_operator_ry_.memory()
                + energy_current_operator_rz_.memory()) / kBytesToMegaBytes
            << " MB\n";

  zero(spin_derivative_.resize(globals::num_spins, 3));
  zero(energy_current_rx_.resize(globals::num_spins, 3));
  zero(energy_current_ry_.resize(globals::num_spins, 3));
  zero(energy_current_rz_.resize(globals::num_spins, 3));
  zero(energy_current_dot_.resize(globals::num_spins));

  auto cols = globals::solver->monitor_coordinate_columns();
  cols.push_back({"jE_rx", "internal"});
  cols.push_back({"jE_ry", "internal"});
  cols.push_back({"jE_rz", "internal"});
  tsv_.open(jams::output::monitor_filename(name(), "tsv"), std::move(cols));
}

void CudaThermalCurrentMonitor::update(Solver& solver) {
  solver.compute_fields();
  CHECK_CUDA_STATUS(cudaStreamSynchronize(jams::instance().cuda_master_stream().get()));

  const auto& spins = globals::s;
  const auto& field = globals::h;
  const auto& gyro = globals::gyro;
  const auto& mus = globals::mus;
  jams::Vec<double, 3> jE = execute_cuda_thermal_current_kernel(
      stream,
      spins,
      field,
      gyro,
      mus,
      energy_current_operator_rx_,
      energy_current_operator_ry_,
      energy_current_operator_rz_,
      volume_,
      spin_derivative_,
      energy_current_rx_,
      energy_current_ry_,
      energy_current_rz_,
      energy_current_dot_);

  std::vector<double> values;
  values.reserve(tsv_.num_cols());
  solver.append_monitor_coordinates(values);
  values.push_back(jE[0]);
  values.push_back(jE[1]);
  values.push_back(jE[2]);
  tsv_.write_row(values);
}

CudaThermalCurrentMonitor::~CudaThermalCurrentMonitor() {
}
