//
// Created by Joe Barker on 2018/04/20.
//

#include <jams/helpers/exception.h>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include <jams/common.h>
#include <jams/helpers/error.h>
#include <jams/helpers/consts.h>
#include <jams/helpers/utils.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/helpers/output.h"
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/cuda_dipole_fft.h"
#include "jams/interface/config.h"
#include "jams/monitors/cuda_thermal_current.h"
#include "jams/cuda/cuda_common.h"
#include "cuda_thermal_current.h"

namespace {

std::vector<std::string> excluded_hamiltonian_modules_from(
    const libconfig::Setting& settings) {
  std::vector<std::string> excluded_modules;
  if (!settings.exists("exclude_hamiltonians")) {
    return excluded_modules;
  }

  const auto& excluded_setting = settings["exclude_hamiltonians"];
  if (!jams::is_sequence_setting(excluded_setting)) {
    throw jams::ConfigException(
        excluded_setting,
        "exclude_hamiltonians must be an array or list of module names");
  }

  excluded_modules.reserve(excluded_setting.getLength());
  for (auto i = 0; i < excluded_setting.getLength(); ++i) {
    if (!jams::is_string_setting(excluded_setting[i])) {
      throw jams::ConfigException(
          excluded_setting[i],
          "excluded Hamiltonian module name must be a string");
    }
    excluded_modules.push_back(lowercase(std::string(excluded_setting[i].c_str())));
  }

  return excluded_modules;
}

bool is_excluded_hamiltonian(
    const Hamiltonian& hamiltonian,
    const std::vector<std::string>& excluded_modules) {
  const auto module_name = lowercase(hamiltonian.module_name());
  return std::find(excluded_modules.begin(), excluded_modules.end(), module_name)
      != excluded_modules.end();
}

class SparseEnergyCurrentInteractionSink final : public jams::EnergyCurrentInteractionSink {
public:
  SparseEnergyCurrentInteractionSink(jams::SparseMatrix<double>::Builder& rx_builder,
                                     jams::SparseMatrix<double>::Builder& ry_builder,
                                     jams::SparseMatrix<double>::Builder& rz_builder)
      : rx_builder_(rx_builder),
        ry_builder_(ry_builder),
        rz_builder_(rz_builder) {
  }

  void insert(const int site_i,
              const int site_j,
              const jams::Vec<double, 3>& r_ji,
              const jams::Mat<double, 3, 3>& interaction) override {
    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        if (interaction[m][n] == 0.0) {
          continue;
        }

        const int row = 3 * site_i + m;
        const int col = 3 * site_j + n;
        if (r_ji[0] != 0.0) {
          rx_builder_.insert(row, col, r_ji[0] * interaction[m][n]);
        }
        if (r_ji[1] != 0.0) {
          ry_builder_.insert(row, col, r_ji[1] * interaction[m][n]);
        }
        if (r_ji[2] != 0.0) {
          rz_builder_.insert(row, col, r_ji[2] * interaction[m][n]);
        }
      }
    }
  }

private:
  jams::SparseMatrix<double>::Builder& rx_builder_;
  jams::SparseMatrix<double>::Builder& ry_builder_;
  jams::SparseMatrix<double>::Builder& rz_builder_;
};

}  // namespace

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

  const auto excluded_hamiltonian_modules = excluded_hamiltonian_modules_from(settings);

  for (const auto& hamiltonian : globals::solver->hamiltonians()) {
    if (is_excluded_hamiltonian(*hamiltonian, excluded_hamiltonian_modules)) {
      std::cout << "    excluding Hamiltonian '" << hamiltonian->module_name()
                << "' from thermal-current transport operator\n";
      continue;
    }

    if (auto* dipole_fft = dynamic_cast<CudaDipoleFFTHamiltonian*>(hamiltonian.get())) {
      dipole_fft->ensure_energy_current_tensors();
      dipole_fft_energy_current_providers_.push_back(dipole_fft);
      continue;
    }

    switch (hamiltonian->energy_current_interaction_support()) {
      case Hamiltonian::EnergyCurrentInteractionSupport::None:
        continue;
      case Hamiltonian::EnergyCurrentInteractionSupport::Supported:
      {
        SparseEnergyCurrentInteractionSink sink(
            energy_current_operator_builders[0],
            energy_current_operator_builders[1],
            energy_current_operator_builders[2]);
        hamiltonian->add_energy_current_interactions(sink);
        break;
      }
      case Hamiltonian::EnergyCurrentInteractionSupport::Unsupported:
        throw std::runtime_error(
            "thermal-current monitor does not support energy-current interactions for Hamiltonian '"
            + hamiltonian->name() + "' (module '" + hamiltonian->module_name() + "')");
    }
  }

  energy_current_operator_rx_ = energy_current_operator_builders[0].build();
  energy_current_operator_ry_ = energy_current_operator_builders[1].build();
  energy_current_operator_rz_ = energy_current_operator_builders[2].build();
  has_sparse_energy_current_operator_ =
      energy_current_operator_rx_.num_non_zero() > 0
      || energy_current_operator_ry_.num_non_zero() > 0
      || energy_current_operator_rz_.num_non_zero() > 0;

  const double volume_lattice_units = volume(globals::lattice->get_supercell());
  if (volume_lattice_units <= 0.0) {
    throw std::runtime_error("thermal-current monitor requires a positive simulation volume");
  }

  const double lattice_parameter_nm = globals::lattice->parameter() * kMeterToNanometer;
  if (lattice_parameter_nm <= 0.0) {
    throw std::runtime_error("thermal-current monitor requires a positive lattice parameter");
  }

  current_density_prefactor_ =
      -0.5 / (volume_lattice_units * lattice_parameter_nm * lattice_parameter_nm);

  std::cout << "    energy current operator rx non-zero: " << energy_current_operator_rx_.num_non_zero() << "\n";
  std::cout << "    energy current operator ry non-zero: " << energy_current_operator_ry_.num_non_zero() << "\n";
  std::cout << "    energy current operator rz non-zero: " << energy_current_operator_rz_.num_non_zero() << "\n";
  std::cout << "    energy current operator memory: "
            << (energy_current_operator_rx_.memory()
                + energy_current_operator_ry_.memory()
                + energy_current_operator_rz_.memory()) / kBytesToMegaBytes
            << " MB\n";
  std::size_t dipole_fft_energy_current_tensor_memory = 0;
  for (const auto* provider : dipole_fft_energy_current_providers_) {
    dipole_fft_energy_current_tensor_memory += provider->energy_current_tensor_memory();
  }
  if (!dipole_fft_energy_current_providers_.empty()) {
    std::cout << "    dipole FFT energy current tensor memory: "
              << dipole_fft_energy_current_tensor_memory / kBytesToMegaBytes
              << " MB\n";
  }

  zero(spin_derivative_.resize(globals::num_spins, 3));
  zero(energy_current_rx_.resize(globals::num_spins, 3));
  zero(energy_current_ry_.resize(globals::num_spins, 3));
  zero(energy_current_rz_.resize(globals::num_spins, 3));
  zero(energy_current_dot_.resize(globals::num_spins));
  if (!dipole_fft_energy_current_providers_.empty()) {
    zero(dipole_fft_energy_current_rx_.resize(globals::num_spins, 3));
    zero(dipole_fft_energy_current_ry_.resize(globals::num_spins, 3));
    zero(dipole_fft_energy_current_rz_.resize(globals::num_spins, 3));
  }

  auto cols = globals::solver->monitor_coordinate_columns();
  cols.push_back({"jE_rx", "meV ps^-1 nm^-2"});
  cols.push_back({"jE_ry", "meV ps^-1 nm^-2"});
  cols.push_back({"jE_rz", "meV ps^-1 nm^-2"});
  tsv_.open(jams::output::monitor_filename(name(), "tsv"), std::move(cols));
}

void CudaThermalCurrentMonitor::update(Solver& solver) {
  solver.compute_fields();
  CHECK_CUDA_STATUS(cudaStreamSynchronize(jams::instance().cuda_master_stream().get()));

  const auto& spins = globals::s;
  const auto& field = globals::h;
  const auto& gyro = globals::gyro;
  const auto& inv_mus = globals::inv_mus;
  jams::Vec<double, 3> jE = execute_cuda_thermal_current_kernel(
      stream,
      spins,
      field,
      gyro,
      inv_mus,
      energy_current_operator_rx_,
      energy_current_operator_ry_,
      energy_current_operator_rz_,
      current_density_prefactor_,
      has_sparse_energy_current_operator_,
      spin_derivative_,
      energy_current_rx_,
      energy_current_ry_,
      energy_current_rz_,
      energy_current_dot_);

  const auto& field_spins = solver.spin_array_for_fields();
  for (auto* provider : dipole_fft_energy_current_providers_) {
    provider->calculate_energy_current_fields(
        field_spins,
        dipole_fft_energy_current_rx_,
        dipole_fft_energy_current_ry_,
        dipole_fft_energy_current_rz_);
    provider->wait_on(stream.get());

    const auto dipole_jE = execute_cuda_thermal_current_field_reduction(
        stream,
        current_density_prefactor_,
        spin_derivative_,
        dipole_fft_energy_current_rx_,
        dipole_fft_energy_current_ry_,
        dipole_fft_energy_current_rz_,
        energy_current_dot_);
    jE[0] += dipole_jE[0];
    jE[1] += dipole_jE[1];
    jE[2] += dipole_jE[2];
  }

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
