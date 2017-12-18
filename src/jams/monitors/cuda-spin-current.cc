//
// Created by Joe Barker on 2017/10/04.
//

#include <iomanip>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/monitors/cuda-spin-current.h"

using namespace std;

CudaSpinCurrentMonitor::CudaSpinCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {

//  if (!solver->is_cuda_solver()) {
//    throw std::runtime_error("CUDA spin current monitor is only for CUDA solvers");
//  }

  jams_warning("This monitor currently assumes the exchange interaction is DIAGONAL AND ISOTROPIC");

  std::string exchange_file_name;

  // search for exchange hamiltonian
  InteractionList<Mat3> neighbour_list;

  const libconfig::Setting& hamiltonian_settings = config->lookup("hamiltonians");
  for (int i = 0; i < hamiltonian_settings.getLength(); ++i) {
    std::string module_name = hamiltonian_settings[i]["module"].c_str();
    if (module_name == "exchange") {
      exchange_file_name = hamiltonian_settings[i]["exc_file"].c_str();

      std::ifstream interaction_file(exchange_file_name.c_str());
      if (interaction_file.fail()) {
        throw std::runtime_error("failed to open interaction file:" + exchange_file_name);
      }

      neighbour_list = generate_neighbour_list_from_file(hamiltonian_settings[i], interaction_file);

      break;
    }
  }

  if (exchange_file_name.empty()) {
    throw std::runtime_error("no exchange hamiltonian found");
  }

  SparseMatrix<Vec3> interaction_matrix;

  interaction_matrix.resize(globals::num_spins, globals::num_spins);
  interaction_matrix.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

  for (unsigned i = 0; i < neighbour_list.size(); ++i) {
    for (auto const &nbr: neighbour_list[i]) {
      unsigned j = nbr.first;
      double Jij = nbr.second[0][0];
      if (i > j) continue;
      auto r_i = lattice->atom_position(i);
      auto r_j = lattice->atom_position(j);
      interaction_matrix.insertValue(i, j, lattice->displacement(r_i, r_j) * Jij);
    }
  }

  cout << "  converting interaction matrix format from MAP to CSR\n";
  interaction_matrix.convertMAP2CSR();
  cout << "  exchange matrix memory (CSR): " << interaction_matrix.calculateMemory() << " MB\n";

  cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_matrix_.row, (interaction_matrix.rows()+1)*sizeof(int)));
  cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_matrix_.col, (interaction_matrix.nonZero())*sizeof(int)));


  cuda_api_error_check(cudaMemcpy(dev_csr_matrix_.row, interaction_matrix.rowPtr(),
                                  (interaction_matrix.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

  cuda_api_error_check(cudaMemcpy(dev_csr_matrix_.col, interaction_matrix.colPtr(),
                                  (interaction_matrix.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

  // not sure how Vec3 will copy so lets be safe
  jblib::Array<double, 2> val(interaction_matrix.nonZero(), 3);
  for (unsigned i = 0; i < interaction_matrix.nonZero(); ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      val(i, j) = interaction_matrix.val(i)[j];
    }
  }

  cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_matrix_.val, val.elements()*sizeof(double)));

  cuda_api_error_check(cudaMemcpy(dev_csr_matrix_.val, val.data(),
                                  val.elements()*sizeof(double), cudaMemcpyHostToDevice));

  dev_spin_current_rx_x.resize(globals::num_spins);
  dev_spin_current_rx_y.resize(globals::num_spins);
  dev_spin_current_rx_z.resize(globals::num_spins);

  dev_spin_current_ry_x.resize(globals::num_spins);
  dev_spin_current_ry_y.resize(globals::num_spins);
  dev_spin_current_ry_z.resize(globals::num_spins);

  dev_spin_current_rz_x.resize(globals::num_spins);
  dev_spin_current_rz_y.resize(globals::num_spins);
  dev_spin_current_rz_z.resize(globals::num_spins);

  dev_spin_current_rx_x.zero();
  dev_spin_current_rx_y.zero();
  dev_spin_current_rx_z.zero();

  dev_spin_current_ry_x.zero();
  dev_spin_current_ry_y.zero();
  dev_spin_current_ry_z.zero();

  dev_spin_current_rz_x.zero();
  dev_spin_current_rz_y.zero();
  dev_spin_current_rz_z.zero();

  std::string name = seedname + "_js.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  outfile << std::setw(12) << "time" << "\t";
  outfile << std::setw(12) << "js_rx_x" << "\t";
  outfile << std::setw(12) << "js_rx_y" << "\t";
  outfile << std::setw(12) << "js_rx_z" << "\t";
  outfile << std::setw(12) << "js_ry_x" << "\t";
  outfile << std::setw(12) << "js_ry_y" << "\t";
  outfile << std::setw(12) << "js_ry_z" << "\t";
  outfile << std::setw(12) << "js_rz_x" << "\t";
  outfile << std::setw(12) << "js_rz_y" << "\t";
  outfile << std::setw(12) << "js_rz_z" << std::endl;
}

void CudaSpinCurrentMonitor::update(Solver *solver) {
  Mat3 js = execute_cuda_spin_current_kernel(
          stream,
          globals::num_spins,
          solver->dev_ptr_spin(),
          dev_csr_matrix_.val,
          dev_csr_matrix_.row,
          dev_csr_matrix_.col,
          dev_spin_current_rx_x.data(),
          dev_spin_current_rx_y.data(),
          dev_spin_current_rx_z.data(),
          dev_spin_current_ry_x.data(),
          dev_spin_current_ry_y.data(),
          dev_spin_current_ry_z.data(),
          dev_spin_current_rz_x.data(),
          dev_spin_current_rz_y.data(),
          dev_spin_current_rz_z.data()
  );

//  const double units = (lattice->parameter() * 1e-9) * kBohrMagneton * kGyromagneticRatio;

  outfile << std::setw(4) << std::scientific << solver->time() << "\t";
  for (auto r_m = 0; r_m < 3; ++r_m) {
    for (auto n = 0; n < 3; ++ n) {
      outfile << std::setw(12) << js[r_m][n] << "\t";
    }
  }
  outfile << "\n";
}

bool CudaSpinCurrentMonitor::is_converged() {
  return false;
}

CudaSpinCurrentMonitor::~CudaSpinCurrentMonitor() {
  outfile.close();

  if (dev_csr_matrix_.row) {
    cudaFree(dev_csr_matrix_.row);
    dev_csr_matrix_.row = nullptr;
  }

  if (dev_csr_matrix_.col) {
    cudaFree(dev_csr_matrix_.col);
    dev_csr_matrix_.col = nullptr;
  }

  if (dev_csr_matrix_.val) {
    cudaFree(dev_csr_matrix_.val);
    dev_csr_matrix_.val = nullptr;
  }
}
