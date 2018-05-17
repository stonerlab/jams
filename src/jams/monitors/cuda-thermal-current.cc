//
// Created by Joe Barker on 2018/04/20.
//


#include <jams/helpers/exception.h>
#include <jams/core/interactions.h>
#include <jams/helpers/error.h>
#include <jams/helpers/consts.h>
#include <iomanip>
#include <jams/helpers/maths.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/core/globals.h"
#include "jams/interface/config.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/monitors/cuda-thermal-current.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/containers/csr.h"

namespace {
    // convert a list of triads into a CSR like 3D sparse matrix
    using TriadList = std::vector<Triad<Vec3>>;

    void print_triad_list(const TriadList& triads, std::ostream &stream) {
      for (const auto &x : triads) {
        stream << x.i << "\t" << x.j << "\t" << x.k << "\t" << x.value << "\n";
      }
    }

    void sort_triad_list(TriadList& triads) {
      std::stable_sort(triads.begin(), triads.end(),
              [](const Triad<Vec3> &a, const Triad<Vec3> &b) -> bool {
                  return a.k < b.k;
              });

      std::stable_sort(triads.begin(), triads.end(),
              [](const Triad<Vec3> &a, const Triad<Vec3> &b) -> bool {
                  return a.j < b.j;
              });

      std::stable_sort(triads.begin(), triads.end(),
              [](const Triad<Vec3> &a, const Triad<Vec3> &b) -> bool {
                  return a.i < b.i;
              });
    }

    void convert_triad_list_to_csr_format(TriadList triads, const int num_rows,
                                          jblib::Array<int, 1> &index_pointers,
                                          jblib::Array<int, 2> &index_data,
                                          jblib::Array<double, 2> &value_data)
    {
      index_pointers.resize(num_rows + 1);
      index_data.resize(triads.size(), 2);
      value_data.resize(triads.size(), 3);

      index_pointers.zero();
      index_data.zero();
      value_data.zero();

      sort_triad_list(triads);

      index_pointers[0] = 0;
      unsigned index_counter = 0;
      for (auto i = 0; i < num_rows; ++i) {
        for (auto n = index_pointers[i]; n < triads.size(); ++n) {
          if (triads[n].i != i) {
            break;
          }

          index_data(n, 0) = triads[n].j;
          index_data(n, 1) = triads[n].k;

          for (auto m = 0; m < 3; ++m) {
            value_data(n, m) = triads[n].value[m];
          }
          index_counter++;
        }
        index_pointers[i+1] = index_counter;
      }
      index_pointers[num_rows] = triads.size();
    }
};

CudaThermalCurrentMonitor::CudaThermalCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {
  using namespace std;
  jams_warning("This monitor automatically identifies the FIRST exchange hamiltonian in the config");
  jams_warning("This monitor currently assumes the exchange interaction is DIAGONAL AND ISOTROPIC");

  const auto& exchange_settings = config_find_setting_by_key_value_pair(config->lookup("hamiltonians"), "module", "exchange");

  const std::string exchange_file_name = exchange_settings["exc_file"];
  std::ifstream interaction_file(exchange_file_name);

  if (interaction_file.fail()) {
    throw std::runtime_error("failed to open interaction file:" + exchange_file_name);
  }

  cout << "    interaction file name: " << exchange_file_name << endl;

  const auto neighbour_list = generate_neighbour_list_from_file(exchange_settings, interaction_file);
  const auto triad_list = generate_triads_from_neighbour_list(neighbour_list);

  cout << "    total ijk triads: " << triad_list.size() << endl;

  double memory_estimate = (globals::num_spins + 1) * sizeof(int)
                           + 2 * triad_list.size() * sizeof(int)
                           + 3 * triad_list.size() * sizeof(double);

  cout << "    triad matrix memory (CSR): " << memory_estimate / (1024 * 1024) << "MB" << endl;

  cout << "  initializing CUDA device data" << endl;

  initialize_device_data(triad_list);

  outfile.open(seedname + "_jq.tsv");
  outfile.setf(std::ios::right);
  outfile << std::setw(12) << "time" << "\t";
  outfile << std::setw(12) << "jq_rx" << "\t";
  outfile << std::setw(12) << "jq_ry" << "\t";
  outfile << std::setw(12) << "jq_rz" << endl;
}

void CudaThermalCurrentMonitor::update(Solver *solver) {
  Vec3 js = execute_cuda_thermal_current_kernel(
          stream,
          globals::num_spins,
          solver->dev_ptr_spin(),
          dev_csr_matrix_.row,
          dev_csr_matrix_.col,
          dev_csr_matrix_.val,
          dev_thermal_current_rx.data(),
          dev_thermal_current_ry.data(),
          dev_thermal_current_rz.data()
  );

  outfile << std::setw(4) << std::scientific << solver->time() << "\t";
  for (auto n = 0; n < 3; ++ n) {
    outfile << std::setw(12) << js[n] << "\t";
  }
  outfile << std::endl;
}

bool CudaThermalCurrentMonitor::is_converged() {
  return false;
}

CudaThermalCurrentMonitor::~CudaThermalCurrentMonitor() {
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

CudaThermalCurrentMonitor::TriadList CudaThermalCurrentMonitor::generate_triads_from_neighbour_list(const InteractionList<Mat3>& nbr_list) {
  TriadList triads;

  // Jij * Jjk
  for (auto i = 0; i < nbr_list.size(); ++i) {
    // ij
    for (auto const &nbr_j: nbr_list[i]) {
      const int j = nbr_j.first;
      const auto Jij = nbr_j.second[0][0];
      // jk
      for (auto const &nbr_k: nbr_list[j]) {
        const int k = nbr_k.first;
        const auto Jjk = nbr_k.second[0][0];
        if (i == j || j == k || i == k) continue;
        if (i > j || j > k || i > k) continue;
        auto r_i = lattice->atom_position(i);
        auto r_j = lattice->atom_position(j);
        auto r_k = lattice->atom_position(k);
        triads.push_back({i, j , k, Jij * Jjk * lattice->displacement(r_i, r_k)});
      }
    }
  }

  // Jij * Jik
  for (auto i = 0; i < nbr_list.size(); ++i) {
    // ij
    for (auto const &nbr_j: nbr_list[i]) {
      const int j = nbr_j.first;
      const auto Jij = nbr_j.second[0][0];
      // ik
      for (auto const &nbr_k: nbr_list[i]) {
        const int k = nbr_k.first;
        const auto Jik = nbr_k.second[0][0];
        if (i == j || j == k || i == k) continue;
        if (i > j || j > k || i > k) continue;
        auto r_i = lattice->atom_position(i);
        auto r_j = lattice->atom_position(j);
        auto r_k = lattice->atom_position(k);
        triads.push_back({i, j , k, Jij * Jik * lattice->displacement(r_j, r_i)});
      }
    }
  }

  // Jik * Jjk
  for (auto i = 0; i < nbr_list.size(); ++i) {
    // ik
    for (auto const &nbr_k: nbr_list[i]) {
      const int  k = nbr_k.first;
      const auto Jik = nbr_k.second[0][0];
      // jk
      for (auto const &nbr_j: nbr_list[k]) {
        const int  j = nbr_j.first;
        const auto Jjk = nbr_j.second[0][0];
        if (i == j || j == k || i == k) continue;
        if (i > j || j > k || i > k) continue;
        auto r_i = lattice->atom_position(i);
        auto r_j = lattice->atom_position(j);
        auto r_k = lattice->atom_position(k);
        triads.push_back({i, j , k, Jik * Jjk * lattice->displacement(r_k, r_j)});
      }
    }
  }

  return triads;
}

void CudaThermalCurrentMonitor::initialize_device_data(const TriadList &triads) {

  jblib::Array<int, 1>    index_pointers;
  jblib::Array<int, 2>    index_data;
  jblib::Array<double, 2> value_data;

  convert_triad_list_to_csr_format(triads, globals::num_spins, index_pointers, index_data, value_data);

//
//  for (auto i = 0; i < globals::num_spins; ++i) {
//
//    const int begin = index_pointers[i];
//    const int end = index_pointers[i + 1];
//
//    for (int n = begin; n < end; ++n) {
//      const int j = index_data[2 * n];
//      const int k = index_data[2 * n + 1];
//
//      std::cerr << begin << "\t" << end << "\t" <<  i << "\t" << j << "\t" << k << "\t" << value_data[3*n + 0] << "\t" << value_data[3*n + 1] << "\t" << value_data[3*n + 2] << std::endl;
//    }
//  }

  cuda_copy_array_to_device_pointer(index_pointers, &dev_csr_matrix_.row);
  cuda_copy_array_to_device_pointer(index_data, &dev_csr_matrix_.col);
  cuda_copy_array_to_device_pointer(value_data, &dev_csr_matrix_.val);

  dev_thermal_current_rx.resize(globals::num_spins);
  dev_thermal_current_ry.resize(globals::num_spins);
  dev_thermal_current_rz.resize(globals::num_spins);

  dev_thermal_current_rx.zero();
  dev_thermal_current_ry.zero();
  dev_thermal_current_rz.zero();
}
