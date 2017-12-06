#include <set>
#include <fstream>

#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/consts.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/cuda/cuda_sparsematrix.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "exchange_neartree.h"

using namespace std;

ExchangeNeartreeHamiltonian::~ExchangeNeartreeHamiltonian() {
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}

void ExchangeNeartreeHamiltonian::insert_interaction(const int i, const int j, const Mat3 &value) {
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 3; ++n) {
      if (std::abs(value[m][n]) > energy_cutoff_) {
        if(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          if(interaction_matrix_.getMatrixMode() == SPARSE_FILL_MODE_LOWER) {
            if (i >= j) {
              interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n]);
            }
          } else {
            if (i <= j) {
              interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n]);
            }
          }
        } else {
          interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n]);
        }
      }
    }
  }}

ExchangeNeartreeHamiltonian::ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {

    std::ofstream debug_file;

    if (debug_is_enabled()) {
      debug_file.open("debug_exchange.dat");

      std::ofstream pos_file("debug_pos.dat");
      for (int n = 0; n < lattice->num_materials(); ++n) {
        for (int i = 0; i < globals::num_spins; ++i) {
          if (lattice->atom_material_id(i) == n) {
            pos_file << i << "\t" <<  lattice->atom_position(i)[0] << "\t" <<  lattice->atom_position(i)[1] << "\t" << lattice->atom_position(i)[2] << "\n";
          }
        }
        pos_file << "\n\n";
      }
      pos_file.close();
    }

    energy_cutoff_ = 1E-26;  // Joules
    if (settings.exists("energy_cutoff")) {
        energy_cutoff_ = settings["energy_cutoff"];
    }
    cout << "interaction energy cutoff" << energy_cutoff_ << "\n";

    distance_tolerance_ = 1e-6; // fractional coordinate units
    if (settings.exists("distance_tolerance")) {
        distance_tolerance_ = settings["distance_tolerance"];
    }

    cout << "distance_tolerance " << distance_tolerance_ << "\n";

    // --- SAFETY ---
    // check that no atoms in the unit cell are closer together than the distance_tolerance_
  for (auto i = 0; i < lattice->motif_size(); ++i) {
    for (auto j = i+1; j < lattice->motif_size(); ++j) {
      const auto distance = abs(lattice->motif_atom(i).pos - lattice->motif_atom(j).pos);
      if(distance < distance_tolerance_) {
        jams_error("Atoms %d and %d in the unit_cell are closer together (%f) than the distance_tolerance (%f).\n"
                        "Check position file or relax distance_tolerance for exchange module",
                i, j, distance, distance_tolerance_);
      }
    }
  }
    // --------------

    //---------------------------------------------------------------------
    // read interactions from config
    //---------------------------------------------------------------------

    if (!settings.exists("interactions")) {
      jams_error("No interactions defined in ExchangeNeartree hamiltonian");
    }

    interaction_list_.resize(lattice->num_materials());

    for (int i = 0; i < settings["interactions"].getLength(); ++i) {

      std::string type_name_A = settings["interactions"][i][0].c_str();
      std::string type_name_B = settings["interactions"][i][1].c_str();

      double inner_radius = settings["interactions"][i][2];
      double outer_radius = settings["interactions"][i][3];

      double jij_value = double(settings["interactions"][i][4]) / kBohrMagneton;

      auto type_id_A = lattice->material_id(type_name_A);
      auto type_id_B = lattice->material_id(type_name_B);

      InteractionNT jij = {type_id_A, type_id_B, inner_radius, outer_radius, jij_value};

      interaction_list_[type_id_A].push_back(jij);
    }

    //---------------------------------------------------------------------
    // create interaction matrix
    //---------------------------------------------------------------------

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);
    interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

    cout << "\ncomputed interactions\n";

    int counter = 0;
    for (int i = 0; i < globals::num_spins; ++i) {
      std::vector<bool> is_already_interacting(globals::num_spins, false);

      int type = lattice->atom_material_id(i);

      for (int j = 0; j < interaction_list_[type].size(); ++j) {
        std::vector<Atom> nbr_lower;
        std::vector<Atom> nbr_upper;

        lattice->atom_neighbours(i, interaction_list_[type][j].inner_radius, nbr_lower);
        lattice->atom_neighbours(i, interaction_list_[type][j].outer_radius + distance_tolerance_, nbr_upper);

        std::vector<Atom> nbr(std::max(nbr_lower.size(), nbr_upper.size()));

        auto compare_func = [](Atom a, Atom b) { return a.id > b.id; };

        std::sort(nbr_lower.begin(), nbr_lower.end(), compare_func);
        std::sort(nbr_upper.begin(), nbr_upper.end(), compare_func);

        auto it = std::set_difference(nbr_upper.begin(), nbr_upper.end(), nbr_lower.begin(), nbr_lower.end(), nbr.begin(), compare_func);

        nbr.resize(it - nbr.begin());

        for (const Atom n : nbr) {
          if (n.id == i) {
            continue;
          }

          if (n.material == interaction_list_[type][j].material[1]) {

            // don't allow self interaction
            if (is_already_interacting[n.id]) {
              jams_error("Multiple interactions between spins %d and %d.\n", i, n.id);
            }
            is_already_interacting[n.id] = true;

            double jij = interaction_list_[type][j].value;

            // std::cout << i << "\t" << n.id << "\t" << jij << std::endl;

            insert_interaction(i, n.id, {jij, 0.0, 0.0, 0.0, jij, 0.0, 0.0, 0.0, jij});
            counter++;

            if (debug_is_enabled()) {
              debug_file << i << "\t" << n.id << "\t";
              debug_file << lattice->atom_position(i)[0] << "\t";
              debug_file << lattice->atom_position(i)[1] << "\t";
              debug_file << lattice->atom_position(i)[2] << "\t";
              debug_file << lattice->atom_position(n.id)[0] << "\t";
              debug_file << lattice->atom_position(n.id)[1] << "\t";
              debug_file << lattice->atom_position(n.id)[2] << "\n";
            }
          }
        }
      }
      if (debug_is_enabled()) {
        debug_file << "\n\n";
      }
    }

  if (debug_is_enabled()) {
      debug_file.close();
    }

    cout << "  total interactions " << counter << "\n";

  cout << "    converting interaction matrix format from MAP to CSR\n";
  interaction_matrix_.convertMAP2CSR();
  cout << "    exchange matrix memory (CSR): " << interaction_matrix_.calculateMemory() << " (MB)\n";

  // transfer arrays to cuda device if needed
  if (solver->is_cuda_solver()) {
#ifdef CUDA
    cudaStreamCreate(&dev_stream_);

    dev_energy_ = jblib::CudaArray<double, 1>(energy_);
    dev_field_  = jblib::CudaArray<double, 1>(field_);

    cout << "    init cusparse\n";
    cusparseStatus_t status;
    status = cusparseCreate(&cusparse_handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      jams_error("cusparse Library initialization failed");
    }
    cusparseSetStream(cusparse_handle_, dev_stream_);

    sparsematrix_copy_host_csr_to_cuda_csr(interaction_matrix_, dev_csr_interaction_matrix_);
#endif
  }

}

// --------------------------------------------------------------------------

double ExchangeNeartreeHamiltonian::calculate_total_energy() {
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_one_spin_energy(i);
    }
    return e_total;
}

// --------------------------------------------------------------------------

double ExchangeNeartreeHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double jij_sj[3] = {0.0, 0.0, 0.0};
    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = s.data();

    for (int m = 0; m < 3; ++m) {
      int begin = ptrb[3*i+m]; int end = ptre[3*i+m];
      for (int j = begin; j < end; ++j) {
        jij_sj[m] = jij_sj[m] + x[ indx[j] ]*val[j];
      }
    }
    return -(s(i,0)*jij_sj[0] + s(i,1)*jij_sj[1] + s(i,2)*jij_sj[2]);
}

// --------------------------------------------------------------------------

double ExchangeNeartreeHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double local_field[3], e_initial, e_final;

    calculate_one_spin_field(i, local_field);

    e_initial = -(spin_initial[0]*local_field[0] + spin_initial[1]*local_field[1] + spin_initial[2]*local_field[2]);
    e_final = -(spin_final[0]*local_field[0] + spin_final[1]*local_field[1] + spin_final[2]*local_field[2]);

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    local_field[0] = 0.0, local_field[1] = 0.0; local_field[2] = 0.0;

    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = s.data();
    int j, m, begin, end;

    for (m = 0; m < 3; ++m) {
      begin = ptrb[3*i+m]; end = ptre[3*i+m];
      for (j = begin; j < end; ++j) {
        // k = indx[j];
        local_field[m] = local_field[m] + x[ indx[j] ]*val[j];
      }
    }
}

// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::calculate_fields() {
  assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);
  double one = 1.0;
  double zero = 0.0;
  const char transa[1] = {'N'};
  const char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};
  const int num_rows = globals::num_spins3;
  const int num_cols = globals::num_spins3;

  if (solver->is_cuda_solver()) {
#ifdef CUDA
    cusparseStatus_t stat =
            cusparseDcsrmv(cusparse_handle_,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    num_rows,
                    num_cols,
                    interaction_matrix_.nonZero(),
                    &one,
                    dev_csr_interaction_matrix_.descr,
                    dev_csr_interaction_matrix_.val,
                    dev_csr_interaction_matrix_.row,
                    dev_csr_interaction_matrix_.col,
                    solver->dev_ptr_spin(),
                    &zero,
                    dev_field_.data());

    if (debug_is_enabled()) {
      if (stat != CUSPARSE_STATUS_SUCCESS) {
        throw cuda_api_exception("cusparse failure", __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }
      assert(stat == CUSPARSE_STATUS_SUCCESS);
    }
#endif  // CUDA
  } else {
#ifdef USE_MKL

    mkl_dcsrmv(transa, &num_rows, &num_cols, &one, matdescra, interaction_matrix_.valPtr(),
            interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(),
            &zero, field_.data());
#else
    jams_dcsrmv(transa, num_rows, num_cols, 1.0, matdescra, interaction_matrix_.valPtr(),
      interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, field_.data());
#endif
  }
}
