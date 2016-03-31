#include <set>

#include "core/globals.h"
#include "core/utils.h"
#include "core/consts.h"
#include "core/cuda_defs.h"
#include "core/cuda_sparsematrix.h"



#include "hamiltonian/exchange_neartree.h"

void ExchangeNeartreeHamiltonian::insert_interaction(const int i, const int j, const jblib::Matrix<double, 3, 3> &value) {
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

ExchangeNeartreeHamiltonian::ExchangeNeartreeHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings) {

    is_debug_enabled_ = false;
    std::ofstream debug_file;

    if (settings.exists("debug")) {
      is_debug_enabled_ = settings["debug"];
    }

    // if (settings.exists("sparse_format")) {
    //   set_sparse_matrix_format(std::string(settings["sparse_format"]));
    // }

    if (is_debug_enabled_) {
      debug_file.open("debug_exchange.dat");

      std::ofstream pos_file("debug_pos.dat");
      for (int n = 0; n < lattice.num_materials(); ++n) {
        for (int i = 0; i < globals::num_spins; ++i) {
          if (lattice.atom_material(i) == n) {
            pos_file << i << "\t" <<  lattice.atom_position(i).x << "\t" <<  lattice.atom_position(i).y << "\t" << lattice.atom_position(i).z << "\n";
          }
        }
        pos_file << "\n\n";
      }
      pos_file.close();
    }

    // output in default format for now
    outformat_ = TEXT;

    energy_cutoff_ = 1E-26;  // Joules
    if (settings.exists("energy_cutoff")) {
        energy_cutoff_ = settings["energy_cutoff"];
    }
    ::output.write("\ninteraction energy cutoff\n  %e\n", energy_cutoff_);

    distance_tolerance_ = 1e-3; // fractional coordinate units
    if (settings.exists("distance_tolerance")) {
        distance_tolerance_ = settings["distance_tolerance"];
    }

    ::output.write("\ndistance_tolerance\n  %e\n", distance_tolerance_);

    // --- SAFETY ---
    // check that no atoms in the unit cell are closer together than the distance_tolerance_
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      for (int j = i+1; j < lattice.num_unit_cell_positions(); ++j) {
        if( abs(lattice.unit_cell_position(i) - lattice.unit_cell_position(j)) < distance_tolerance_ ) {
          jams_error("Atoms %d and %d in the unit_cell are closer together (%f) than the distance_tolerance (%f).\n"
                     "Check position file or relax distance_tolerance for exchange module",
                      i, j, abs(lattice.unit_cell_position(i) - lattice.unit_cell_position(j)), distance_tolerance_);
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

    int type_id_A, type_id_B;
    std::string type_name_A, type_name_B;
    double jij_radius, jij_value;

    interaction_list_.resize(lattice.num_materials());

    for (int i = 0; i < settings["interactions"].getLength(); ++i) {

      type_name_A = settings["interactions"][i][0].c_str();
      type_name_B = settings["interactions"][i][1].c_str();

      jij_radius = settings["interactions"][i][2];
      jij_value = double(settings["interactions"][i][3]) / kBohrMagneton;

      // std::cout << type_name_A << "\t" << type_name_B << "\t" << jij_radius << "\t" << jij_value << std::endl;

      type_id_A = lattice.material_id(type_name_A);
      type_id_B = lattice.material_id(type_name_B);

      // std::cout << type_id_A << "\t" << type_id_B << "\t" << jij_radius << "\t" << jij_value << std::endl;

      InteractionNT jij = {type_id_A, type_id_B, jij_radius, jij_value};

      interaction_list_[type_id_A].push_back(jij);
    }

    //---------------------------------------------------------------------
    // create interaction matrix
    //---------------------------------------------------------------------

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);
    interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

    ::output.write("\ncomputed interactions\n");

    int counter = 0;
    for (int i = 0; i < globals::num_spins; ++i) {
      std::vector<bool> is_already_interacting(globals::num_spins, false);

      int type = lattice.atom_material(i);

      for (int j = 0; j < interaction_list_[type].size(); ++j) {
        std::vector<Atom> nbr_lower;
        std::vector<Atom> nbr_upper;

        lattice.atom_neighbours(i, interaction_list_[type][j].radius - distance_tolerance_, nbr_lower);
        lattice.atom_neighbours(i, interaction_list_[type][j].radius + distance_tolerance_, nbr_upper);


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

            if (is_debug_enabled_) {
              debug_file << i << "\t" << n.id << "\t";
              debug_file << lattice.atom_position(i).x << "\t";
              debug_file << lattice.atom_position(i).y << "\t";
              debug_file << lattice.atom_position(i).z << "\t";
              debug_file << lattice.atom_position(n.id).x << "\t";
              debug_file << lattice.atom_position(n.id).y << "\t";
              debug_file << lattice.atom_position(n.id).z << "\n";
            }
          }
        }
      }
      if (is_debug_enabled_) {
        debug_file << "\n\n";
      }
    }

    if (is_debug_enabled_) {
      debug_file.close();
    }

    ::output.write("  total interactions: %d\n", counter);

    // resize member arrays
    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);

    ::output.write("  converting interaction matrix format from MAP to CSR\n");
    interaction_matrix_.convertMAP2CSR();
    ::output.write("  exchange matrix memory (CSR): %f MB\n", interaction_matrix_.calculateMemory());

    //---------------------------------------------------------------------
    // initialize CUDA arrays
    //---------------------------------------------------------------------

    if (solver->is_cuda_solver()) {
#ifdef CUDA

        cudaStreamCreate(&dev_stream_);

        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_ = jblib::CudaArray<double, 1>(field_);

        if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR) {
          ::output.write("  * Initialising CUSPARSE...\n");
          cusparseStatus_t status;
          status = cusparseCreate(&cusparse_handle_);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            jams_error("CUSPARSE Library initialization failed");
          }
          cusparseSetStream(cusparse_handle_, dev_stream_);


          // create matrix descriptor
          status = cusparseCreateMatDescr(&cusparse_descra_);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            jams_error("CUSPARSE Matrix descriptor initialization failed");
          }
          cusparseSetMatType(cusparse_descra_,CUSPARSE_MATRIX_TYPE_GENERAL);
          cusparseSetMatIndexBase(cusparse_descra_,CUSPARSE_INDEX_BASE_ZERO);

          ::output.write("  allocating memory on device\n");
          cuda_api_error_check(
            cudaMalloc((void**)&dev_csr_interaction_matrix_.row, (interaction_matrix_.rows()+1)*sizeof(int)));
          cuda_api_error_check(
            cudaMalloc((void**)&dev_csr_interaction_matrix_.col, (interaction_matrix_.nonZero())*sizeof(int)));
          cuda_api_error_check(
            cudaMalloc((void**)&dev_csr_interaction_matrix_.val, (interaction_matrix_.nonZero())*sizeof(double)));

          cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.row, interaction_matrix_.rowPtr(),
                (interaction_matrix_.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

          cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.col, interaction_matrix_.colPtr(),
                (interaction_matrix_.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

          cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.val, interaction_matrix_.valPtr(),
                (interaction_matrix_.nonZero())*sizeof(double), cudaMemcpyHostToDevice));

        } else if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_DIA) {
          // ::output.write("  converting interaction matrix format from map to dia");
          // interaction_matrix_.convertMAP2DIA();
          ::output.write("  estimated memory usage (DIA): %f MB\n", interaction_matrix_.calculateMemory());
          dev_dia_interaction_matrix_.blocks = std::min<int>(DIA_BLOCK_SIZE, (globals::num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
          ::output.write("  allocating memory on device\n");

          // allocate rows
          cuda_api_error_check(
            cudaMalloc((void**)&dev_dia_interaction_matrix_.row, (interaction_matrix_.diags())*sizeof(int)));
          // allocate values
          cuda_api_error_check(
            cudaMallocPitch((void**)&dev_dia_interaction_matrix_.val, &dev_dia_interaction_matrix_.pitch,
              (interaction_matrix_.rows())*sizeof(double), interaction_matrix_.diags()));
          // copy rows
          cuda_api_error_check(
            cudaMemcpy(dev_dia_interaction_matrix_.row, interaction_matrix_.dia_offPtr(),
              (size_t)((interaction_matrix_.diags())*(sizeof(int))), cudaMemcpyHostToDevice));
          // convert val array into double which may be float or double
          std::vector<double> float_values(interaction_matrix_.rows()*interaction_matrix_.diags(), 0.0);

          for (int i = 0; i < interaction_matrix_.rows()*interaction_matrix_.diags(); ++i) {
            float_values[i] = static_cast<double>(interaction_matrix_.val(i));
          }

          // copy values
          cuda_api_error_check(
            cudaMemcpy2D(dev_dia_interaction_matrix_.val, dev_dia_interaction_matrix_.pitch, &float_values[0],
              interaction_matrix_.rows()*sizeof(double), interaction_matrix_.rows()*sizeof(double),
              interaction_matrix_.diags(), cudaMemcpyHostToDevice));

          dev_dia_interaction_matrix_.pitch = dev_dia_interaction_matrix_.pitch/sizeof(double);
        }
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

double ExchangeNeartreeHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
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
    // dev_s needs to be found from the solver

    if (solver->is_cuda_solver()) {
#ifdef CUDA
      if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR) {
        const double one = 1.0;
        const double zero = 0.0;
        cusparseStatus_t stat =
        cusparseDcsrmv(cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          globals::num_spins3,
          globals::num_spins3,
          interaction_matrix_.nonZero(),
          &one,
          cusparse_descra_,
          dev_csr_interaction_matrix_.val,
          dev_csr_interaction_matrix_.row,
          dev_csr_interaction_matrix_.col,
          solver->dev_ptr_spin(),
          &zero,
          dev_field_.data());
        assert(stat == CUSPARSE_STATUS_SUCCESS);
      } else if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_DIA) {
        spmv_dia_kernel<<< dev_dia_interaction_matrix_.blocks, DIA_BLOCK_SIZE >>>
            (globals::num_spins3, globals::num_spins3, interaction_matrix_.diags(), dev_dia_interaction_matrix_.pitch, 1.0, 0.0,
            dev_dia_interaction_matrix_.row, dev_dia_interaction_matrix_.val, solver->dev_ptr_spin(), dev_field_.data());
      }
#endif  // CUDA
    } else {
      if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL) {
        // general matrix (i.e. Monte Carlo Solvers)
        char transa[1] = {'N'};
        char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
        double one = 1.0;
        double zero = 0.0;
        mkl_dcsrmv(transa, &globals::num_spins3, &globals::num_spins3, &one, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), &zero, field_.data());
#else
        jams_dcsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, field_.data());
#endif
      } else {
        // symmetric matrix (i.e. Heun Solvers)
        char transa[1] = {'N'};
        char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
        double one = 1.0;
        double zero = 0.0;
        mkl_dcsrmv(transa, &globals::num_spins3, &globals::num_spins3, &one, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), &zero, field_.data());
#else
        jams_dcsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, field_.data());
#endif
      }
    }
}
// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::output_energies(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_energies_text();
        case HDF5:
            jams_error("Exchange energy output: HDF5 not yet implemented");
        default:
            jams_error("Exchange energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::output_fields(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_fields_text();
        case HDF5:
            jams_error("Exchange energy output: HDF5 not yet implemented");
        default:
            jams_error("Exchange energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::output_energies_text() {
    using namespace globals;

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_energy_.copy_to_host_array(energy_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_eng_uniaxial_"+zero_pad_number(outcount)+".dat");

    std::ofstream outfile(filename.c_str());

    outfile << "# type | rx (nm) | ry (nm) | rz (nm) | d2z | d4z | d6z" << std::endl;

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << lattice.atom_material(i);

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile <<  lattice.parameter()*lattice.atom_position(i)[j];
        }

        // energy

    }
    outfile.close();
}

// --------------------------------------------------------------------------

void ExchangeNeartreeHamiltonian::output_fields_text() {

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_field_.copy_to_host_array(field_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_field_uniaxial_"+zero_pad_number(outcount)+".dat");

    // using direct file access for performance
    std::ofstream outfile(filename.c_str());
    outfile.setf(std::ios::right);

    outfile << "#";
    outfile << std::setw(16) << "type";
    outfile << std::setw(16) << "rx (nm)";
    outfile << std::setw(16) << "ry (nm)";
    outfile << std::setw(16) << "rz (nm)";
    outfile << std::setw(16) << "hx (nm)";
    outfile << std::setw(16) << "hy (nm)";
    outfile << std::setw(16) << "hz (nm)";
    outfile << "\n";

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << std::setw(16) << lattice.atom_material(i);

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::fixed << lattice.parameter()*lattice.atom_position(i)[j];
        }

        // fields
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::scientific << field_(i,j);
        }
        outfile << "\n";
    }
    outfile.close();
}

sparse_matrix_format_t ExchangeNeartreeHamiltonian::sparse_matrix_format() {
  return interaction_matrix_format_;
}

void ExchangeNeartreeHamiltonian::set_sparse_matrix_format(std::string &format_name) {
  if (capitalize(format_name) == "CSR") {
    interaction_matrix_format_ = SPARSE_MATRIX_FORMAT_CSR;
  } else if (capitalize(format_name) == "DIA") {
    if (solver->is_cuda_solver() != true) {
      jams_error("ExchangeNeartreeHamiltonian::set_sparse_matrix_format: DIA format is only supported for CUDA");
    }
    interaction_matrix_format_ = SPARSE_MATRIX_FORMAT_DIA;
  } else {
    jams_error("ExchangeNeartreeHamiltonian::set_sparse_matrix_format: Unknown format requested %s", format_name.c_str());
  }
}

double ExchangeNeartreeHamiltonian::calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final) {
  using namespace globals;

  if (i == j) {
    return 0.0;
  } else {
    return 0.0;
  }
}
