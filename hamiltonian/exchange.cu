#include <set>
#include <tuple>

#include "core/exception.h"
#include "core/globals.h"
#include "core/utils.h"
#include "core/consts.h"
#include "core/cuda_defs.h"
#include "core/cuda_sparsematrix.h"
#include "core/interaction-list.h"

#include "jblib/math/summations.h"

#include "hamiltonian/exchange.h"


bool operator <(const inode_t& x, const inode_t& y) {
    return std::tie(x.k, x.a, x.b, x.c) < std::tie(y.k, y.a, y.b, y.c);
}

void ExchangeHamiltonian::insert_interaction(const int i, const int j, const jblib::Matrix<double, 3, 3> &value) {
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

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings)
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

    if (!settings.exists("exc_file")) {
        jams_error("ExchangeHamiltonian: an exchange file must be specified");
    }

    std::string interaction_filename = settings["exc_file"];

    ::output.verbose("\ninteraction filename (%s)\n", interaction_filename.c_str());

    // read in typeA typeB rx ry rz Jij
    std::ifstream interaction_file(interaction_filename.c_str());

    if (interaction_file.fail()) {
      jams_error("failed to open interaction file %s", interaction_filename.c_str());
    }

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

    ::output.write("\ninteraction vectors (%s)\n", interaction_filename.c_str());

    InteractionList<inode_pair_t> unitcell_interactions(lattice.num_unit_cell_positions());

    neighbour_list_.resize(globals::num_spins);

    bool use_symops = true;
    if (settings.exists("symops")) {
      use_symops = settings["symops"];
    }

    read_interactions(interaction_file, unitcell_interactions, use_symops);
   
    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);

    if (solver->is_cuda_solver()) {
#ifdef CUDA
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
      // interaction_matrix_.setMatrixMode(SPARSE_FILL_MODE_LOWER);
#endif  //CUDA
    } else {
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
    }

    ::output.write("\ncomputed interactions\n");

    bool is_all_inserts_successful = true;
    int counter = 0;
    // loop over the translation vectors for lattice size
    for (int i = 0; i < lattice.num_unit_cells(0); ++i) {
      for (int j = 0; j < lattice.num_unit_cells(1); ++j) {
        for (int k = 0; k < lattice.num_unit_cells(2); ++k) {
          // loop over atoms in the unit_cell
          for (int m = 0; m < lattice.num_unit_cell_positions(); ++m) {
            int local_site = lattice.site_index_by_unit_cell(i, j, k, m);

            std::vector<bool> is_already_interacting(globals::num_spins, false);
            is_already_interacting[local_site] = true;  // don't allow self interaction

            // loop over all possible interaction vectors
            for (auto const &pair: unitcell_interactions[m]) {

              const inode_t inode = pair.second.node;
              const Mat3 tensor   = pair.second.value;

              inode_t ivec = 
              {(lattice.num_unit_cell_positions() + m + inode.k)%lattice.num_unit_cell_positions(),
                i + inode.a,
                j + inode.b,
                k + inode.c};

              if (lattice.apply_boundary_conditions(ivec.a, ivec.b, ivec.c) == false) {
                continue;
              }

              int neighbour_site = lattice.site_index_by_unit_cell(ivec.a, ivec.b, ivec.c, ivec.k);

              // failsafe check that we only interact with any given site once through the input exchange file.
              if (is_already_interacting[neighbour_site]) {
                jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_site, ivec.a, ivec.b, ivec.c, ivec.k);
              }
              is_already_interacting[neighbour_site] = true;

              if (tensor.max_norm() > energy_cutoff_) {
                neighbour_list_.insert(local_site, neighbour_site, tensor);
                counter++;

                if (is_debug_enabled_) {
                  debug_file << local_site << "\t" << neighbour_site << "\t";
                  debug_file << lattice.atom_position(local_site).x << "\t";
                  debug_file << lattice.atom_position(local_site).y << "\t";
                  debug_file << lattice.atom_position(local_site).z << "\t";
                  debug_file << lattice.atom_position(neighbour_site).x << "\t";
                  debug_file << lattice.atom_position(neighbour_site).y << "\t";
                  debug_file << lattice.atom_position(neighbour_site).z << "\n";
                }
              } else {
                is_all_inserts_successful = false;
              }
            }
            if (is_debug_enabled_) {
              debug_file << "\n\n";
            }
          }
        }
      }
    }

    if (is_debug_enabled_) {
      debug_file.close();
    }

    for (int i = 0; i < globals::num_spins; ++i) {
      for (auto const &j: neighbour_list_[i]) {
        insert_interaction(i, j.first, j.second);
      }
    }

    if (!is_all_inserts_successful) {
      jams_warning("Some interactions were ignored due to the energy cutoff (%e)", energy_cutoff_);
    }

    ::output.write("  total unit cell interactions: %d\n", counter);

    // resize member arrays
    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);

    ::output.write("  converting interaction matrix format from MAP to CSR\n");
    interaction_matrix_.convertMAP2CSR();
    ::output.write("  exchange matrix memory (CSR): %f MB\n", interaction_matrix_.calculateMemory());

    // transfer arrays to cuda device if needed
    if (solver->is_cuda_solver()) {
#ifdef CUDA

        cudaStreamCreate(&dev_stream_);

        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_  = jblib::CudaArray<double, 1>(field_);

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

void ExchangeHamiltonian::read_interactions(std::ifstream &interaction_file,
  InteractionList<inode_pair_t> &interactions, bool use_symops) {

  std::ofstream unfolded_interaction_file;

  if (is_debug_enabled_) {
    unfolded_interaction_file.open(std::string(seedname+"_unfolded_exc.dat").c_str());
    if(unfolded_interaction_file.fail()) {
      throw general_exception("failed to open unfolded interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }
  ::output.write("  reading interactions and applying symmetry operations\n");

  if (::output.is_verbose()) {
    ::output.verbose("unit cell realspace\n");
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      jblib::Vec3<double> rij = lattice.unit_cell_position(i);
      ::output.verbose("%8d % 6.6f % 6.6f % 6.6f\n", i, rij[0], rij[1], rij[2]);
    }
  }

  int counter = 0;
  int line_number = 0;

  // read the unit_cell into an array from the positions file
  for (std::string line; getline(interaction_file, line); ) {

    if(string_is_comment(line)) {
      continue;
    }

    std::stringstream   is(line);

    std::string type_name_A, type_name_B;
    is >> type_name_A >> type_name_B;

    if (is.bad()) {
      throw general_exception("failed to read types in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    Vec3 interaction_vector;
    is >> interaction_vector.x >> interaction_vector.y >> interaction_vector.z;

    if (is.bad()) {
      throw general_exception("failed to read interaction vector in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    Mat3 tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (file_columns(line) == 6) {
      // one Jij component given - diagonal
      is >> tensor[0][0];
      tensor[1][1] = tensor[0][0];
      tensor[2][2] = tensor[0][0];
    } else if (file_columns(line) == 14) {
      // nine Jij components given - full tensor
      is >> tensor[0][0] >> tensor[0][1] >> tensor[0][2];
      is >> tensor[1][0] >> tensor[1][1] >> tensor[1][2];
      is >> tensor[2][0] >> tensor[2][1] >> tensor[2][2];
    } else {
      throw general_exception("number of Jij values in exchange files must be 1 or 9, check your input on line " + line_number, __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    if (is.bad()) {
      throw general_exception("failed to read exchange tensor in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    tensor = tensor / kBohrMagneton;

    ::output.verbose("exchange file line: %s\n", line.c_str());

    std::vector<Vec3> symmetric_vectors;

    if (use_symops) {
      for (int i = 0; i < lattice.num_sym_ops(); i++) {
        Vec3 rij = lattice.cartesian_to_fractional(interaction_vector);
        symmetric_vectors.push_back(lattice.sym_rotation(i, rij));
      }
    } else {
      symmetric_vectors.push_back(lattice.cartesian_to_fractional(interaction_vector));
    }

    // if the origin of the unit cell is in the center of the lattice vectors with other
    // atoms positioned around it (+ve and -ve) then we have to use nint later instead of
    // floor to work out which unit cell offset to use.
    //
    // currently only unit cells with origins at the corner or the center are supported
    bool is_centered_lattice = false;
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      if (lattice.unit_cell_position(i).x < 0.0 || lattice.unit_cell_position(i).y < 0.0 || lattice.unit_cell_position(i).z < 0.0) {
        is_centered_lattice = true;
        jams_warning("Centered lattice is detected. Make sure you know what you are doing!");
        break;
      }
    }

    ::output.verbose("unit cell interactions\n");
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      std::set<inode_t> interaction_set;

      // only process for interactions belonging to this material
      if (lattice.unit_cell_material_name(i) == type_name_A) {

        // unit_cell position in basis vector space
        Vec3 pij = lattice.unit_cell_position(i);
        ::output.verbose("unit_cell position %d [%s] (% 3.6f % 3.6f % 3.6f)\n", i, lattice.unit_cell_material_name(i).c_str(), pij.x, pij.y, pij.z);
        ::output.verbose("interaction vector %d [%s] (% 3.6f % 3.6f % 3.6f)\n", line_number, type_name_B.c_str(), interaction_vector.x, interaction_vector.y, interaction_vector.z);

        for(auto const& rij: symmetric_vectors) {

          // this gives the integer unit cell of the interaction
          Vec3 uij = rij + pij;

          for (int k = 0; k < 3; ++k) {
            if (is_centered_lattice) {
              // usually nint is floor(x+0.5) but it depends on how the cell is defined :(
              // it seems using ceil is better supported with spglib
              uij[k] = ceil(uij[k]-0.5);
            } else {
              // adding the distance_tolerance_ allows us to still floor properly when the precision
              // of the interaction vectors is not so good.
              uij[k] = floor(uij[k] + distance_tolerance_);
            }
          }

          // now calculate the unit_cell offset
          Vec3 unit_cell_offset = rij + pij - uij;

          Vec3 rij_cart = lattice.fractional_to_cartesian(rij);

          ::output.verbose("% 3.6f % 3.6f % 3.6f | % 3.6f % 3.6f % 3.6f | % 6d % 6d % 6d | % 3.6f % 3.6f % 3.6f\n",
            rij.x ,rij.y, rij.z, rij_cart.x ,rij_cart.y, rij_cart.z, int(uij.x), int(uij.y), int(uij.z),
            unit_cell_offset.x, unit_cell_offset.y, unit_cell_offset.z);


          // find which unit_cell position this offset corresponds to
          // it is possible that it does not correspond to a position in which case the
          // "cell_nbr" is -1
          int cell_nbr = -1;
          for (int k = 0; k < lattice.num_unit_cell_positions(); ++k) {
            Vec3 pos = lattice.unit_cell_position(k);
            if ( fabs(pos.x - unit_cell_offset.x) < distance_tolerance_
              && fabs(pos.y - unit_cell_offset.y) < distance_tolerance_
              && fabs(pos.z - unit_cell_offset.z) < distance_tolerance_ ) {
              cell_nbr = k;
              break;
            }
          }

          if (cell_nbr != -1) {
            // check the unit_cell partner is of the type that was specified in the exchange input file
            if (lattice.unit_cell_material_name(cell_nbr) != type_name_B) {
              ::output.verbose("wrong type \n");
              continue;
            }

            inode_t interaction_node = {cell_nbr - i, int(uij[0]), int(uij[1]), int(uij[2])};

            std::pair<std::set<inode_t>::iterator,bool> ret;

            ret = interaction_set.insert(interaction_node);

            if (ret.second == true) {
              ::output.verbose("*** % 8d [%s] % 8d [%s] :: % 8d % 8d % 8d\n", i, lattice.unit_cell_material_name(i).c_str(), interaction_node.k + i, lattice.unit_cell_material_name(interaction_node.k + i).c_str(), interaction_node.a, interaction_node.b, interaction_node.b);
              interactions.insert(i, counter, {interaction_node, tensor});

              if (is_debug_enabled_) {
                unfolded_interaction_file << std::setw(12) << type_name_A << "\t";
                unfolded_interaction_file << std::setw(12) << type_name_B << "\t";
                unfolded_interaction_file << std::setw(12) << std::fixed << rij_cart.x << "\t";
                unfolded_interaction_file << std::setw(12) << std::fixed << rij_cart.y << "\t";
                unfolded_interaction_file << std::setw(12) << std::fixed << rij_cart.z << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[0][0] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[0][1] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[0][2] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[1][0] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[1][1] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[1][2] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[2][0] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[2][1] << "\t";
                unfolded_interaction_file << std::setw(12) << std::scientific << tensor[2][2] << std::endl;
              }

              counter++;
            }
          }
        }
        // for (std::set<jblib::Vec4<int> >::iterator it = interaction_set.begin(); it != interaction_set.end(); ++it) {



        // }
      }
    }
    line_number++;
  }
  ::output.write("  total unit cell interactions: %d\n", counter);

  unfolded_interaction_file.close();
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_total_energy() {
    jblib::KahanSum total_energy;

    for (int i = 0; i < globals::num_spins; ++i) {
        total_energy.add(calculate_one_spin_energy(i));
    }
    return total_energy.value();
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_one_spin_energy(const int i) {
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

double ExchangeHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double local_field[3], e_initial, e_final;

    calculate_one_spin_field(i, local_field);

    e_initial = -(spin_initial[0]*local_field[0] + spin_initial[1]*local_field[1] + spin_initial[2]*local_field[2]);
    e_final = -(spin_final[0]*local_field[0] + spin_final[1]*local_field[1] + spin_final[2]*local_field[2]);

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
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

void ExchangeHamiltonian::calculate_fields() {
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

void ExchangeHamiltonian::output_energies(OutputFormat format) {
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

void ExchangeHamiltonian::output_fields(OutputFormat format) {
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

void ExchangeHamiltonian::output_energies_text() {
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

void ExchangeHamiltonian::output_fields_text() {

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

sparse_matrix_format_t ExchangeHamiltonian::sparse_matrix_format() {
  return interaction_matrix_format_;
}

void ExchangeHamiltonian::set_sparse_matrix_format(std::string &format_name) {
  if (capitalize(format_name) == "CSR") {
    interaction_matrix_format_ = SPARSE_MATRIX_FORMAT_CSR;
  } else if (capitalize(format_name) == "DIA") {
    if (solver->is_cuda_solver() != true) {
      jams_error("ExchangeHamiltonian::set_sparse_matrix_format: DIA format is only supported for CUDA");
    }
    interaction_matrix_format_ = SPARSE_MATRIX_FORMAT_DIA;
  } else {
    jams_error("ExchangeHamiltonian::set_sparse_matrix_format: Unknown format requested %s", format_name.c_str());
  }
}

double ExchangeHamiltonian::calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final) {
  using namespace globals;

  return 0.0;
  // if (i == j) {
  //   return 0.0;
  // } else {

  //   // Mat3 J;

  //   // J = neighbour_list_[i][j];

  //   // try {
  //   //   J = neighbour_list_[i].at(j);
  //   // }
  //   // catch(std::out_of_range) {
  //   //   return 0.0;
  //   // }

  //   Vec3 Js = neighbour_list_.interactions(i)[j] * (sj_final - sj_initial);
  //   return -(s(i, 0) * Js[0] + s(i, 1) * Js[1] + s(i, 2) * Js[2]);
  // }
}
