#include <set>

#include "core/globals.h"
#include "core/utils.h"
#include "core/consts.h"
#include "core/cuda_defs.h"
#include "core/cuda_sparsematrix.h"



#include "hamiltonian/exchange.h"
// #include "hamiltonian/exchange_kernel.h"

namespace {
  struct vec4_compare {
    bool operator() (const jblib::Vec4<int>& lhs, const jblib::Vec4<int>& rhs) const {
      if (lhs.x < rhs.x) {
        return true;
      } else if (lhs.x == rhs.x && lhs.y < rhs.y) {
        return true;
      } else if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z) {
        return true;
      } else if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w < rhs.w) {
        return true;
      }
      return false;
    }
  };

}

bool ExchangeHamiltonian::insert_interaction(const int m, const int n, const jblib::Matrix<double, 3, 3> &value) {

  int counter = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (fabs(value[i][j]) > energy_cutoff_) {
        counter++;
        if(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          if(interaction_matrix_.getMatrixMode() == SPARSE_FILL_MODE_LOWER) {
            if(m >= n){
              interaction_matrix_.insertValue(3*m+i, 3*n+j, value[i][j]/kBohrMagneton);
            }
          }else{
            if(m <= n){

              interaction_matrix_.insertValue(3*m+i, 3*n+j, value[i][j]/kBohrMagneton);
            }
          }
        }else{
          interaction_matrix_.insertValue(3*m+i, 3*n+j, value[i][j]/kBohrMagneton);
        }
      }
    }
  }

  if (counter == 0) {
    return false;
  }

  return true;
}

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
          if (lattice.material(i) == n) {
            pos_file << i << "\t" <<  lattice.position(i).x << "\t" <<  lattice.position(i).y << "\t" << lattice.position(i).z << "\n";
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

    std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > int_interaction_list(lattice.num_unit_cell_positions());

    bool use_symops = true;
    if (settings.exists("symops")) {
      use_symops = settings["symops"];
    }

    if (use_symops) {
      read_interactions_with_symmetry(interaction_filename, int_interaction_list);
    } else {
      read_interactions(interaction_filename, int_interaction_list);
    }

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);

    // if (solver->is_cuda_solver()) {
// #ifdef CUDA
//       interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
//       interaction_matrix_.setMatrixMode(SPARSE_FILL_MODE_LOWER);
// #endif  //CUDA
//     } else {
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
    // }

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
            for (int n = 0, nend = int_interaction_list[m].size(); n < nend; ++n) {

              jblib::Vec4<int> int_lookup_vec(
                i + int_interaction_list[m][n].first.x,
                j + int_interaction_list[m][n].first.y,
                k + int_interaction_list[m][n].first.z,
                (lattice.num_unit_cell_positions() + m + int_interaction_list[m][n].first.w)%lattice.num_unit_cell_positions());

              if (lattice.apply_boundary_conditions(int_lookup_vec) == false) {
                continue;
              }

              int neighbour_site = lattice.site_index_by_unit_cell(
                int_lookup_vec.x, int_lookup_vec.y, int_lookup_vec.z, int_lookup_vec.w);

              // failsafe check that we only interact with any given site once through the input exchange file.
              if (is_already_interacting[neighbour_site]) {
                // jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_site, int_lookup_vec.x, int_lookup_vec.y, int_lookup_vec.z, int_lookup_vec.w);
              }
              is_already_interacting[neighbour_site] = true;

              if (insert_interaction(local_site, neighbour_site, int_interaction_list[m][n].second)) {
                if (is_debug_enabled_) {
                  debug_file << local_site << "\t" << neighbour_site << "\t";
                  debug_file << lattice.position(local_site).x << "\t";
                  debug_file << lattice.position(local_site).y << "\t";
                  debug_file << lattice.position(local_site).z << "\t";
                  debug_file << lattice.position(neighbour_site).x << "\t";
                  debug_file << lattice.position(neighbour_site).y << "\t";
                  debug_file << lattice.position(neighbour_site).z << "\n";
                }
                // if(local_site >= neighbour_site) {
                //   std::cerr << local_site << "\t" << neighbour_site << "\t" << neighbour_site << "\t" << local_site << std::endl;
                // }
                counter++;
              } else {
                is_all_inserts_successful = false;
              }
              if (insert_interaction(neighbour_site, local_site, int_interaction_list[m][n].second)) {
                counter++;
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
        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_ = jblib::CudaArray<double, 1>(field_);

        if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR) {
          ::output.write("  * Initialising CUSPARSE...\n");
          cusparseStatus_t status;
          status = cusparseCreate(&cusparse_handle_);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            jams_error("CUSPARSE Library initialization failed");
          }

          // create matrix descriptor
          status = cusparseCreateMatDescr(&cusparse_descra_);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            jams_error("CUSPARSE Matrix descriptor initialization failed");
          }
          cusparseSetMatType(cusparse_descra_,CUSPARSE_MATRIX_TYPE_GENERAL);
          cusparseSetMatIndexBase(cusparse_descra_,CUSPARSE_INDEX_BASE_ZERO);

          ::output.write("  allocating memory on device\n");
          CUDA_CALL(cudaMalloc((void**)&dev_csr_interaction_matrix_.row, (interaction_matrix_.rows()+1)*sizeof(int)));
          CUDA_CALL(cudaMalloc((void**)&dev_csr_interaction_matrix_.col, (interaction_matrix_.nonZero())*sizeof(int)));
          CUDA_CALL(cudaMalloc((void**)&dev_csr_interaction_matrix_.val, (interaction_matrix_.nonZero())*sizeof(double)));

          CUDA_CALL(cudaMemcpy(dev_csr_interaction_matrix_.row, interaction_matrix_.rowPtr(),
                (interaction_matrix_.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

          CUDA_CALL(cudaMemcpy(dev_csr_interaction_matrix_.col, interaction_matrix_.colPtr(),
                (interaction_matrix_.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

          CUDA_CALL(cudaMemcpy(dev_csr_interaction_matrix_.val, interaction_matrix_.valPtr(),
                (interaction_matrix_.nonZero())*sizeof(double), cudaMemcpyHostToDevice));

        } else if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_DIA) {
          // ::output.write("  converting interaction matrix format from map to dia");
          // interaction_matrix_.convertMAP2DIA();
          ::output.write("  estimated memory usage (DIA): %f MB\n", interaction_matrix_.calculateMemory());
          dev_dia_interaction_matrix_.blocks = std::min<int>(DIA_BLOCK_SIZE, (globals::num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
          ::output.write("  allocating memory on device\n");

          // allocate rows
          CUDA_CALL(cudaMalloc((void**)&dev_dia_interaction_matrix_.row, (interaction_matrix_.diags())*sizeof(int)));
          // allocate values
          CUDA_CALL(cudaMallocPitch((void**)&dev_dia_interaction_matrix_.val, &dev_dia_interaction_matrix_.pitch,
              (interaction_matrix_.rows())*sizeof(double), interaction_matrix_.diags()));
          // copy rows
          CUDA_CALL(cudaMemcpy(dev_dia_interaction_matrix_.row, interaction_matrix_.dia_offPtr(),
              (size_t)((interaction_matrix_.diags())*(sizeof(int))), cudaMemcpyHostToDevice));
          // convert val array into double which may be float or double
          std::vector<double> float_values(interaction_matrix_.rows()*interaction_matrix_.diags(), 0.0);

          for (int i = 0; i < interaction_matrix_.rows()*interaction_matrix_.diags(); ++i) {
            float_values[i] = static_cast<double>(interaction_matrix_.val(i));
          }

          // copy values
          CUDA_CALL(cudaMemcpy2D(dev_dia_interaction_matrix_.val, dev_dia_interaction_matrix_.pitch, &float_values[0],
              interaction_matrix_.rows()*sizeof(double), interaction_matrix_.rows()*sizeof(double),
              interaction_matrix_.diags(), cudaMemcpyHostToDevice));

          dev_dia_interaction_matrix_.pitch = dev_dia_interaction_matrix_.pitch/sizeof(double);
        }
#endif
  }

}

void ExchangeHamiltonian::read_interactions(const std::string &filename,
  std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list) {

  ::output.write("  reading interactions (no symops)\n");


  std::ifstream interaction_file(filename.c_str());

  if(interaction_file.fail()) {
    jams_error("failed to open interaction file %s", filename.c_str());
  }

  std::ofstream unfolded_interaction_file;
  if (is_debug_enabled_) {
    unfolded_interaction_file.open(std::string(seedname+"_unfolded_exc.dat").c_str());
  }

  int_interaction_list.resize(lattice.num_unit_cell_positions());


  int counter = 0;
  int line_number = 0;

  ::output.verbose("\ninteraction vectors (%s)\n", filename.c_str());

  if (::output.is_verbose()) {
    ::output.verbose("unit cell realspace\n");
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      jblib::Vec3<double> rij = lattice.unit_cell_position(i);
      ::output.verbose("%8d % 6.6f % 6.6f % 6.6f\n", i, rij[0], rij[1], rij[2]);
    }
  }

  // read the unit_cell into an array from the positions file
  for (std::string line; getline(interaction_file, line); ) {
    std::stringstream is(line);

    std::string type_name_A, type_name_B;
    // read type names
    is >> type_name_A >> type_name_B;

    jblib::Vec3<double> interaction_vector;
    is >> interaction_vector.x >> interaction_vector.y >> interaction_vector.z;

    jblib::Matrix<double, 3, 3> tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
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
      jams_error("number of Jij values in exchange files must be 1 or 9, check your input on line %d", counter);
    }

    ::output.verbose("exchange file line: %s\n", line.c_str());

    jblib::Vec3<double> r(interaction_vector.x, interaction_vector.y, interaction_vector.z);

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
      std::set<jblib::Vec4<int>, vec4_compare> interaction_set;
      // only process for interactions belonging to this material
      if (lattice.unit_cell_material_name(i) == type_name_A) {
        // unit_cell position in basis vector space
        jblib::Vec3<double> pij = lattice.unit_cell_position(i);
        ::output.verbose("unit_cell position %d [%s] (% 3.6f % 3.6f % 3.6f)\n", i, lattice.unit_cell_material_name(i).c_str(), pij.x, pij.y, pij.z);
        ::output.verbose("interaction vector %d [%s] (% 3.6f % 3.6f % 3.6f)\n", line_number, type_name_B.c_str(), r.x, r.y, r.z);


        // position of neighbour in real space
        jblib::Vec3<double> rij = lattice.cartesian_to_fractional(r);
        // this gives the integer unit cell of the interaction
        jblib::Vec3<double> uij = rij + pij;
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
        // this 4-vector specifies the integer number of lattice vectors to the unit cell and the fourth
        // component is the atoms number within the unit_cell
        jblib::Vec4<int> fast_integer_vector;
        for (int k = 0; k < 3; ++k) {
          fast_integer_vector[k] = uij[k];
        }

        // now calculate the unit_cell offset
        jblib::Vec3<double> unit_cell_offset;

        unit_cell_offset = rij + pij - uij;

        jblib::Vec3<double> rij_cart = lattice.fractional_to_cartesian(rij);
        ::output.verbose("% 3.6f % 3.6f % 3.6f | % 3.6f % 3.6f % 3.6f | % 6d % 6d % 6d | % 3.6f % 3.6f % 3.6f\n",
          rij.x ,rij.y, rij.z, rij_cart.x ,rij_cart.y, rij_cart.z, int(uij.x), int(uij.y), int(uij.z),
          unit_cell_offset.x, unit_cell_offset.y, unit_cell_offset.z);


        // find which unit_cell position this offset corresponds to
        // it is possible that it does not correspond to a position in which case the
        // "unit_cell_partner" is -1
        int unit_cell_partner = -1;
        for (int k = 0; k < lattice.num_unit_cell_positions(); ++k) {
          jblib::Vec3<double> pos = lattice.unit_cell_position(k);
          if ( fabs(pos.x - unit_cell_offset.x) < distance_tolerance_
            && fabs(pos.y - unit_cell_offset.y) < distance_tolerance_
            && fabs(pos.z - unit_cell_offset.z) < distance_tolerance_ ) {
            unit_cell_partner = k;
          break;
        }
      }

      if (unit_cell_partner != -1) {
        // fast_integer_vector[3] = unit_cell_partner - i;
        // check the unit_cell partner is of the type that was specified in the exchange input file
        if (lattice.unit_cell_material_name(unit_cell_partner) != type_name_B) {
          ::output.verbose("wrong type \n");
          continue;
        }
        jblib::Vec4<int> fast_integer_vector(uij[0], uij[1], uij[2], unit_cell_partner - i);
        ::output.verbose("*** % 8d [%s] % 8d [%s] :: % 8d % 8d % 8d\n", i, lattice.unit_cell_material_name(i).c_str(), fast_integer_vector[3] + i, lattice.unit_cell_material_name(fast_integer_vector[3] + i).c_str(), fast_integer_vector[0], fast_integer_vector[1], fast_integer_vector[2]);
        int_interaction_list[i].push_back(std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >(fast_integer_vector, tensor));

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
  line_number++;
}
::output.write("  total unit cell interactions: %d\n", counter);
unfolded_interaction_file.close();
}

void ExchangeHamiltonian::read_interactions_with_symmetry(const std::string &filename,
  std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list) {

  ::output.write("  reading interactions and applying symmetry operations\n");

  std::ifstream interaction_file(filename.c_str());


  if(interaction_file.fail()) {
    jams_error("failed to open interaction file %s", filename.c_str());
  }

  std::ofstream unfolded_interaction_file;
  if (is_debug_enabled_) {
    unfolded_interaction_file.open(std::string(seedname+"_unfolded_exc.dat").c_str());
  }

  int_interaction_list.resize(lattice.num_unit_cell_positions());

  int counter = 0;
  int line_number = 0;

  ::output.verbose("\ninteraction vectors (%s)\n", filename.c_str());

  if (::output.is_verbose()) {
    ::output.verbose("unit cell realspace\n");
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      jblib::Vec3<double> rij = lattice.unit_cell_position(i);
      ::output.verbose("%8d % 6.6f % 6.6f % 6.6f\n", i, rij[0], rij[1], rij[2]);
    }
  }

  // read the unit_cell into an array from the positions file
  for (std::string line; getline(interaction_file, line); ) {
    std::stringstream is(line);

    std::string type_name_A, type_name_B;

    // read type names
    is >> type_name_A >> type_name_B;

    jblib::Vec3<double> interaction_vector;
    is >> interaction_vector.x >> interaction_vector.y >> interaction_vector.z;

    jblib::Matrix<double, 3, 3> tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
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
      jams_error("number of Jij values in exchange files must be 1 or 9, check your input on line %d", counter);
    }

    ::output.verbose("exchange file line: %s\n", line.c_str());

    jblib::Vec3<double> r(interaction_vector.x, interaction_vector.y, interaction_vector.z);
    jblib::Vec3<double> r_sym;

    jblib::Array<double, 2> sym_interaction_vecs(lattice.num_sym_ops(), 3);


    for (int i = 0; i < lattice.num_sym_ops(); i++) {
      jblib::Vec3<double> rij(r[0], r[1], r[2]);
      rij = lattice.cartesian_to_fractional(r);
      r_sym = lattice.sym_rotation(i, rij);
      for (int j = 0; j < 3; ++j) {
        sym_interaction_vecs(i,j) = r_sym[j]; // + spglib_dataset_->translations[i][j];
      }
      // ::output.verbose("%f %f %f\n", r_sym[0], r_sym[1], r_sym[2]);
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
      std::set<jblib::Vec4<int>, vec4_compare> interaction_set;
      // only process for interactions belonging to this material
      if (lattice.unit_cell_material_name(i) == type_name_A) {

        // unit_cell position in basis vector space
        jblib::Vec3<double> pij = lattice.unit_cell_position(i);
        ::output.verbose("unit_cell position %d [%s] (% 3.6f % 3.6f % 3.6f)\n", i, lattice.unit_cell_material_name(i).c_str(), pij.x, pij.y, pij.z);
        ::output.verbose("interaction vector %d [%s] (% 3.6f % 3.6f % 3.6f)\n", line_number, type_name_B.c_str(), r.x, r.y, r.z);

        for (int j = 0; j < sym_interaction_vecs.size(0); ++ j) {

          // position of neighbour in real space
          jblib::Vec3<double> rij(sym_interaction_vecs(j,0), sym_interaction_vecs(j,1), sym_interaction_vecs(j,2));

          // this gives the integer unit cell of the interaction
          jblib::Vec3<double> uij = rij + pij;
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

          // this 4-vector specifies the integer number of lattice vectors to the unit cell and the fourth
          // component is the atoms number within the unit_cell
          jblib::Vec4<int> fast_integer_vector;
          for (int k = 0; k < 3; ++k) {
            fast_integer_vector[k] = uij[k];
          }

          // now calculate the unit_cell offset
          jblib::Vec3<double> unit_cell_offset;

          unit_cell_offset = rij + pij - uij;

          jblib::Vec3<double> rij_cart = lattice.fractional_to_cartesian(rij);
          ::output.verbose("% 3.6f % 3.6f % 3.6f | % 3.6f % 3.6f % 3.6f | % 6d % 6d % 6d | % 3.6f % 3.6f % 3.6f\n",
            rij.x ,rij.y, rij.z, rij_cart.x ,rij_cart.y, rij_cart.z, int(uij.x), int(uij.y), int(uij.z),
            unit_cell_offset.x, unit_cell_offset.y, unit_cell_offset.z);


          // find which unit_cell position this offset corresponds to
          // it is possible that it does not correspond to a position in which case the
          // "unit_cell_partner" is -1
          int unit_cell_partner = -1;
          for (int k = 0; k < lattice.num_unit_cell_positions(); ++k) {
            jblib::Vec3<double> pos = lattice.unit_cell_position(k);
            if ( fabs(pos.x - unit_cell_offset.x) < distance_tolerance_
              && fabs(pos.y - unit_cell_offset.y) < distance_tolerance_
              && fabs(pos.z - unit_cell_offset.z) < distance_tolerance_ ) {
              unit_cell_partner = k;
              break;
            }
          }

          // ::output.verbose("% 8d % 8d :: % 6.6f % 6.6f % 6.6f | % 8d % 8d % 8d | % 6.6f % 6.6f % 6.6f | % 6.6f % 6.6f % 6.6f | % 6.6f % 6.6f % 6.6f | % 6.6f % 6.6f % 6.6f\n",
          //   i, unit_cell_partner, rij[0], rij[1], rij[2], int(uij[0]), int(uij[1]), int(uij[2]),
          //   r0[0], r0[1], r0[2], unit_cell_offset[0], unit_cell_offset[1], unit_cell_offset[2], real_rij[0], real_rij[1], real_rij[2], real_offset[0], real_offset[1], real_offset[2]);

          if (unit_cell_partner != -1) {
            // fast_integer_vector[3] = unit_cell_partner - i;
            // check the unit_cell partner is of the type that was specified in the exchange input file
            if (lattice.unit_cell_material_name(unit_cell_partner) != type_name_B) {
              ::output.verbose("wrong type \n");
              continue;
            }
            jblib::Vec4<int> fast_integer_vector(uij[0], uij[1], uij[2], unit_cell_partner - i);
            // ::output.write("%d %d %d %d %d % 3.6f % 3.6f % 3.6f\n", i, unit_cell_partner, fast_integer_vector[0], fast_integer_vector[1], fast_integer_vector[2], unit_cell_offset[0], unit_cell_offset[1], unit_cell_offset[2]);

            std::pair<std::set<jblib::Vec4<int> >::iterator,bool> ret;

            ret = interaction_set.insert(fast_integer_vector);

            if (ret.second == true) {
              ::output.verbose("*** % 8d [%s] % 8d [%s] :: % 8d % 8d % 8d\n", i, lattice.unit_cell_material_name(i).c_str(), fast_integer_vector[3] + i, lattice.unit_cell_material_name(fast_integer_vector[3] + i).c_str(), fast_integer_vector[0], fast_integer_vector[1], fast_integer_vector[2]);
              int_interaction_list[i].push_back(std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >(fast_integer_vector, tensor));

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
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_one_spin_energy(i);
    }
    return e_total;
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
        outfile << lattice.material(i);

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile <<  lattice.parameter()*lattice.position(i)[j];
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
        outfile << std::setw(16) << lattice.material(i);

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::fixed << lattice.parameter()*lattice.position(i)[j];
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
