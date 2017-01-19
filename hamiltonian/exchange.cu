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

namespace {
  int find_motif_index(const Vec3 &offset, const double tolerance = 1e-5) {
    // find which unit_cell position this offset corresponds to
    // it is possible that it does not correspond to a position in which case the
    // -1 is returned
    for (int k = 0; k < lattice.num_unit_cell_positions(); ++k) {
      Vec3 pos = lattice.unit_cell_position(k);
      if ( std::abs(pos.x - offset.x) < tolerance
        && std::abs(pos.y - offset.y) < tolerance
        && std::abs(pos.z - offset.z) < tolerance ) {
        return k;
      }
    }
    return -1;
  }

  int find_neighbour_index(const inode_t &node_i, const inode_t &node_j) {

    int n = lattice.num_unit_cell_positions();

    inode_t ivec = {(n + node_i.k + node_j.k)%n,
                         node_i.a + node_j.a,
                         node_i.b + node_j.b,
                         node_i.c + node_j.c};

    if (lattice.apply_boundary_conditions(ivec.a, ivec.b, ivec.c) == false) {
      return -1;
    }

    return lattice.site_index_by_unit_cell(ivec.a, ivec.b, ivec.c, ivec.k);
  }

  Vec3 round_to_integer_lattice(const Vec3 &q_ij, const bool is_centered_lattice = false, const double tolerance = 1e-5) {
    Vec3 u_ij;
    if (is_centered_lattice) {
      // usually nint is floor(x+0.5) but it depends on how the cell is defined :(
      // it seems using ceil is better supported with spglib
      for (int k = 0; k < 3; ++k) {
        u_ij[k] = ceil(q_ij[k]-0.5);
      }
    } else {
        // adding the distance_tolerance_ allows us to still floor properly when the precision
        // of the interaction vectors is not so good.
      for (int k = 0; k < 3; ++k) {
        u_ij[k] = floor(q_ij[k] + tolerance);
      }
    }
    return u_ij;
  }

  bool generate_inode(const int motif_index, const interaction_t &interaction, bool is_centered_lattice, inode_t &node) {

    node = {-1, -1, -1, -1};

    // only process for interactions belonging to this type
    if (lattice.unit_cell_material_name(motif_index) != interaction.type_i) {
      return false;
    }

    Vec3 p_ij_frac = lattice.unit_cell_position(motif_index);
    Vec3 r_ij_frac = lattice.cartesian_to_fractional(interaction.r_ij);

    Vec3 q_ij = r_ij_frac + p_ij_frac; // fractional interaction vector shifted by motif position
    Vec3 u_ij = round_to_integer_lattice(q_ij, is_centered_lattice);
    int nbr_motif_index = find_motif_index(q_ij - u_ij);

    // does an atom exist at the motif position
    if (nbr_motif_index == -1) {
      return false;
    }

    // is the nbr atom of the type specified
    if (lattice.unit_cell_material_name(nbr_motif_index) != interaction.type_j) {
      return false;
    }

    node = {nbr_motif_index - motif_index, int(u_ij[0]), int(u_ij[1]), int(u_ij[2])};

    return true;
  }

  void safety_check_distance_tolerance(const double &tolerance) {
    // check that no atoms in the unit cell are closer together than the tolerance
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      for (int j = i+1; j < lattice.num_unit_cell_positions(); ++j) {
        if( abs(lattice.unit_cell_position(i) - lattice.unit_cell_position(j)) < tolerance ) {
          jams_error("Atoms %d and %d in the unit_cell are closer together (%f) than the distance_tolerance (%f).\n"
                     "Check position file or relax distance_tolerance for exchange module",
                      i, j, abs(lattice.unit_cell_position(i) - lattice.unit_cell_position(j)), tolerance);
        }
      }
    }
  }
}


bool operator <(const inode_t& x, const inode_t& y) {
    return std::tie(x.k, x.a, x.b, x.c) < std::tie(y.k, y.a, y.b, y.c);
}

//---------------------------------------------------------------------

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
  }
}

//---------------------------------------------------------------------

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings) {

  //---------------------------------------------------------------------
  // read settings
  //---------------------------------------------------------------------

    is_debug_enabled_ = false;
    settings.lookupValue("debug", is_debug_enabled_);

    std::string interaction_filename = settings.lookup("exc_file").c_str();
    std::ifstream interaction_file(interaction_filename.c_str());
    if (interaction_file.fail()) {
      jams_error("failed to open interaction file %s", interaction_filename.c_str());
    }
    ::output.write("\ninteraction filename (%s)\n", interaction_filename.c_str());

    bool use_symops = true;
    settings.lookupValue("symops", use_symops);

    bool print_unfolded = false;
    settings.lookupValue("print_unfolded", print_unfolded);

    energy_cutoff_ = 1E-26;  // Joules
    settings.lookupValue("energy_cutoff", energy_cutoff_);
    ::output.write("\ninteraction energy cutoff\n  %e\n", energy_cutoff_);

    distance_tolerance_ = 1e-3; // fractional coordinate units
    settings.lookupValue("distance_tolerance", distance_tolerance_);
    ::output.write("\ndistance_tolerance\n  %e\n", distance_tolerance_);
    
    safety_check_distance_tolerance(distance_tolerance_);

    if (is_debug_enabled_) {
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


    //---------------------------------------------------------------------
    // generate interaction list
    //---------------------------------------------------------------------

    std::vector<interaction_t> interaction_data, unfolded_interaction_data;

    InteractionList<inode_pair_t> interaction_template;

    read_interaction_data(interaction_file, interaction_data);

    generate_interaction_templates(interaction_data, unfolded_interaction_data, interaction_template, use_symops);


    if (print_unfolded || is_debug_enabled_) {
      std::ofstream unfolded_interaction_file(std::string(seedname+"_unfolded_exc.tsv").c_str());

      if(unfolded_interaction_file.fail()) {
        throw general_exception("failed to open unfolded interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }

      write_interaction_data(unfolded_interaction_file, unfolded_interaction_data);
    }

    generate_neighbour_list(interaction_template, neighbour_list_);

    if (is_debug_enabled_) {
      std::ofstream debug_file("DEBUG_exchange_nbr_list.tsv");
      write_neighbour_list(debug_file, neighbour_list_);
      debug_file.close();
    }

    //---------------------------------------------------------------------
    // create sparse matrix
    //---------------------------------------------------------------------
   
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

    for (int i = 0; i < neighbour_list_.size(); ++i) {
      for (auto const &j: neighbour_list_[i]) {
        insert_interaction(i, j.first, j.second);
      }
    }

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

//---------------------------------------------------------------------

void ExchangeHamiltonian::read_interaction_data(std::ifstream &file, std::vector<interaction_t> &interaction_data) {
  int line_number = 0;

  // read the unit_cell into an array from the positions file
  for (std::string line; getline(file, line); ) {
    if(string_is_comment(line)) {
      continue;
    }

    std::stringstream   is(line);
    interaction_t interaction;

    is >> interaction.type_i >> interaction.type_j;

    if (is.bad()) {
      throw general_exception("failed to read types in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    is >> interaction.r_ij.x >> interaction.r_ij.y >> interaction.r_ij.z;

    if (is.bad()) {
      throw general_exception("failed to read interaction vector in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    const int num_info_cols = 5;

    if (file_columns(line) - num_info_cols == 1) {
      // one Jij component given - diagonal
      double J;
      is >> J;
      interaction.J_ij[0][0] =   J;  interaction.J_ij[0][1] = 0.0; interaction.J_ij[0][2] = 0.0;
      interaction.J_ij[1][0] = 0.0;  interaction.J_ij[1][1] =   J; interaction.J_ij[1][2] = 0.0;
      interaction.J_ij[2][0] = 0.0;  interaction.J_ij[2][1] = 0.0; interaction.J_ij[2][2] =   J;
    } else if (file_columns(line) - num_info_cols == 9) {
      // nine Jij components given - full tensor
      is >> interaction.J_ij[0][0] >> interaction.J_ij[0][1] >> interaction.J_ij[0][2];
      is >> interaction.J_ij[1][0] >> interaction.J_ij[1][1] >> interaction.J_ij[1][2];
      is >> interaction.J_ij[2][0] >> interaction.J_ij[2][1] >> interaction.J_ij[2][2];
    } else {
      throw general_exception("number of Jij values in exchange files must be 1 or 9, check your input on line " + line_number, __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    if (is.bad()) {
      throw general_exception("failed to read exchange tensor in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    interaction.J_ij = interaction.J_ij / kBohrMagneton;

    interaction_data.push_back(interaction);

    line_number++;
  }
}

//---------------------------------------------------------------------

void ExchangeHamiltonian::generate_neighbour_list(const InteractionList<inode_pair_t> &interaction_template, InteractionList<Mat3> &nbr_list) {
bool is_all_inserts_successful = true;
  int counter = 0;
  // loop over the translation vectors for lattice size
  for (int i = 0; i < lattice.num_unit_cells(0); ++i) {
    for (int j = 0; j < lattice.num_unit_cells(1); ++j) {
      for (int k = 0; k < lattice.num_unit_cells(2); ++k) {
        // loop over atoms in the unit_cell
        for (int m = 0; m < lattice.num_unit_cell_positions(); ++m) {
          int local_site = lattice.site_index_by_unit_cell(i, j, k, m);

          inode_t node_i = {m, i, j, k};

          std::vector<bool> is_already_interacting(globals::num_spins, false);
          is_already_interacting[local_site] = true;  // don't allow self interaction

          // loop over all possible interaction vectors
          for (auto const &pair: interaction_template[m]) {

            const inode_t node_j = pair.second.node;
            const Mat3 tensor   = pair.second.value;

            int neighbour_index = find_neighbour_index(node_i, node_j);

            // failsafe check that we only interact with any given site once through the input exchange file.
            if (is_already_interacting[neighbour_index]) {
              // jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_index, ivec.a, ivec.b, ivec.c, ivec.k);
              jams_error("Multiple interactions between spins %d and %d\n", local_site, neighbour_index);
            }

            is_already_interacting[neighbour_index] = true;

            if (tensor.max_norm() > energy_cutoff_) {
              nbr_list.insert(local_site, neighbour_index, tensor);
              counter++;
            } else {
              is_all_inserts_successful = false;
            }
          }
        }
      }
    }
  }
  if (!is_all_inserts_successful) {
    jams_warning("Some interactions were ignored due to the energy cutoff (%e)", energy_cutoff_);
  }
  ::output.write("  total unit cell interactions: %d\n", counter);
}


//---------------------------------------------------------------------

void ExchangeHamiltonian::generate_interaction_templates(
  const std::vector<interaction_t> &interaction_data,
        std::vector<interaction_t> &unfolded_interaction_data,
     InteractionList<inode_pair_t> &interactions, bool use_symops) {

  ::output.write("  reading interactions and applying symmetry operations\n");

  if (::output.is_verbose()) {
    ::output.verbose("unit cell realspace\n");
    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      jblib::Vec3<double> rij = lattice.unit_cell_position(i);
      ::output.verbose("%8d % 6.6f % 6.6f % 6.6f\n", i, rij[0], rij[1], rij[2]);
    }
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

  unfolded_interaction_data.clear();

  int interaction_counter = 0;
  for (auto const & interaction : interaction_data) {
    std::vector<interaction_t> symops_interaction_data;

    if (use_symops) {
      interaction_t symops_interaction = interaction;
      Vec3 r_ij_frac = lattice.cartesian_to_fractional(symops_interaction.r_ij); // interaction vector in fractional coordinates
      // calculate symmetric vectors based on crystal symmetry
      for (int i = 0; i < lattice.num_sym_ops(); i++) {
        symops_interaction.r_ij = lattice.fractional_to_cartesian(lattice.sym_rotation(i, r_ij_frac));
        symops_interaction_data.push_back(symops_interaction);
      }
    } else {
      symops_interaction_data.push_back(interaction);
    }

    for (int i = 0; i < lattice.num_unit_cell_positions(); ++i) {
      // calculate all unique inode vectors for (symmetric) interactions based on the current line
      std::set<inode_t> unique_interactions;
      for(auto const& symops_interaction: symops_interaction_data) {
        inode_t new_node;

        // try to generate an inode
        if (!generate_inode(i, symops_interaction, is_centered_lattice, new_node)) {
          continue;
        }

        // check if the new (unique) by insertion into std::set
        if (unique_interactions.insert(new_node).second == true) {
          // it is new (unique)
          unfolded_interaction_data.push_back(symops_interaction);
          interactions.insert(i, interaction_counter, {new_node, interaction.J_ij});
          interaction_counter++;
        }
      }
    } // for unit cell positions
  } // for interactions
  ::output.write("  total unique interactions for unitcell: %d\n", interaction_counter);
}

//---------------------------------------------------------------------

void ExchangeHamiltonian::write_interaction_data(std::ostream &output, const std::vector<interaction_t> &data) {
  for (auto const & interaction : data) {
    output << std::setw(12) << interaction.type_i << "\t";
    output << std::setw(12) << interaction.type_j << "\t";
    output << std::setw(12) << std::fixed << interaction.r_ij.x << "\t";
    output << std::setw(12) << std::fixed << interaction.r_ij.y << "\t";
    output << std::setw(12) << std::fixed << interaction.r_ij.z << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[0][0] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[0][1] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[0][2] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[1][0] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[1][1] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[1][2] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[2][0] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[2][1] * kBohrMagneton << "\t";
    output << std::setw(12) << std::scientific << interaction.J_ij[2][2] * kBohrMagneton << std::endl;
  }
}

//---------------------------------------------------------------------

void ExchangeHamiltonian::write_neighbour_list(std::ostream &output, const InteractionList<Mat3> &list) {
  for (int i = 0; i < list.size(); ++i) {
    for (auto const & nbr : list[i]) {
      int j = nbr.first;
      output << std::setw(12) << i << "\t";
      output << std::setw(12) << j << "\t";
      output << lattice.atom_position(i).x << "\t";
      output << lattice.atom_position(i).y << "\t";
      output << lattice.atom_position(i).z << "\t";
      output << lattice.atom_position(j).x << "\t";
      output << lattice.atom_position(j).y << "\t";
      output << lattice.atom_position(j).z << "\t";
      output << std::setw(12) << std::scientific << nbr.second[0][0] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[0][1] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[0][2] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[1][0] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[1][1] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[1][2] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[2][0] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[2][1] * kBohrMagneton << "\t";
      output << std::setw(12) << std::scientific << nbr.second[2][2] * kBohrMagneton << "\n";
    }
    output << "\n" << std::endl;
  }
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

//---------------------------------------------------------------------

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

//---------------------------------------------------------------------

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
