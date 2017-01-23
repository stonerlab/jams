#include <set>

#include "core/globals.h"


#include "core/interactions.h"
#include "core/utils.h"
#include "core/exception.h"

namespace { //anon
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

  //---------------------------------------------------------------------
  void generate_interaction_templates(
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

  void read_interaction_data(std::ifstream &file, std::vector<interaction_t> &interaction_data) {
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
        throw general_exception("number of Jij values in exchange files must be 1 or 9, check your input on line " + std::to_string(line_number), __FILE__, __LINE__, __PRETTY_FUNCTION__);
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
  void generate_neighbour_list(const InteractionList<inode_pair_t> &interaction_template, InteractionList<Mat3> &nbr_list, double energy_cutoff) {
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

              if (tensor.max_norm() > energy_cutoff) {
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
      jams_warning("Some interactions were ignored due to the energy cutoff (%e)", energy_cutoff);
    }
    ::output.write("  total unit cell interactions: %d\n", counter);
  }




} // namespace anon

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

void generate_neighbour_list_from_file(std::ifstream &file, double energy_cutoff, bool use_symops, bool print_unfolded, InteractionList<Mat3>& neighbour_list) {
  std::vector<interaction_t> interaction_data, unfolded_interaction_data;

  InteractionList<inode_pair_t> interaction_template;

  read_interaction_data(file, interaction_data);

  generate_interaction_templates(interaction_data, unfolded_interaction_data, interaction_template, use_symops);


  if (print_unfolded) {
    std::ofstream unfolded_interaction_file(std::string(seedname+"_unfolded_exc.tsv").c_str());

    if(unfolded_interaction_file.fail()) {
      throw general_exception("failed to open unfolded interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    write_interaction_data(unfolded_interaction_file, unfolded_interaction_data);
  }

  generate_neighbour_list(interaction_template, neighbour_list, energy_cutoff);
}

//---------------------------------------------------------------------

void write_interaction_data(std::ostream &output, const std::vector<interaction_t> &data) {
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

void write_neighbour_list(std::ostream &output, const InteractionList<Mat3> &list) {
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