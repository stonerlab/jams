#include <cmath> 
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <set>

#include "jams/core/interactions.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/exception.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/matrix.h"

using namespace std;

InteractionFileFormat exchange_file_format_from_string(std::string s) {
  if (capitalize(s) == "JAMS") {
    return InteractionFileFormat::JAMS;
  }

  if (capitalize(s) == "KKR") {
    return InteractionFileFormat::KKR;
  }

  throw std::runtime_error("Unknown exchange file format");
}

CoordinateFormat coordinate_format_from_string(std::string s) {
  if (capitalize(s) == "CART" || capitalize(s) == "CARTESIAN") {
    return CoordinateFormat::Cartesian;
  }

  if (capitalize(s) == "FRAC" || capitalize(s) == "FRACTIONAL") {
    return CoordinateFormat::Fractional;
  }

  throw std::runtime_error("Unknown coordinate format");
}


namespace { //anon
  int find_motif_index(const Vec3 &offset, const double tolerance = 1e-5) {
    // find which unit_cell position this offset corresponds to
    // it is possible that it does not correspond to a position in which case the
    // -1 is returned
    for (int k = 0; k < lattice->motif_size(); ++k) {
      auto pos = lattice->motif_atom(k).pos;
      if ( std::abs(pos[0] - offset[0]) < tolerance
        && std::abs(pos[1] - offset[1]) < tolerance
        && std::abs(pos[2] - offset[2]) < tolerance ) {
        return k;
      }
    }
    return -1;
  }

  int find_neighbour_index(const inode_t &node_i, const inode_t &node_j) {

    int n = lattice->motif_size();

    inode_t ivec = {(n + node_i.k + node_j.k)%n,
                         node_i.a + node_j.a,
                         node_i.b + node_j.b,
                         node_i.c + node_j.c};

    if (lattice->apply_boundary_conditions(ivec.a, ivec.b, ivec.c) == false) {
      return -1;
    }

    return lattice->site_index_by_unit_cell(ivec.a, ivec.b, ivec.c, ivec.k);
  }

  Vec3 round_to_integer_lattice(const Vec3 &q_ij, const double tolerance=1e-6) {
    Vec3 u_ij;
      for (int k = 0; k < 3; ++k) {
        u_ij[k] = floor(q_ij[k] + tolerance);
      }
    return u_ij;
  }

    bool generate_inode(const unitcell_interaction_t &interaction, inode_t &node) {

      node = {-1, -1, -1, -1};

      Vec3 p_ij_frac = lattice->motif_atom(interaction.pos_i).pos;
      Vec3 r_ij_frac = lattice->cartesian_to_fractional(interaction.r_ij);

      Vec3 q_ij = r_ij_frac + p_ij_frac; // fractional interaction vector shifted by motif position
      Vec3 u_ij = round_to_integer_lattice(q_ij);

      Vec3 dr = q_ij - u_ij;

      int nbr_motif_index = find_motif_index(dr);

      if (nbr_motif_index == -1) {
        throw std::runtime_error("Inconsistency in interaction template (no motif position found): "
                                 + std::to_string(interaction.pos_i) + " "
                                 + std::to_string(interaction.pos_j) + " "
                                 + std::to_string(interaction.r_ij[0])  + " " + std::to_string(interaction.r_ij[1]) + " " + std::to_string(interaction.r_ij[2]));
      }

      if (nbr_motif_index != interaction.pos_j) {
        throw std::runtime_error("Inconsistency in interaction template (incorrect motif position: " + std::to_string(nbr_motif_index) + ")"
                                 + std::to_string(interaction.pos_i) + " "
                                 + std::to_string(interaction.pos_j) + " "
                                 + std::to_string(interaction.r_ij[0])  + " " + std::to_string(interaction.r_ij[1]) + " " + std::to_string(interaction.r_ij[2]));
      }

      node = {interaction.pos_j - interaction.pos_i, int(u_ij[0]), int(u_ij[1]), int(u_ij[2])};

      return true;
    }

  bool generate_inode(const int motif_index, const typename_interaction_t &interaction, inode_t &node) {

    node = {-1, -1, -1, -1};

    Vec3 p_ij_frac = lattice->motif_atom(motif_index).pos;
    Vec3 r_ij_frac = lattice->cartesian_to_fractional(interaction.r_ij);

    Vec3 q_ij = r_ij_frac + p_ij_frac; // fractional interaction vector shifted by motif position
    Vec3 u_ij = round_to_integer_lattice(q_ij);
    int nbr_motif_index = find_motif_index(q_ij - u_ij);

    // does an atom exist at the motif position
    if (nbr_motif_index == -1) {
      return false;
    }

    node = {nbr_motif_index - motif_index, int(u_ij[0]), int(u_ij[1]), int(u_ij[2])};

    return true;
  }

  //---------------------------------------------------------------------
  void generate_interaction_templates(
    const std::vector<typename_interaction_t> &interaction_data,
          std::vector<typename_interaction_t> &unfolded_interaction_data,
       InteractionList<inode_pair_t> &interactions, bool use_symops) {

    cout << "  reading interactions and applying symmetry operations\n";

    unfolded_interaction_data.clear();

    int interaction_counter = 0;
    for (auto const & interaction : interaction_data) {
      std::vector<typename_interaction_t> symops_interaction_data;

      if (use_symops) {
        auto symops_interaction = interaction;
        auto symmetric_points = lattice->generate_symmetric_points(symops_interaction.r_ij, 1e-6);
        for (const auto p : symmetric_points) {
          symops_interaction.r_ij = p;
          symops_interaction_data.push_back(symops_interaction);
        }
      } else {
        symops_interaction_data.push_back(interaction);
      }

      for (int i = 0; i < lattice->motif_size(); ++i) {
        // calculate all unique inode vectors for (symmetric) interactions based on the current line
        std::set<inode_t> unique_interactions;
        for(auto const& symops_interaction: symops_interaction_data) {
          inode_t new_node;

          // try to generate an inode
          if (!generate_inode(i, symops_interaction, new_node)) {
            continue;
          }

          // check if the new (unique) by insertion into std::set
          if (unique_interactions.insert(new_node).second == true) {
            // it is new (unique)
            unfolded_interaction_data.push_back(symops_interaction);
            interactions.insert(i, interaction_counter, {new_node, interaction.type_i, interaction.type_j, interaction.J_ij});
            interaction_counter++;
          }
        }
      } // for unit cell positions
    } // for interactions
    cout << "  total unique interactions for unitcell " << interaction_counter << "\n";
  }

  //---------------------------------------------------------------------
    void read_jams_format_interaction_data(std::ifstream &file, std::vector<typename_interaction_t> &interaction_data, CoordinateFormat coord_format, double energy_cutoff, double radius_cutoff) {
    int line_number = 0;

    unsigned energy_cutoff_counter = 0;
    unsigned radius_cutoff_counter = 0;

    // read the unit_cell into an array from the positions file
    for (std::string line; getline(file, line); ) {
      if(string_is_comment(line)) {
        continue;
      }

      std::stringstream   is(line);
      typename_interaction_t interaction;

      is >> interaction.type_i >> interaction.type_j;

      if(is.fail()) {
        throw std::runtime_error("Interaction file unitcell type format is incorrect for JAMS format");
      }

      if (is.bad()) {
        throw general_exception("failed to read types in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }

      is >> interaction.r_ij[0] >> interaction.r_ij[1] >> interaction.r_ij[2];

      if (coord_format == CoordinateFormat::Fractional) {
        interaction.r_ij = ::lattice->fractional_to_cartesian(interaction.r_ij);
      }

      if (is.bad()) {
        throw general_exception("failed to read interaction vector in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }

      const int num_info_cols = 5;

      if (file_columns(line) - num_info_cols == 1) {
        // one Jij component given - diagonal
        double J;
        is >> J;

        if(is.fail()) {
          throw std::runtime_error("Interaction file Jij format is incorrect for JAMS format");
        }

        interaction.J_ij[0][0] =   J;  interaction.J_ij[0][1] = 0.0; interaction.J_ij[0][2] = 0.0;
        interaction.J_ij[1][0] = 0.0;  interaction.J_ij[1][1] =   J; interaction.J_ij[1][2] = 0.0;
        interaction.J_ij[2][0] = 0.0;  interaction.J_ij[2][1] = 0.0; interaction.J_ij[2][2] =   J;
      } else if (file_columns(line) - num_info_cols == 9) {
        // nine Jij components given - full tensor
        is >> interaction.J_ij[0][0] >> interaction.J_ij[0][1] >> interaction.J_ij[0][2];
        is >> interaction.J_ij[1][0] >> interaction.J_ij[1][1] >> interaction.J_ij[1][2];
        is >> interaction.J_ij[2][0] >> interaction.J_ij[2][1] >> interaction.J_ij[2][2];

        if(is.fail()) {
          throw std::runtime_error("Interaction file Jij format is incorrect for JAMS format");
        }
      } else {
        throw general_exception("number of Jij values in exchange files must be 1 or 9, check your input on line " + std::to_string(line_number), __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }

      if (is.bad()) {
        throw general_exception("failed to read exchange tensor in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }

      interaction.J_ij = interaction.J_ij / kBohrMagneton;

      if (max_norm(interaction.J_ij) < energy_cutoff) {
        energy_cutoff_counter++;
        continue;
      }

      if (abs(interaction.r_ij) > (radius_cutoff + 1e-5)) {
        radius_cutoff_counter++;
        continue;
      }

      interaction_data.push_back(interaction);

      line_number++;
    }

    if (radius_cutoff_counter != 0) {
      jams_warning("%u interactions were ignored due to the radius cutoff (%e)", radius_cutoff_counter, radius_cutoff);
    }

    if (energy_cutoff_counter != 0) {
      jams_warning("%u interactions were ignored due to the energy cutoff (%e)", energy_cutoff_counter, energy_cutoff);
    }

  }

    void read_kkr_format_interaction_data(std::ifstream &file, InteractionList<inode_pair_t> &interactions, CoordinateFormat coord_format, double energy_cutoff, double radius_cutoff) {
      int line_number = 0;
      int interaction_counter = 0;

      unsigned energy_cutoff_counter = 0;
      unsigned radius_cutoff_counter = 0;

      bool is_centered_lattice = false;

      for (auto i = 0; i < lattice->motif_size(); ++i) {
        const auto& atom = lattice->motif_atom(i);
        if (atom.pos[0] < 0.0 || atom.pos[1] < 0.0 || atom.pos[2] < 0.0) {
          is_centered_lattice = true;
          jams_warning("Centered lattice is detected. Make sure you know what you are doing!");
          break;
        }
      }

      // read the unit_cell into an array from the positions file
      for (std::string line; getline(file, line); ) {
        if(string_is_comment(line)) {
          continue;
        }

        inode_t inode;

        std::stringstream   is(line);
        unitcell_interaction_t interaction;

        is >> interaction.pos_i >> interaction.pos_j;

        if(is.fail()) {
          throw std::runtime_error("Interaction file unitcell number format is incorrect for KKR format");
        }

        // use zero based indexing
        interaction.pos_i--;
        interaction.pos_j--;

        if (is.bad()) {
          throw general_exception("failed to read types in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
        }

        is >> interaction.r_ij[0] >> interaction.r_ij[1] >> interaction.r_ij[2];

        if (coord_format == CoordinateFormat::Fractional) {
          interaction.r_ij = ::lattice->fractional_to_cartesian(interaction.r_ij);
        }

        if (is.bad()) {
          throw general_exception("failed to read interaction vector in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
        }

        const int num_info_cols = 5;

        if (file_columns(line) - num_info_cols == 1) {
          // one Jij component given - diagonal
          double J;
          is >> J;

          if(is.fail()) {
            throw std::runtime_error("Interaction file Jij format is incorrect for KKR format");
          }

          interaction.J_ij[0][0] =   J;  interaction.J_ij[0][1] = 0.0; interaction.J_ij[0][2] = 0.0;
          interaction.J_ij[1][0] = 0.0;  interaction.J_ij[1][1] =   J; interaction.J_ij[1][2] = 0.0;
          interaction.J_ij[2][0] = 0.0;  interaction.J_ij[2][1] = 0.0; interaction.J_ij[2][2] =   J;
        } else if (file_columns(line) - num_info_cols == 9) {
          // nine Jij components given - full tensor
          is >> interaction.J_ij[0][0] >> interaction.J_ij[0][1] >> interaction.J_ij[0][2];
          is >> interaction.J_ij[1][0] >> interaction.J_ij[1][1] >> interaction.J_ij[1][2];
          is >> interaction.J_ij[2][0] >> interaction.J_ij[2][1] >> interaction.J_ij[2][2];

          if(is.fail()) {
            throw std::runtime_error("Interaction file Jij format is incorrect for KKR format");
          }
        } else {
          throw general_exception("number of Jij values in exchange files must be 1 or 9, check your input on line " + std::to_string(line_number), __FILE__, __LINE__, __PRETTY_FUNCTION__);
        }

        if (is.bad()) {
          throw general_exception("failed to read exchange tensor in line " + std::to_string(line_number) + " of interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
        }

        if (max_norm(interaction.J_ij) < energy_cutoff) {
          energy_cutoff_counter++;
          continue;
        }

        if (abs(interaction.r_ij) > (radius_cutoff + 1e-5)) {
          radius_cutoff_counter++;
          continue;
        }

        interaction.J_ij = interaction.J_ij / kBohrMagneton;


        if (!generate_inode(interaction, inode)) {
//          continue;
          throw std::runtime_error("Inconsistency in the KKR exchange templates");
        }

        auto material_i = lattice->material(interaction.pos_i).name;
        auto material_j = lattice->material(interaction.pos_j).name;

        interactions.insert(interaction.pos_i, interaction_counter, {inode, material_i, material_j, interaction.J_ij});

        interaction_counter++;
        line_number++;
      }

      if (radius_cutoff_counter != 0) {
        jams_warning("%u interactions were ignored due to the radius cutoff (%e)", radius_cutoff_counter, radius_cutoff);
      }

      if (energy_cutoff_counter != 0) {
        jams_warning("%u interactions were ignored due to the energy cutoff (%e)", energy_cutoff_counter, energy_cutoff);
      }
    }

  //---------------------------------------------------------------------
  void
  generate_neighbour_list(const InteractionList<inode_pair_t> &interaction_template, InteractionList<Mat3> &nbr_list) {
    unsigned interaction_counter = 0;
    // loop over the translation vectors for lattice size
    for (int i = 0; i < lattice->size(0); ++i) {
      for (int j = 0; j < lattice->size(1); ++j) {
        for (int k = 0; k < lattice->size(2); ++k) {
          // loop over atoms in the interaction template
          // we use interaction_template.size() because if some atoms have no interaction then this can be smaller than
          // the number of atoms in the unit cell
          for (int m = 0; m < interaction_template.size(); ++m) {

            if (interaction_template[m].size() == 0) {
              continue;
            }

            int local_site = lattice->site_index_by_unit_cell(i, j, k, m);

            inode_t node_i = {m, i, j, k};

            std::vector<bool> is_already_interacting(globals::num_spins, false);
            is_already_interacting[local_site] = true;  // don't allow self interaction

            // loop over all possible interaction vectors
            for (auto const pair: interaction_template[m]) {

              const inode_t node_j = pair.second.node;
              const auto material_i = pair.second.type_i;
              const auto material_j = pair.second.type_j;
              const Mat3 tensor = pair.second.value;


              int neighbour_index = find_neighbour_index(node_i, node_j);

              if (neighbour_index == -1) {
                // no neighbour found
                continue;
              }

              // catch if the site has a different material (presumably an impurity site)
              if (lattice->material(local_site).name != material_i) {
                continue;
              }

              if (lattice->material(neighbour_index).name != material_j) {
                continue;
              }

              // failsafe check that we only interact with any given site once through the input exchange file.
              if (is_already_interacting[neighbour_index]) {
                // jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_index, ivec.a, ivec.b, ivec.c, ivec.k);
                jams_error("Multiple interactions between spins %d and %d\n", local_site, neighbour_index);
              }

              is_already_interacting[neighbour_index] = true;

              nbr_list.insert(local_site, neighbour_index, tensor);
              interaction_counter++;
            }
          }
        }
      }
    }

    cout << "  total system interactions: " <<  interaction_counter << "\n";
  }




} // namespace anon

void safety_check_distance_tolerance(const double &tolerance) {
  // check that no atoms in the unit cell are closer together than the tolerance

  for (auto i = 0; i < lattice->motif_size(); ++i) {
    for (auto j = i+1; j < lattice->motif_size(); ++j) {
      const auto distance = abs(lattice->motif_atom(i).pos - lattice->motif_atom(j).pos);
      if(distance < tolerance) {
        jams_error("Atoms %d and %d in the unit_cell are closer together (%f) than the distance_tolerance (%f).\n"
                   "Check position file or relax distance_tolerance for exchange module",
                    i, j, distance, tolerance);
      }
    }
  }
}

void generate_neighbour_list_from_file(std::ifstream &file, InteractionFileFormat file_format, CoordinateFormat coord_format, double energy_cutoff,
                                       double radius_cutoff, bool use_symops, bool print_unfolded,
                                       InteractionList<Mat3> &neighbour_list) {
  std::vector<typename_interaction_t> interaction_data, unfolded_interaction_data;

  InteractionList<inode_pair_t> interaction_template;

  if (file_format == InteractionFileFormat::JAMS) {
    read_jams_format_interaction_data(file, interaction_data, coord_format, energy_cutoff, radius_cutoff);
    generate_interaction_templates(interaction_data, unfolded_interaction_data, interaction_template, use_symops);

    if (print_unfolded) {
      std::ofstream unfolded_interaction_file(std::string(seedname+"_unfolded_exc.tsv").c_str());

      if(unfolded_interaction_file.fail()) {
        throw general_exception("failed to open unfolded interaction file", __FILE__, __LINE__, __PRETTY_FUNCTION__);
      }

      write_interaction_data(unfolded_interaction_file, unfolded_interaction_data, coord_format);
    }
  } else if (file_format == InteractionFileFormat::KKR) {
    read_kkr_format_interaction_data(file, interaction_template, coord_format, energy_cutoff, radius_cutoff);
  }


  cout << "  num unit cell interactions per position:\n";
  for (auto i = 0; i < interaction_template.size(); ++i) {
    cout << "    " << i << ": " << interaction_template.num_interactions(i) << "\n";
  }

  generate_neighbour_list(interaction_template, neighbour_list);
}

//---------------------------------------------------------------------

void write_interaction_data(std::ostream &output, const std::vector<typename_interaction_t> &data,
                            CoordinateFormat coord_format) {
  for (auto const & interaction : data) {
    output << std::setw(12) << interaction.type_i << "\t";
    output << std::setw(12) << interaction.type_j << "\t";
    if (coord_format == CoordinateFormat::Cartesian) {
      output << std::setw(12) << std::fixed << interaction.r_ij[0] << "\t";
      output << std::setw(12) << std::fixed << interaction.r_ij[1] << "\t";
      output << std::setw(12) << std::fixed << interaction.r_ij[2] << "\t";
    } else {
      auto r_ij_frac = lattice->cartesian_to_fractional(interaction.r_ij);
      output << std::setw(12) << std::fixed << r_ij_frac[0] << "\t";
      output << std::setw(12) << std::fixed << r_ij_frac[1] << "\t";
      output << std::setw(12) << std::fixed << r_ij_frac[2] << "\t";
    }
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
      output << lattice->atom_position(i)[0] << "\t";
      output << lattice->atom_position(i)[1] << "\t";
      output << lattice->atom_position(i)[2] << "\t";
      output << lattice->atom_position(j)[0] << "\t";
      output << lattice->atom_position(j)[1] << "\t";
      output << lattice->atom_position(j)[2] << "\t";
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
