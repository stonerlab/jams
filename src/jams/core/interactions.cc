#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <set>

#include "jams/core/types.h"
#include "jams/core/interactions.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/exception.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/matrix.h"

using namespace std;
using libconfig::Setting;

namespace { //anon
    void apply_symops(vector<InteractionData>& interactions) {
      vector<InteractionData> symops_interaction_data;

      for (auto const &J : interactions) {
        auto new_J = J;
        auto symmetric_points = lattice->generate_symmetric_points(new_J.r_ij, 1e-5);
        for (const auto p : symmetric_points) {
          new_J.r_ij = p;
          symops_interaction_data.push_back(new_J);
        }
      }

      swap(interactions, symops_interaction_data);
    }

    string get_motif_material_name(const int unit_cell_pos) {
      return lattice->material_name(lattice->motif_atom(unit_cell_pos).material);
    }

    string get_spin_material_name(const int spin_index) {
      return lattice->material_name(lattice->atom_material_id(spin_index));
    }

    int find_motif_index(const Vec3 &offset, const double tolerance = 1e-5) {
      // find which unit_cell position this offset corresponds to
      // it is possible that it does not correspond to a position in which case the
      // -1 is returned
      for (int k = 0; k < lattice->motif_size(); ++k) {
        auto pos = lattice->motif_atom(k).pos;
        if (   abs(pos[0] - offset[0]) < tolerance
            && abs(pos[1] - offset[1]) < tolerance
            && abs(pos[2] - offset[2]) < tolerance) {
          return k;
        }
      }
      return -1;
    }

    Vec3 round_to_integer_lattice(const Vec3 &q_ij, const double tolerance = 1e-6) {
      Vec3 u_ij;
      for (int k = 0; k < 3; ++k) {
        u_ij[k] = floor(q_ij[k] + tolerance);
      }
      return u_ij;
    }

    int find_unitcell_partner(int i, Vec3 r_ij) {
      // returns -1 if no partner is found

      Vec3 p_ij_frac = lattice->motif_atom(i).pos;
      Vec3 r_ij_frac = lattice->cartesian_to_fractional(r_ij);
      // fractional interaction vector shifted by motif position
      Vec3 q_ij = r_ij_frac + p_ij_frac;
      Vec3 u_ij = round_to_integer_lattice(q_ij);

      return find_motif_index(q_ij - u_ij);
    }

    void complete_interaction_typenames_names(vector<InteractionData>& interactions) {
      // if you want to apply symmetry operations this should be done before calling this function
      apply_transform(interactions,
                      [&](InteractionData J) -> InteractionData {
                          J.type_i = get_motif_material_name(J.unit_cell_pos_i);
                          J.type_j = get_motif_material_name(J.unit_cell_pos_j);
                          return J;
                      });
    }

    void complete_interaction_unitcell_positions(vector<InteractionData>& interactions) {
      // if you want to apply symmetry operations this should be done before calling this function

      vector<InteractionData> new_data;
      new_data.reserve(interactions.size());

      for (const auto& J : interactions) {
        for (int i = 0; i < lattice->motif_size(); ++i) {
          auto new_J = J;
          // check i has the same type
          if (get_motif_material_name(i) != J.type_i) continue;
          new_J.unit_cell_pos_i = i;

          auto j = find_unitcell_partner(i, J.r_ij);

          // check if no corresponding positions exists
          if (j == -1) continue;
          new_J.unit_cell_pos_j = j;

          new_data.push_back(new_J);
        }
      }
      swap(interactions, new_data);
    }
} // namespace anon

InteractionFileDescription
discover_interaction_file_format(ifstream &file) {
  InteractionFileDescription desc;

  auto initial_pos = file.tellg();

  file.seekg(0);

  for (string line; getline(file, line);) {
    if (string_is_comment(line)) {
      continue;
    }

    auto num_cols = file_columns(line);
    if (num_cols == 6) {
      desc.dimension = InteractionType::SCALAR;
    } else if (num_cols == 14) {
      desc.dimension = InteractionType::TENSOR;
    } else {
      throw runtime_error("interaction file has an incorrect number of columns");
    }

    stringstream is(line);

    // discover type from int or string
    string s1, s2;
    is >> s1 >> s2;

    if (string_is_int(s1) != string_is_int(s2)) {
      // s1 and s2 should not have different types
      break;
    }

    if (string_is_int(s1)) {
      desc.type = InteractionFileFormat::KKR;
    } else {
      desc.type = InteractionFileFormat::JAMS;
    }

    file.seekg(initial_pos);
    return desc;
  }

  throw runtime_error("failed to discover interaction file format");
}

InteractionFileDescription
discover_interaction_setting_format(Setting& setting) {
  InteractionFileDescription desc;

  if (!(setting[0][0].getType() == setting[0][1].getType())) {
    throw runtime_error("interaction type format is incorrect");
  }

  if (setting[0][0].isNumber()) {
    desc.type = InteractionFileFormat::KKR;
  } else {
    desc.type = InteractionFileFormat::JAMS;

  }

  if (!setting[0][2].isArray()) {
    throw runtime_error("interaction vector format is incorrect");
  }

  if (!((setting[0][3].isArray() && setting[0][3].getLength() == 9) || setting[0][3].isNumber())) {
    throw runtime_error("interaction energy format is incorrect");
  }

  if (setting[0][3].isNumber()) {
    desc.dimension = InteractionType::SCALAR;
  } else {
    desc.dimension = InteractionType::TENSOR;
  }

    return desc;
}

vector<InteractionData>
interactions_from_file(ifstream &file, const InteractionFileDescription& desc) {
  assert(desc.type != InteractionFileFormat::UNDEFINED );
  assert(desc.dimension != InteractionType ::UNDEFINED);

  vector<InteractionData> interactions;
  int line_number = 0;
  for (string line; getline(file, line);) {
    if (string_is_comment(line)) {
      continue;
    }

    stringstream is(line);
    InteractionData interaction;

    if (desc.type == InteractionFileFormat::JAMS) {
      is >> interaction.type_i >> interaction.type_j;
    } else {
      is >> interaction.unit_cell_pos_i >> interaction.unit_cell_pos_j;
      // use zero based indexing
      interaction.unit_cell_pos_i--;
      interaction.unit_cell_pos_j--;
    }

    is >> interaction.r_ij[0] >> interaction.r_ij[1] >> interaction.r_ij[2];

    if (desc.dimension == InteractionType::SCALAR) {
      double J;
      is >> J;
      interaction.J_ij[0][0] = J;
      interaction.J_ij[0][1] = 0.0;
      interaction.J_ij[0][2] = 0.0;
      interaction.J_ij[1][0] = 0.0;
      interaction.J_ij[1][1] = J;
      interaction.J_ij[1][2] = 0.0;
      interaction.J_ij[2][0] = 0.0;
      interaction.J_ij[2][1] = 0.0;
      interaction.J_ij[2][2] = J;
    } else {
      is >> interaction.J_ij[0][0] >> interaction.J_ij[0][1] >> interaction.J_ij[0][2];
      is >> interaction.J_ij[1][0] >> interaction.J_ij[1][1] >> interaction.J_ij[1][2];
      is >> interaction.J_ij[2][0] >> interaction.J_ij[2][1] >> interaction.J_ij[2][2];
    }

    if (is.bad() || is.fail()) {
      throw jams::runtime_error("failed to read line " + to_string(line_number) + " of interaction file",
                                __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }

    interactions.push_back(interaction);
  }

  return interactions;
}

vector<InteractionData>
interactions_from_settings(Setting &setting, const InteractionFileDescription& desc) {
  assert(desc.type != InteractionFileFormat::UNDEFINED );
  assert(desc.dimension != InteractionType ::UNDEFINED);

  vector<InteractionData> interactions;
  if (!setting.isList()) {
    throw runtime_error("exchange settings must be a list");
  }

  for (auto i = 0; i < setting.getLength(); ++i) {
    InteractionData J;

    if (desc.type == InteractionFileFormat::KKR) {
      // use zero based indexing
      J.unit_cell_pos_i = int(setting[i][0])-1;
      J.unit_cell_pos_j = int(setting[i][1])-1;
    } else {
      J.type_i = setting[i][0].c_str();
      J.type_j = setting[i][1].c_str();
    }

    J.r_ij = {setting[i][2][0], setting[i][2][1], setting[i][2][2]};


    if (desc.dimension == InteractionType::SCALAR) {
      J.J_ij = {setting[i][3], 0.0, 0.0, 0.0, setting[i][3], 0.0, 0.0, 0.0, setting[i][3]};
    } else {
      J.J_ij = {setting[i][3][0], setting[i][3][1], setting[i][3][2],
                setting[i][3][3], setting[i][3][4], setting[i][3][5],
                setting[i][3][6], setting[i][3][7], setting[i][3][8]};
    }

    interactions.push_back(J);
  }

  return interactions;
}

void
post_process_interactions(vector<InteractionData> &interactions, const InteractionFileDescription& desc, CoordinateFormat coord_format, bool use_symops, double energy_cutoff, double radius_cutoff) {
  if (coord_format == CoordinateFormat::FRACTIONAL) {
    apply_transform(interactions, [](InteractionData J) -> InteractionData {
        J.r_ij = ::lattice->fractional_to_cartesian(J.r_ij);
        return J;
    });
  }

  if (use_symops && desc.type == InteractionFileFormat::JAMS) {
    // we apply symops before predicates, this will be more costly,
    // but means that predicates work the same regardless of whether
    // the input was given symmetrised or not
    apply_symops(interactions);
  }

  // apply any predicates
  if (energy_cutoff > 0.0) {
    apply_predicate(interactions, [&](InteractionData J) -> bool {return max_norm(J.J_ij) < energy_cutoff;});
  }

  if (energy_cutoff > 0.0) {
    apply_predicate(interactions, [&](InteractionData J) -> bool {return abs(J.r_ij) > (radius_cutoff + 1e-5);});
  }

  // complete any missing data (i.e. type names or unit cell positions

  // fill missing possible unit cell positions (if the file is JAMS format)
  if (desc.type == InteractionFileFormat::JAMS) {
    complete_interaction_unitcell_positions(interactions);
  }

  // fill missing type names (if the file is KKR format)
  if (desc.type == InteractionFileFormat::KKR) {
    complete_interaction_typenames_names(interactions);
  }
}

IntegerInteractionData
integer_interaction_from_data(const InteractionData& J) {
  Vec3 p_ij_frac = lattice->motif_atom(J.unit_cell_pos_i).pos;
  Vec3 r_ij_frac = lattice->cartesian_to_fractional(J.r_ij);
  Vec3 q_ij = r_ij_frac + p_ij_frac; // fractional interaction vector shifted by motif position
  Vec3 u_ij = round_to_integer_lattice(q_ij);
  IntegerInteractionData x;

  x.unit_cell_pos_i = J.unit_cell_pos_i;
  x.unit_cell_pos_j = J.unit_cell_pos_j;
  x.u_ij = Vec3i{int(u_ij[0]), int(u_ij[1]), int(u_ij[2])};
  x.J_ij = J.J_ij;
  x.type_i = J.type_i;
  x.type_j = J.type_j;

  return x;
}

vector<IntegerInteractionData>
generate_integer_lookup_data(vector<InteractionData> &interactions) {
  vector<IntegerInteractionData> integer_offset_data;
  integer_offset_data.reserve(interactions.size());

  for (const auto& J : interactions) {
    integer_offset_data.push_back(integer_interaction_from_data(J));
  }
  return integer_offset_data;
}

InteractionList<Mat3>
neighbour_list_from_interactions(vector<InteractionData> &interactions) {
  auto integer_template = generate_integer_lookup_data(interactions);

  InteractionList<Mat3> nbr_list(globals::num_spins);

  // loop over the translation vectors for lattice size
  for (int i = 0; i < lattice->size(0); ++i) {
    for (int j = 0; j < lattice->size(1); ++j) {
      for (int k = 0; k < lattice->size(2); ++k) {
        // loop over atoms in the interaction template
        for (const auto& I : integer_template) {
          const int m = I.unit_cell_pos_i;

          int local_site = lattice->site_index_by_unit_cell(i, j, k, m);

          Vec3i d_unit_cell = Vec3i{i, j, k} + I.u_ij;

          // check if interaction goes outside of an open boundary
          if (lattice->apply_boundary_conditions(d_unit_cell[0], d_unit_cell[1], d_unit_cell[2]) == false) {
            continue;
          }

          int nbr_site = lattice->site_index_by_unit_cell(d_unit_cell[0], d_unit_cell[1], d_unit_cell[2],
                                                          I.unit_cell_pos_j);

          // catch if the site has a different material (presumably an impurity site)
          if (get_spin_material_name(local_site) != I.type_i) {
            continue;
          }

          if (get_spin_material_name(nbr_site) != I.type_j) {
            continue;
          }

          // TODO: check there are no doubled interactions

          nbr_list.insert(local_site, nbr_site, I.J_ij);
        }
      }
    }
  }

  return nbr_list;
}

InteractionList<Mat3>
generate_neighbour_list(ifstream &file, CoordinateFormat coord_format, bool use_symops, double energy_cutoff, double radius_cutoff) {
  auto file_desc = discover_interaction_file_format(file);
  auto interactions = interactions_from_file(file, file_desc);

  post_process_interactions(interactions, file_desc, coord_format, use_symops, energy_cutoff, radius_cutoff);

  // now the interaction data should be in the same format regardless of the input
  // calculate the neighbourlist from here
  return neighbour_list_from_interactions(interactions);
}

InteractionList<Mat3>
generate_neighbour_list(Setting& setting, CoordinateFormat coord_format, bool use_symops, double energy_cutoff, double radius_cutoff) {
  auto file_desc = discover_interaction_setting_format(setting);
  auto interactions = interactions_from_settings(setting, file_desc);

  post_process_interactions(interactions, file_desc, coord_format, use_symops, energy_cutoff, radius_cutoff);

  // now the interaction data should be in the same format regardless of the input
  // calculate the neighbourlist from here
  return neighbour_list_from_interactions(interactions);
}

void
safety_check_distance_tolerance(const double &tolerance) {
  // check that no atoms in the unit cell are closer together than the tolerance

  for (auto i = 0; i < lattice->motif_size(); ++i) {
    for (auto j = i + 1; j < lattice->motif_size(); ++j) {
      const auto distance = abs(lattice->motif_atom(i).pos - lattice->motif_atom(j).pos);
      if (distance < tolerance) {
        die("Atoms %d and %d in the unit_cell are closer together (%f) than the distance_tolerance (%f).\n"
            "Check position file or relax distance_tolerance for exchange module",
            i, j, distance, tolerance);
      }
    }
  }
}

void
write_interaction_data(ostream &output, const vector<InteractionData> &data, CoordinateFormat coord_format) {
  for (auto const &interaction : data) {
    output << setw(12) << interaction.unit_cell_pos_i << "\t";
    output << setw(12) << interaction.unit_cell_pos_j << "\t";
    output << setw(12) << interaction.type_i << "\t";
    output << setw(12) << interaction.type_j << "\t";
    output << setw(12) << fixed << abs(interaction.r_ij) << "\t";
    if (coord_format == CoordinateFormat::CARTESIAN) {
      output << setw(12) << fixed << interaction.r_ij[0] << "\t";
      output << setw(12) << fixed << interaction.r_ij[1] << "\t";
      output << setw(12) << fixed << interaction.r_ij[2] << "\t";
    } else {
      auto r_ij_frac = lattice->cartesian_to_fractional(interaction.r_ij);
      output << setw(12) << fixed << r_ij_frac[0] << "\t";
      output << setw(12) << fixed << r_ij_frac[1] << "\t";
      output << setw(12) << fixed << r_ij_frac[2] << "\t";
    }
    output << setw(12) << scientific << interaction.J_ij[0][0] << "\t";
    output << setw(12) << scientific << interaction.J_ij[0][1] << "\t";
    output << setw(12) << scientific << interaction.J_ij[0][2] << "\t";
    output << setw(12) << scientific << interaction.J_ij[1][0] << "\t";
    output << setw(12) << scientific << interaction.J_ij[1][1] << "\t";
    output << setw(12) << scientific << interaction.J_ij[1][2] << "\t";
    output << setw(12) << scientific << interaction.J_ij[2][0] << "\t";
    output << setw(12) << scientific << interaction.J_ij[2][1] << "\t";
    output << setw(12) << scientific << interaction.J_ij[2][2] << endl;
  }
}
void
write_neighbour_list(ostream &output, const InteractionList<Mat3> &list) {
  for (int i = 0; i < list.size(); ++i) {
    for (auto const &nbr : list[i]) {
      int j = nbr.first;
      output << setw(12) << i << "\t";
      output << setw(12) << j << "\t";
      output << lattice->atom_position(i)[0] << "\t";
      output << lattice->atom_position(i)[1] << "\t";
      output << lattice->atom_position(i)[2] << "\t";
      output << lattice->atom_position(j)[0] << "\t";
      output << lattice->atom_position(j)[1] << "\t";
      output << lattice->atom_position(j)[2] << "\t";
      output << setw(12) << scientific << nbr.second[0][0] << "\t";
      output << setw(12) << scientific << nbr.second[0][1] << "\t";
      output << setw(12) << scientific << nbr.second[0][2] << "\t";
      output << setw(12) << scientific << nbr.second[1][0] << "\t";
      output << setw(12) << scientific << nbr.second[1][1] << "\t";
      output << setw(12) << scientific << nbr.second[1][2] << "\t";
      output << setw(12) << scientific << nbr.second[2][0] << "\t";
      output << setw(12) << scientific << nbr.second[2][1] << "\t";
      output << setw(12) << scientific << nbr.second[2][2] << "\n";
    }
    output << "\n" << endl;
  }
}
