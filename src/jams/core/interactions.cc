#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <numeric>
#include <optional>
#include <jams/helpers/output.h>

#include "jams/core/types.h"
#include "jams/core/interactions.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/exception.h"


void neighbour_list_checks(const jams::InteractionList<Mat3, 2>& list, const std::vector<InteractionChecks>& checks);

namespace { //anon
    void apply_symops(std::vector<InteractionData>& interactions) {
      std::vector<InteractionData> symops_interaction_data;

      for (auto const &J : interactions) {
        auto new_J = J;
        auto symmetric_points = globals::lattice->generate_symmetric_points(J.basis_site_i, new_J.interaction_vector_cart, jams::defaults::lattice_tolerance);
        for (const auto p : symmetric_points) {
          new_J.interaction_vector_cart = p;
          symops_interaction_data.push_back(new_J);
        }
      }

      swap(interactions, symops_interaction_data);
    }

    std::string basis_site_material_name(const int unit_cell_pos) {
      return globals::lattice->material_name(
          globals::lattice->basis_site_atom(unit_cell_pos).material_index);
    }

    std::optional<int> find_basis_site_index(const Vec3 &offset, const double tolerance) {
      // find which basis site this offset corresponds to. It is possible that it does not correspond to a position in
      // which case the optional return is falsey.
      for (int k = 0; k < globals::lattice->num_basis_sites(); ++k) {
        auto pos = globals::lattice->basis_site_atom(k).position_frac;
        if (approximately_equal(pos, offset, tolerance)) {
          return k;
        }
      }
      return std::nullopt;
    }

    /// Returns the integer lattice translation vector T of an arbitrary vector r accounting for difficulties
    /// in the precision at the edges and corners of the cell.
    Vec3 lattice_translation_vector(const Vec3& r_frac, const double tolerance) {
      // If we are very close to the origin or edge of a cell then rounding with floor() to find the cell translation
      // vector can be tricky because smaller errors due to floating point precision (not least from the user input)
      // could put us in the wrong cell. Therefore we first check for the case that we are very close (within
      // tolerance) of a cell origin, in which case we round to that origin. Otherwise we use
      // floor() in the usual way.
      Vec3 T;
      for (auto n = 0; n < 3; ++n) {
        double nearest_integer = std::nearbyint(r_frac[n]);
        double floored_value = std::floor(r_frac[n]);

        if (approximately_zero(r_frac[n] - nearest_integer, tolerance)) {
          T[n] = nearest_integer;
        } else {
          T[n] = floored_value;
        }
      }
      return T;
    }

    std::optional<int> find_unitcell_partner(int i, Vec3 r_ij, double tolerance) {
      // returns -1 if no partner is found

      Vec3 p_i_frac = globals::lattice->basis_site_atom(i).position_frac;
      Vec3 r_ij_frac = globals::lattice->cartesian_to_fractional(r_ij);
      // fractional interaction vector shifted by motif position
      Vec3 q_ij = r_ij_frac + p_i_frac;

      return find_basis_site_index(q_ij - lattice_translation_vector(q_ij, tolerance), tolerance);
    }

    void complete_interaction_typenames_names(std::vector<InteractionData>& interactions) {
      apply_transform(interactions,
                      [&](InteractionData J) -> InteractionData {
                          J.type_i = basis_site_material_name(J.basis_site_i);
                          J.type_j = basis_site_material_name(J.basis_site_j);
                          return J;
                      });
    }

    void complete_interaction_unitcell_positions(std::vector<InteractionData>& interactions, double distance_tolerance) {
      std::vector<InteractionData> new_data;
      new_data.reserve(interactions.size());

      for (const auto& J : interactions) {
        for (int i = 0; i < globals::lattice->num_basis_sites(); ++i) {
          auto new_J = J;
          // check i has the same type
          if (basis_site_material_name(i) != J.type_i) continue;
          new_J.basis_site_i = i;

          auto basis_site_partner = find_unitcell_partner(i, J.interaction_vector_cart, distance_tolerance);
          // not such position exists
          if (!basis_site_partner) continue;

          int j = *basis_site_partner;

          // check j has the same type
          if (basis_site_material_name(j) != J.type_j) continue;

          new_J.basis_site_j = j;

          new_data.push_back(new_J);
        }
      }
      swap(interactions, new_data);
    }
} // namespace anon

InteractionFileDescription
discover_interaction_file_format(std::ifstream &file) {
  InteractionFileDescription desc;

  auto initial_pos = file.tellg();

  file.seekg(0);

  for (std::string line; getline(file, line);) {
    if (string_is_comment(line)) {
      continue;
    }

    auto num_cols = file_columns(line);
    if (num_cols == 6) {
      desc.dimension = InteractionType::SCALAR;
    } else if (num_cols == 14) {
      desc.dimension = InteractionType::TENSOR;
    } else {
      throw std::runtime_error("interaction file has an incorrect number of columns");
    }

    std::stringstream is(line);

    // discover type from int or string
    std::string s1, s2;
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

  throw std::runtime_error("failed to discover interaction file format");
}

InteractionFileDescription
discover_interaction_setting_format(libconfig::Setting& setting) {
  InteractionFileDescription desc;

  if (!(setting[0][0].getType() == setting[0][1].getType())) {
    throw std::runtime_error("interaction type format is incorrect");
  }

  if (setting[0][0].isNumber()) {
    desc.type = InteractionFileFormat::KKR;
  } else {
    desc.type = InteractionFileFormat::JAMS;

  }

  if (!setting[0][2].isArray()) {
    throw std::runtime_error("interaction vector format is incorrect");
  }

  if (!((setting[0][3].isArray() && setting[0][3].getLength() == 9) || setting[0][3].isNumber())) {
    throw std::runtime_error("interaction energy format is incorrect");
  }

  if (setting[0][3].isNumber()) {
    desc.dimension = InteractionType::SCALAR;
  } else {
    desc.dimension = InteractionType::TENSOR;
  }

    return desc;
}

std::vector<InteractionData>
interactions_from_file(std::ifstream &file, const InteractionFileDescription& desc) {
  assert(desc.type != InteractionFileFormat::UNDEFINED );
  assert(desc.dimension != InteractionType ::UNDEFINED);

  std::vector<InteractionData> interactions;
  int line_number = 0;
  for (std::string line; getline(file, line);) {
    if (string_is_comment(line)) {
      continue;
    }

    std::stringstream is(line);
    InteractionData interaction;

    if (desc.type == InteractionFileFormat::JAMS) {
      is >> interaction.type_i >> interaction.type_j;
    } else {
      is >> interaction.basis_site_i >> interaction.basis_site_j;
      // use zero based indexing
      interaction.basis_site_i--;
      interaction.basis_site_j--;
    }

    is >> interaction.interaction_vector_cart[0] >> interaction.interaction_vector_cart[1] >> interaction.interaction_vector_cart[2];

    if (desc.dimension == InteractionType::SCALAR) {
      double J;
      is >> J;
      interaction.interaction_value_tensor = J * kIdentityMat3;
    } else {
      for (auto i : {0,1,2}) {
        for (auto j : {0,1,2}) {
          is >> interaction.interaction_value_tensor[i][j];
        }
      }
    }

    if (is.bad() || is.fail()) {
      throw std::runtime_error("failed to read line " + std::to_string(line_number) + " of interaction file");
    }

    interactions.push_back(interaction);
  }

  return interactions;
}

std::vector<InteractionData>
interactions_from_settings(libconfig::Setting &setting, const InteractionFileDescription& desc) {
  assert(desc.type != InteractionFileFormat::UNDEFINED );
  assert(desc.dimension != InteractionType ::UNDEFINED);

  std::vector<InteractionData> interactions;
  if (!setting.isList()) {
    throw std::runtime_error("exchange settings must be a list");
  }

  for (auto i = 0; i < setting.getLength(); ++i) {
    InteractionData J;

    if (desc.type == InteractionFileFormat::KKR) {
      // use zero based indexing
      J.basis_site_i = int(setting[i][0])-1;
      J.basis_site_j = int(setting[i][1])-1;
    } else {
      J.type_i = setting[i][0].c_str();
      J.type_j = setting[i][1].c_str();
    }

    J.interaction_vector_cart = {setting[i][2][0], setting[i][2][1], setting[i][2][2]};

    if (desc.dimension == InteractionType::SCALAR) {
      J.interaction_value_tensor = double(setting[i][3]) * kIdentityMat3;
    } else {
      J.interaction_value_tensor = {setting[i][3][0], setting[i][3][1], setting[i][3][2],
                                    setting[i][3][3], setting[i][3][4], setting[i][3][5],
                                    setting[i][3][6], setting[i][3][7], setting[i][3][8]};
    }

    interactions.push_back(J);
  }

  return interactions;
}

void
post_process_interactions(std::vector<InteractionData> &interactions, const InteractionFileDescription& desc, CoordinateFormat coord_format, bool use_symops, double energy_cutoff, double radius_cutoff, double distance_tolerance) {
  if (coord_format == CoordinateFormat::FRACTIONAL) {
    apply_transform(interactions, [](InteractionData J) -> InteractionData {
        J.interaction_vector_cart = ::globals::lattice->fractional_to_cartesian(J.interaction_vector_cart);
        return J;
    });
  }

  // Complete any missing data (i.e. type names or unit cell positions). This must be done BEFORE doing any symmetry
  // operations so that the unit cell positions are known.

  // fill missing possible unit cell positions (if the file is JAMS format)
  if (desc.type == InteractionFileFormat::JAMS) {
    complete_interaction_unitcell_positions(interactions, distance_tolerance);
  }

  // fill missing type names (if the file is KKR format)
  if (desc.type == InteractionFileFormat::KKR) {
    complete_interaction_typenames_names(interactions);
  }


  if (use_symops) {
    // we apply symops before predicates, this will be more costly,
    // but means that predicates work the same regardless of whether
    // the input was given symmetrised or not
    apply_symops(interactions);
  }

  // apply any predicates
  if (energy_cutoff > 0.0) {
    apply_predicate(interactions, [&](InteractionData J) -> bool {
      return definately_less_than(max_abs(J.interaction_value_tensor), energy_cutoff, DBL_EPSILON);});
  }

  if (radius_cutoff > 0.0) {
    apply_predicate(interactions, [&](InteractionData J) -> bool {
      return definately_greater_than(norm(J.interaction_vector_cart), radius_cutoff, jams::defaults::lattice_tolerance);});
  }

  // calculate the lattice translation vectors
  apply_transform(interactions, [&](InteractionData J) -> InteractionData {
    Vec3 p_i_frac = globals::lattice->basis_site_atom(J.basis_site_i).position_frac;
    Vec3 p_j_frac = globals::lattice->basis_site_atom(J.basis_site_j).position_frac;
    Vec3 r_ij_frac = globals::lattice->cartesian_to_fractional(J.interaction_vector_cart);
    Vec3 T = lattice_translation_vector(r_ij_frac + p_i_frac - p_j_frac, distance_tolerance);

    // If r_ij_frac + p_i_frac - p_j_frac is not a cell translation vector then there is a problem with the inputted
    // exchange vectors.
    assert(approximately_zero(T - (r_ij_frac + p_i_frac - p_j_frac), distance_tolerance));

    J.lattice_translation_vector = {int(T[0]), int(T[1]), int(T[2])};
    return J;
  });

}

jams::InteractionList<Mat3, 2>
neighbour_list_from_interactions(std::vector<InteractionData> &interactions) {
  jams::InteractionList<Mat3, 2> nbr_list;

  // loop over the translation vectors for lattice size
  for (int i = 0; i < globals::lattice->size(0); ++i) {
    for (int j = 0; j < globals::lattice->size(1); ++j) {
      for (int k = 0; k < globals::lattice->size(2); ++k) {
        // loop over atoms in the interaction template
        for (const auto& I : interactions) {
          const int m = I.basis_site_i;

          int local_site = globals::lattice->site_index_by_unit_cell(i, j, k, m);

          Vec3i d_unit_cell = Vec3i{i, j, k} + I.lattice_translation_vector;

          // check if interaction goes outside of an open boundary
          if (globals::lattice->apply_boundary_conditions(d_unit_cell[0], d_unit_cell[1], d_unit_cell[2]) == false) {
            continue;
          }

          int nbr_site = globals::lattice->site_index_by_unit_cell(d_unit_cell[0], d_unit_cell[1], d_unit_cell[2],
                                                                   I.basis_site_j);

          if (nbr_list.contains({local_site, nbr_site})) {
            auto r_ij = globals::lattice->displacement(nbr_site, local_site);
            throw std::runtime_error(
                "Multiple interactions for sites " + std::to_string(local_site) + " and " + std::to_string(nbr_site) + "\n"
                + "i: motif pos: " + std::to_string(m + 1) + " unit cell indices: " + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + "\n"
                + "j: motif pos: " + std::to_string(I.basis_site_j + 1) + " unit cell indices: " + std::to_string(I.lattice_translation_vector[0]) + ", " + std::to_string(I.lattice_translation_vector[1]) + ", " + std::to_string(I.lattice_translation_vector[2]) + "\n"
                + "interaction_vector_cart: " + std::to_string(r_ij[0]) + ", " + std::to_string(r_ij[1]) + ", " + std::to_string(r_ij[2])
            );
          }

          // catch if the site has a different material (presumably an impurity site)
          if (globals::lattice->lattice_site_material_name(local_site) != I.type_i || globals::lattice->lattice_site_material_name(nbr_site) != I.type_j) {
            continue;
          }

          nbr_list.insert({local_site, nbr_site}, I.interaction_value_tensor);
        }
      }
    }
  }

  return nbr_list;
}

jams::InteractionList<Mat3, 2>
generate_neighbour_list(std::ifstream &file,
                        CoordinateFormat coord_format,
                        bool use_symops,
                        double energy_cutoff,
                        double radius_cutoff,
                        double distance_tolerance,
                        std::vector<InteractionChecks> checks) {
  auto file_desc = discover_interaction_file_format(file);
  auto interactions = interactions_from_file(file, file_desc);

  post_process_interactions(interactions, file_desc, coord_format, use_symops, energy_cutoff, radius_cutoff, distance_tolerance);
  check_interaction_list_symmetry(interactions);


  // now the interaction data should be in the same format regardless of the input
  // calculate the neighbourlist from here

  auto nbrs = neighbour_list_from_interactions(interactions);

  neighbour_list_checks(nbrs, checks);

  return nbrs;
}

jams::InteractionList<Mat3, 2>
generate_neighbour_list(libconfig::Setting &setting,
                        CoordinateFormat coord_format,
                        bool use_symops,
                        double energy_cutoff,
                        double radius_cutoff,
                        double distance_tolerance,
                        std::vector<InteractionChecks> checks) {
  auto file_desc = discover_interaction_setting_format(setting);
  auto interactions = interactions_from_settings(setting, file_desc);

  post_process_interactions(interactions, file_desc, coord_format, use_symops, energy_cutoff, radius_cutoff, distance_tolerance);
  check_interaction_list_symmetry(interactions);

  // now the interaction data should be in the same format regardless of the input
  // calculate the neighbourlist from here

  auto nbrs = neighbour_list_from_interactions(interactions);

  neighbour_list_checks(nbrs, checks);

  return nbrs;
}

void neighbour_list_checks(const jams::InteractionList<Mat3, 2>& list, const std::vector<InteractionChecks>& checks) {
  for (const auto& check : checks) {
    switch (check) {
      case InteractionChecks::kNoZeroMotifNeighbourCount:
        for (auto i = 0; i < globals::num_spins; ++i) {
          if (list.num_interactions(i) == 0) {
            throw std::runtime_error(
                "inconsistent neighbour list: some sites have no neighbours");
          }
        }
        break;
      case InteractionChecks::kIdenticalMotifNeighbourCount:
        if (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2)) {
          std::vector<unsigned> motif_position_interactions(
              globals::lattice->num_basis_sites());
            for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {
              motif_position_interactions[i] = list.num_interactions(i);
            }

            for (auto i = 0; i < globals::num_spins; ++i) {
              auto pos = globals::lattice->lattice_site_basis_index(i);
              if (list.num_interactions(i) != motif_position_interactions[pos]) {
                throw std::runtime_error(
                    "inconsistent neighbour list: some sites have different numbers of neighbours for the same motif position");
            }
          }
        }
        break;
      case InteractionChecks::kIdenticalMotifTotalExchange:
      if (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2)) {

          // check diagonal part of J0 is the same for each motif position
          auto lambda = [](const Mat3 &prev,
                           const jams::InteractionList<Mat3, 2>::pair_type &next) {
              return prev + next.second;
          };

        std::vector<Mat3> motif_position_total_exchange(globals::lattice->num_basis_sites(),
                                                     kZeroMat3);
          for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {
            auto neighbour_list = list.interactions_of(i);
            motif_position_total_exchange[i] = std::accumulate(
                neighbour_list.begin(), neighbour_list.end(), kZeroMat3,
                lambda);
          }

          for (auto i = 0; i < globals::num_spins; ++i) {
            auto neighbour_list = list.interactions_of(i);

            auto pos = globals::lattice->lattice_site_basis_index(i);

            Mat3 J0 = std::accumulate(neighbour_list.begin(),
                                      neighbour_list.end(), kZeroMat3, lambda);

            if (!approximately_equal(diag(J0),
                                     diag(motif_position_total_exchange[pos]),
                                     1e-6)) {
              throw std::runtime_error("inconsistent neighbour list: J0");
            }
          }
        }
        break;
    }
  }
}

void
safety_check_distance_tolerance(const double &tolerance) {
  // check that no atoms in the unit cell are closer together than the tolerance

  for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {
    for (auto j = i + 1; j < globals::lattice->num_basis_sites(); ++j) {
      const auto distance = norm(globals::lattice->basis_site_atom(i).position_frac - globals::lattice->basis_site_atom(
          j).position_frac);
      if (distance < tolerance) {
        throw jams::SanityException("Atoms ", i, " and ", j, " in the unit cell are close together (", distance,
                                    ") than the distance_tolerance (", tolerance, ").\n Check the positions",
                                    "or relax distance_tolerance");
      }
    }
  }
}

void check_interaction_list_symmetry(const std::vector<InteractionData> &interactions) {
  for (const auto &J : interactions) {
    InteractionData sym_J;
    sym_J.basis_site_i = J.basis_site_j;
    sym_J.basis_site_j = J.basis_site_i;
    sym_J.interaction_vector_cart = -J.interaction_vector_cart;
    sym_J.interaction_value_tensor = transpose(J.interaction_value_tensor);
    sym_J.type_i = J.type_j;
    sym_J.type_j = J.type_i;


    auto it = std::find_if(interactions.begin(), interactions.end(), [&](const InteractionData& sym_J){
      return (sym_J.basis_site_i == J.basis_site_i
      && sym_J.basis_site_j == J.basis_site_j
      && sym_J.type_i == J.type_i
      && sym_J.type_j == J.type_j
      && approximately_equal(sym_J.interaction_vector_cart, J.interaction_vector_cart, jams::defaults::lattice_tolerance)
      && approximately_equal(sym_J.interaction_value_tensor, J.interaction_value_tensor, 1e-4));
    });

    if (it == interactions.end()) {
      std::string message = "Interaction template is not symmetric. " +
      std::to_string(sym_J.basis_site_i + 1) + " " + std::to_string(sym_J.basis_site_j + 1) + " " +
        std::to_string(sym_J.interaction_vector_cart[0]) + " " +
          std::to_string(sym_J.interaction_vector_cart[1]) + " " +
            std::to_string(sym_J.interaction_vector_cart[2]);

      throw jams::SanityException(message);
    }
  }
}

void
write_interaction_data(std::ostream &output, const std::vector<InteractionData> &data, CoordinateFormat coord_format) {
  for (auto const &interaction : data) {
    output << std::setw(12) << interaction.basis_site_i << "\t";
    output << std::setw(12) << interaction.basis_site_j << "\t";
    output << std::setw(12) << interaction.type_i << "\t";
    output << std::setw(12) << interaction.type_j << "\t";
    output << std::setw(12) << std::fixed << norm(interaction.interaction_vector_cart) << "\t";
    if (coord_format == CoordinateFormat::CARTESIAN) {
      output << std::setw(12) << std::fixed << interaction.interaction_vector_cart[0] << "\t";
      output << std::setw(12) << std::fixed << interaction.interaction_vector_cart[1] << "\t";
      output << std::setw(12) << std::fixed << interaction.interaction_vector_cart[2] << "\t";
    } else {
      auto r_ij_frac = globals::lattice->cartesian_to_fractional(interaction.interaction_vector_cart);
      output << std::setw(12) << std::fixed << r_ij_frac[0] << "\t";
      output << std::setw(12) << std::fixed << r_ij_frac[1] << "\t";
      output << std::setw(12) << std::fixed << r_ij_frac[2] << "\t";
    }
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[0][0] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[0][1] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[0][2] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[1][0] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[1][1] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[1][2] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[2][0] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[2][1] << "\t";
    output << std::setw(12) << std::scientific << interaction.interaction_value_tensor[2][2] << std::endl;
  }
}
void
write_neighbour_list(std::ostream &output, const jams::InteractionList<Mat3,2> &list) {
  output << "#";
  output << jams::fmt::integer << "i";
  output << jams::fmt::integer << "j";
  output << jams::fmt::integer << "type_i";
  output << jams::fmt::integer << "type_j";
  output << jams::fmt::decimal << "rx_i";
  output << jams::fmt::decimal << "ry_i";
  output << jams::fmt::decimal << "rz_i";
  output << jams::fmt::decimal << "rx_j";
  output << jams::fmt::decimal << "ry_j";
  output << jams::fmt::decimal << "rz_j";
  output << jams::fmt::decimal << "rx_ij";
  output << jams::fmt::decimal << "ry_ij";
  output << jams::fmt::decimal << "rz_ij";
  output << jams::fmt::decimal << "|interaction_vector_cart|";
  output << jams::fmt::sci << "Jij_xx";
  output << jams::fmt::sci << "Jij_xy";
  output << jams::fmt::sci << "Jij_xz";
  output << jams::fmt::sci << "Jij_yx";
  output << jams::fmt::sci << "Jij_yy";
  output << jams::fmt::sci << "Jij_yz";
  output << jams::fmt::sci << "Jij_zx";
  output << jams::fmt::sci << "Jij_zy";
  output << jams::fmt::sci << "Jij_zz" << "\n";

  for (int n = 0; n < list.size(); ++n) {
      auto i = list[n].first[0];
      auto j = list[n].first[1];
      auto rij = globals::lattice->displacement(i, j);
      auto Jij = list[n].second;
      output << jams::fmt::integer << i;
      output << jams::fmt::integer << j;
      output << jams::fmt::integer << globals::lattice->lattice_site_material_name(i);
      output << jams::fmt::integer << globals::lattice->lattice_site_material_name(j);
      output << jams::fmt::decimal << globals::lattice->lattice_site_position_cart(i)[0];
      output << jams::fmt::decimal << globals::lattice->lattice_site_position_cart(i)[1];
      output << jams::fmt::decimal << globals::lattice->lattice_site_position_cart(i)[2];
      output << jams::fmt::decimal << globals::lattice->lattice_site_position_cart(j)[0];
      output << jams::fmt::decimal << globals::lattice->lattice_site_position_cart(j)[1];
      output << jams::fmt::decimal << globals::lattice->lattice_site_position_cart(j)[2];
      output << jams::fmt::decimal << rij[0];
      output << jams::fmt::decimal << rij[1];
      output << jams::fmt::decimal << rij[2];
      output << jams::fmt::decimal << norm(rij);
      output << jams::fmt::sci << std::scientific << Jij[0][0];
      output << jams::fmt::sci << std::scientific << Jij[0][1];
      output << jams::fmt::sci << std::scientific << Jij[0][2];
      output << jams::fmt::sci << std::scientific << Jij[1][0];
      output << jams::fmt::sci << std::scientific << Jij[1][1];
      output << jams::fmt::sci << std::scientific << Jij[1][2];
      output << jams::fmt::sci << std::scientific << Jij[2][0];
      output << jams::fmt::sci << std::scientific << Jij[2][1];
      output << jams::fmt::sci << std::scientific << Jij[2][2] << "\n";
  }
}

