#include <jams/helpers/interaction_list_helpers.h>
#include <jams/core/units.h>
#include <jams/helpers/output.h>


jams::InteractionList<Mat3, 2> jams::InteractionListFromSettings(const libconfig::Setting& settings) {
  bool use_symops = true;
  settings.lookupValue("symops", use_symops);

  double interaction_prefactor = 1.0;
  settings.lookupValue("interaction_prefactor", interaction_prefactor);

  double energy_cutoff = 0.0;
  settings.lookupValue("energy_cutoff", energy_cutoff);

  double radius_cutoff = 0.0;
  settings.lookupValue("radius_cutoff", radius_cutoff);

  double distance_tolerance = jams::defaults::lattice_tolerance; // fractional coordinate units
  settings.lookupValue("distance_tolerance", distance_tolerance);

  safety_check_distance_tolerance(distance_tolerance);

  std::string energy_unit_name = jams::defaults::energy_unit_name;
  settings.lookupValue("energy_units", energy_unit_name);

  // old setting name for backwards compatibility
  if (settings.exists("unit_name")) {
    settings.lookupValue("unit_name", energy_unit_name);
  }

  if (!jams::internal_energy_unit_conversion.count(energy_unit_name)) {
    throw std::runtime_error("energy units: " + energy_unit_name + " is not known");
  }

  double energy_unit_conversion = jams::internal_energy_unit_conversion.at(energy_unit_name);

  std::vector<InteractionChecks> interaction_checks;

  if (!settings.exists("check_no_zero_motif_neighbour_count")) {
    interaction_checks.push_back(InteractionChecks::kNoZeroMotifNeighbourCount);
  } else {
    if (bool(settings["check_no_zero_motif_neighbour_count"]) == true) {
      interaction_checks.push_back(InteractionChecks::kNoZeroMotifNeighbourCount);
    }
  }

  if (!settings.exists("check_identical_motif_neighbour_count")) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifNeighbourCount);
  } else {
    if (bool(settings["check_identical_motif_neighbour_count"]) == true) {
      interaction_checks.push_back(InteractionChecks::kIdenticalMotifNeighbourCount);
    }
  }

  if (!settings.exists("check_identical_motif_total_exchange")) {
    interaction_checks.push_back(InteractionChecks::kIdenticalMotifTotalExchange);
  } else {
    if (bool(settings["check_identical_motif_total_exchange"]) == true) {
      interaction_checks.push_back(InteractionChecks::kIdenticalMotifTotalExchange);
    }
  }

  std::string coordinate_format_name = "CARTESIAN";
  settings.lookupValue("coordinate_format", coordinate_format_name);
  CoordinateFormat coord_format = coordinate_format_from_string(coordinate_format_name);

  // exc_file is to maintain backwards compatibility
  if (settings.exists("exc_file")) {
    std::ifstream interaction_file(settings["exc_file"].c_str());
    if (interaction_file.fail()) {
      jams_die("failed to open interaction file");
    }

    return neighbour_list_from_tsv(
        interaction_file, coord_format, use_symops, interaction_prefactor, energy_unit_conversion,
        energy_cutoff, radius_cutoff, interaction_checks);
  } else if (settings.exists("interactions")) {
    return neighbour_list_from_settings(
        settings["interactions"], coord_format, use_symops, interaction_prefactor, energy_unit_conversion,
        energy_cutoff, radius_cutoff, interaction_checks);
  } else {
    throw std::runtime_error("'exc_file' or 'interactions' settings are required");
  }

}


void jams::PrintInteractionList(std::ostream &os,
                                const jams::InteractionList<Mat3, 2> &neighbour_list) {
  os << "#";
  os << jams::fmt::integer << "i";
  os << jams::fmt::integer << "j";
  os << jams::fmt::integer << "type_i";
  os << jams::fmt::integer << "type_j";
  os << jams::fmt::decimal << "rx_i";
  os << jams::fmt::decimal << "ry_i";
  os << jams::fmt::decimal << "rz_i";
  os << jams::fmt::decimal << "rx_j";
  os << jams::fmt::decimal << "ry_j";
  os << jams::fmt::decimal << "rz_j";
  os << jams::fmt::decimal << "rx_ij";
  os << jams::fmt::decimal << "ry_ij";
  os << jams::fmt::decimal << "rz_ij";
  os << jams::fmt::decimal << "|r_ij|";
  os << jams::fmt::sci << "Jij_xx";
  os << jams::fmt::sci << "Jij_xy";
  os << jams::fmt::sci << "Jij_xz";
  os << jams::fmt::sci << "Jij_yx";
  os << jams::fmt::sci << "Jij_yy";
  os << jams::fmt::sci << "Jij_yz";
  os << jams::fmt::sci << "Jij_zx";
  os << jams::fmt::sci << "Jij_zy";
  os << jams::fmt::sci << "Jij_zz" << "\n";

  for (int n = 0; n < neighbour_list.size(); ++n) {
    auto i = neighbour_list[n].first[0];
    auto j = neighbour_list[n].first[1];
    auto rij = globals::lattice->displacement(i, j);
    auto Jij = neighbour_list[n].second;
    os << jams::fmt::integer << i;
    os << jams::fmt::integer << j;
    os << jams::fmt::integer << globals::lattice->atom_material_name(i);
    os << jams::fmt::integer << globals::lattice->atom_material_name(j);
    os << jams::fmt::decimal << globals::lattice->atom_position(i)[0];
    os << jams::fmt::decimal << globals::lattice->atom_position(i)[1];
    os << jams::fmt::decimal << globals::lattice->atom_position(i)[2];
    os << jams::fmt::decimal << globals::lattice->atom_position(j)[0];
    os << jams::fmt::decimal << globals::lattice->atom_position(j)[1];
    os << jams::fmt::decimal << globals::lattice->atom_position(j)[2];
    os << jams::fmt::decimal << rij[0];
    os << jams::fmt::decimal << rij[1];
    os << jams::fmt::decimal << rij[2];
    os << jams::fmt::decimal << norm(rij);
    os << jams::fmt::sci << std::scientific << Jij[0][0];
    os << jams::fmt::sci << std::scientific << Jij[0][1];
    os << jams::fmt::sci << std::scientific << Jij[0][2];
    os << jams::fmt::sci << std::scientific << Jij[1][0];
    os << jams::fmt::sci << std::scientific << Jij[1][1];
    os << jams::fmt::sci << std::scientific << Jij[1][2];
    os << jams::fmt::sci << std::scientific << Jij[2][0];
    os << jams::fmt::sci << std::scientific << Jij[2][1];
    os << jams::fmt::sci << std::scientific << Jij[2][2] << "\n";
  }

}
