// neighbour_lists.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_INTERACTION_LIST_HELPERS
#define INCLUDED_JAMS_INTERACTION_LIST_HELPERS

#include <jams/containers/interaction_list.h>
#include <jams/interface/config.h>
#include <jams/helpers/defaults.h>
#include <jams/core/interactions.h>
#include <jams/containers/sparse_matrix_builder.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <fstream>

namespace jams {

/// @brief Generates an interaction list from libconfig settings. The
/// interactions themselves can be either specified in the settings or in an
/// external tsv file.
///
/// @details Reads the following optional settings (with default values in
/// brackets) and generates an InteractionList for the whole supercell.
///
/// interactions
/// ------------
/// Interaction vectors specified in the settings block.
///
/// exc_file
/// --------
/// Path to a tsv exchange interaction file containing interaction data. The
/// path is relative to the location JAMS is executed in.
///
/// coordinate_format (cartesian)
/// -----------------------------
/// Format the interaction vectors are specified in. This can be 'cartesian' or
/// 'fractional'. Cartesian vectors are in units of lattice constants and
/// fractional are fractions of unit cell vectors.
///
/// symops (true)
/// -------------
/// Generate new interactions by applying the symmetry operations of the lattice
/// as detected in the lattice class.
///
/// interaction_prefactor (1.0)
/// ---------------------------
/// Apply a constant prefactor to all interaction energies. This is used for
/// converting between different Hamiltonian conventions (factors of 1/2, 2, S etc.)
/// NOTE: This prefactor is applied AFTER any energy cutoff operations.
///
/// energy_cutoff (0.0)
/// -------------------
/// Remove interactions where the max abs value of the energy tensor is smaller
/// than this value. Used for cutting off the long tail of tiny noisy energies
/// from electronic structure input.
///
/// radius_cutoff (0.0)
/// -------------------
/// Remove interactions where the interaction vector is larger than this value.
/// A value of 0.0 means no cutoff.
///
/// distance_tolerance (jams::defaults::lattice_tolerance)
/// ------------------------------------------------------
/// Tolerance used to compare distances.
///
/// energy_units (joules)
/// ---------------------
/// Energy unit name for converting to internal JAMS units.
/// See jams/core/units.h for allowed values.
/// NOTE: The energy_cutoff and the interaction energies must be in the same
/// units.
///
/// check_no_zero_motif_neighbour_count (true)
/// ------------------------------------------
/// If true, an exception will be raised if any motif position has zero
/// neighbours (i.e. it is not included in the interaction list). It may be
/// desirable to zero neighbours, for example if another interaction
/// Hamiltonian is coupling these sites.
///
/// check_identical_motif_neighbour_count (true)
/// --------------------------------------------
/// If true, an exception will be raised if any sites in the lattice which
/// have the same motif position in the unit cell, have different numbers
/// of neighbours.
/// NOTE: This check will only run if periodic boundaries are disabled.
///
/// check_identical_motif_total_exchange (true)
/// -------------------------------------------
/// If true, an exception will be raised in any sites in the lattice which
/// have the same motif position in the unit cell, have different total
/// exchange energy. The total exchange energy is calculated from the absolute
/// sum of the diagonal components of the exchange tensor.
/// NOTE: This check will only run if periodic boundaries are disabled.
InteractionList<Mat3, 2> InteractionListFromSettings(const libconfig::Setting& settings);

void PrintInteractionList(std::ostream& os, const InteractionList<Mat3, 2>& neighbour_list);

template<class T, int N>
void PrintInteractionListProperties(std::ostream& os, const InteractionList<T, N>& neighbour_list);
}

// -------------------------- Implementation -------------------------------

template<class T, int N>
void jams::PrintInteractionListProperties(std::ostream& os, const InteractionList<T, N>& neighbour_list) {
  os << "number of interactions: " << neighbour_list.size() << "\n";
  os << "interaction list memory: " << neighbour_list.memory() / kBytesToMegaBytes << " MB" << std::endl;
  std::cout << "interactions per motif position: \n";
  if (globals::lattice->is_periodic(0) && globals::lattice->is_periodic(1) && globals::lattice->is_periodic(2) && !globals::lattice->has_impurities()) {
    for (auto i = 0; i < globals::lattice->num_motif_atoms(); ++i) {
      std::cout << "  " << i << ": " << neighbour_list.num_interactions(i) <<"\n";
    }
  }
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------