// cvar_local_spin_order.cc                                            -*-C++-*-
#include <jams/metadynamics/cvars/cvar_local_spin_order.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>

jams::CVarLocalSpinOrder::CVarLocalSpinOrder(
    const libconfig::Setting &settings) {

  libconfig::Setting& neighbours  = settings["neighbours"];

  std::vector<InteractionData> interaction_template;
  for (auto i = 0; i < neighbours.getLength(); ++i) {
    InteractionData J;

    J.unit_cell_pos_i = int(neighbours[i][0])-1;
    J.unit_cell_pos_j = int(neighbours[i][1])-1;
    J.type_i = globals::lattice->material_name(
        globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
    J.type_j = globals::lattice->material_name(
        globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);

    J.r_ij = {neighbours[i][2][0], neighbours[i][2][1], neighbours[i][2][2]};
    J.J_ij = kIdentityMat3;

    interaction_template.push_back(J);
  }
  jams::InteractionList<Mat3,2> nbrs = neighbour_list_from_interactions(interaction_template);


  neighbour_indices_.resize(globals::num_spins);
  num_neighbour_.resize(globals::num_spins, 0);
  num_spins_selected_ = 0;

  for (auto n = 0; n < nbrs.size(); ++n) {
    auto i = nbrs[n].first[0];
    auto j = nbrs[n].first[1];
    neighbour_indices_[i].push_back(j);
    num_neighbour_[i]++;
  }

  for (auto i = 0; i < globals::num_spins; ++i) {
    if (num_neighbour_[i] != 0.0) {
      num_spins_selected_++;
    }
  }
}

std::string jams::CVarLocalSpinOrder::name() {
  return name_;
}

double jams::CVarLocalSpinOrder::value() {
  return cached_value();
}

double jams::CVarLocalSpinOrder::spin_move_trial_value(int i,
                                                                    const Vec3 &spin_initial,
                                                                    const Vec3 &spin_trial) {
  // check if the spin is of relevant material
  if (num_neighbour_[i] != 0) {
    const double trial_value = cached_value() +
        spin_order_difference(i, spin_initial, spin_trial);
    set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);
    return trial_value;
  }

  set_cache_values(i, spin_initial, spin_trial, cached_value(), cached_value());

  return cached_value();  //Used in CollectiveVariable
}



double jams::CVarLocalSpinOrder::calculate_expensive_cache_value() {
  double spin_order = 0.0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    spin_order += local_spin_order(i);
  }

  return spin_order / num_spins_selected_;
}

double jams::CVarLocalSpinOrder::local_spin_order(const int i) const {
  if (num_neighbour_[i] == 0) {
    return 0.0;
  }

  double order = 0.0;
  for (auto n = 0; n < neighbour_indices_[i].size(); ++n) {
    const auto j = neighbour_indices_[i][n];
    order += globals::s(i,0)*globals::s(j,0) + globals::s(i,1)*globals::s(j,1) + globals::s(i,2)*globals::s(j,2);
  }

  return order / double(num_neighbour_[i]);
}

double jams::CVarLocalSpinOrder::spin_order_difference(int index,
                                                       const Vec3 &spin_initial,
                                                       const Vec3 &spin_final) const {
  // We calculate the difference in the topological charge between the initial
  // and final spin states. When one spin is changed, the topological charge on
  // all sites connected to it through the finite difference stencil also changes.
  // We therefore calculate the difference of the topological charge of the whole
  // stencil.

  if (num_neighbour_[index] == 0) {
    return 0.0;
  }

  montecarlo::set_spin(index, spin_initial);

  double initial_order = local_spin_order(index);
  // Loop over neighbouring sites in the stencil
  for (int n : neighbour_indices_[index]) {
    initial_order += local_spin_order(n);
  }

  montecarlo::set_spin(index, spin_final);

  double final_order = local_spin_order(index);
  // Loop over neighbouring sites in the stencil
  for (int n : neighbour_indices_[index]) {
    final_order += local_spin_order(n);
  }


  montecarlo::set_spin(index, spin_initial);

  return (final_order - initial_order) / num_spins_selected_;
}