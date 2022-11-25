// cvar_topological_charge_finite_diff.cc                              -*-C++-*-
#include <jams/metadynamics/cvars/cvar_topological_charge_finite_diff.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>

jams::CVarTopologicalChargeFiniteDiff::CVarTopologicalChargeFiniteDiff(
    const libconfig::Setting &settings) {

  if (!approximately_equal(lattice->a(), {1.0, 0.0, 0.0}, jams::defaults::lattice_tolerance)) {
    throw std::runtime_error("Metadynamics CV 'topological_charge_finite_diff' "
                             "requires the 'a' lattice parameter to be (1.0, 0.0, 0.0)");
  }

  if (!approximately_equal(lattice->b(), {0.5, sqrt(3)/2, 0.0}, jams::defaults::lattice_tolerance)) {
    throw std::runtime_error("Metadynamics CV 'topological_charge_finite_diff' "
                             "requires the 'b' lattice parameter to be (0.5, 0.8660254, 0.0)");
  }

  // Create the finite difference template. For now we assume a hexagonal lattice
  // with vectors u₁ = x, u2 = x/2 + √3/2 the finite difference scheme
  // will be (see Rohart, Phys. Rev. B 93, 214412 (2016) Supplementary Info)
  //
  // ∂ₓS ≈ [S(rᵢ + u₁ - u₂) + S(rᵢ + u₂) - S(rᵢ - u₁ + u₂) - S(rᵢ - u₂)] / 2
  // ∂ᵧS ≈ [S(rᵢ - u₁ + u₂) + S(rᵢ + u₂) - S(rᵢ + u₁ - u₂) - S(rᵢ - u₂)] / 2√3
  //
  // The same 4 neighbour spins are used in both ∂ₓS and ∂ᵧS, but things might
  // differ at the edge of the system, so we'll construct two neighbour lists:
  // one for ∂ₓS and one for ∂ᵧS.
  //
  // We'll generate these neighbour lists the same we do with exchange
  // interactions using the functions and classes in jams/core/interactions.h.
  // This is quite messy and massively overkill, but will avoid bugs while we
  // do a very quick job here.
  //
  // Usually the interaction template is read from a plain text file or the
  // config. We'll skip that step and build the template manually here.
  //


  // ------------------------------- ∂ₓS ---------------------------------------
  {
    // first index is interaction vector, second is the +/- sign of the
    // contribution
    std::vector<std::pair<Vec3, int>> dx_interaction_data = {
        {Vec3{1, -1, 0}, +1}, // +S(rᵢ + u₁ - u₂)
        {Vec3{-1, 1, 0}, -1}, // -S(rᵢ - u₁ + u₂)
        {Vec3{0, 1, 0},  +1}, // +S(rᵢ + u₂)
        {Vec3{0, -1, 0}, -1}  // -S(rᵢ - u₂)
    };

    std::vector<InteractionData> interaction_template;
    for (auto &data: dx_interaction_data) {
      InteractionData J;
      J.unit_cell_pos_i = 0;
      J.unit_cell_pos_j = 0;
      J.type_i = lattice->material_name(
          lattice->motif_atom(J.unit_cell_pos_i).material_index);
      J.type_j = lattice->material_name(
          lattice->motif_atom(J.unit_cell_pos_j).material_index);
      J.r_ij = ::lattice->fractional_to_cartesian(data.first);
      J.J_ij[0][0] = data.second;
      interaction_template.push_back(J);
    }

    InteractionList<Mat3, 2> nbrs = neighbour_list_from_interactions(
        interaction_template);

    dx_indices_.resize(globals::num_spins);
    dx_values_.resize(globals::num_spins);

    for (auto n = 0; n < nbrs.size(); ++n) {
      auto i = nbrs[n].first[0];
      auto j = nbrs[n].first[1];
      auto sign = nbrs[n].second[0][0];

      // Divide by 2 for the dx direction
      stencil_neighbour_indices_[i].insert(j);
      dx_indices_[i].push_back(j);
      dx_values_[i].push_back(sign / 2.0);
    }
  }

  // ------------------------------- ∂ᵧS ---------------------------------------
  {
    // first index is interaction vector, second is the +/- sign of the
    // contribution
    std::vector<std::pair<Vec3, int>> dx_interaction_data = {
        {Vec3{1, -1, 0}, -1}, // -S(rᵢ + u₁ - u₂)
        {Vec3{-1, 1, 0}, +1}, // +S(rᵢ - u₁ + u₂)
        {Vec3{0, 1, 0},  +1}, // +S(rᵢ + u₂)
        {Vec3{0, -1, 0}, -1}  // -S(rᵢ - u₂)
    };

    std::vector<InteractionData> interaction_template;
    for (auto &data: dx_interaction_data) {
      InteractionData J;
      J.unit_cell_pos_i = 0;
      J.unit_cell_pos_j = 0;
      J.type_i = lattice->material_name(
          lattice->motif_atom(J.unit_cell_pos_i).material_index);
      J.type_j = lattice->material_name(
          lattice->motif_atom(J.unit_cell_pos_j).material_index);
      J.r_ij = ::lattice->fractional_to_cartesian(data.first);
      J.J_ij[0][0] = data.second;
      interaction_template.push_back(J);
    }

    InteractionList<Mat3, 2> nbrs = neighbour_list_from_interactions(
        interaction_template);

    dy_indices_.resize(globals::num_spins);
    dy_values_.resize(globals::num_spins);

    for (auto n = 0; n < nbrs.size(); ++n) {
      auto i = nbrs[n].first[0];
      auto j = nbrs[n].first[1];
      auto sign = nbrs[n].second[0][0];

      // Divide by 2\sqrt{3} for the dy direction
      stencil_neighbour_indices_[i].insert(j);
      dy_indices_[i].push_back(j);
      dy_values_[i].push_back(sign / (2.0*sqrt(3)));
    }
  }


}

std::string jams::CVarTopologicalChargeFiniteDiff::name() {  return name_;
}

double jams::CVarTopologicalChargeFiniteDiff::value() {
  return cached_value();
}

double jams::CVarTopologicalChargeFiniteDiff::spin_move_trial_value(int i,                                                                    const Vec3 &spin_initial,
                                                                    const Vec3 &spin_trial) {
  const double trial_value = cached_value() + topological_charge_difference(i, spin_initial, spin_trial);

  set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);

  return trial_value;  //Used in CollectiveVariable
}



double jams::CVarTopologicalChargeFiniteDiff::calculate_expensive_value() {

  double topological_charge = 0.0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    topological_charge += local_topological_charge(i);
  }

  return topological_charge / (4.0 * kPi);
}

double jams::CVarTopologicalChargeFiniteDiff::local_topological_charge(const int i) const {
  Vec3 ds_x = {0.0, 0.0, 0.0};
  for (auto n = 0; n < dx_indices_[i].size(); ++n) {
    ds_x += dx_values_[i][n] * montecarlo::get_spin(dx_indices_[i][n]);
  }

  Vec3 ds_y = {0.0, 0.0, 0.0};
  for (auto n = 0; n < dy_indices_[i].size(); ++n) {
    ds_y += dy_values_[i][n] * montecarlo::get_spin(dy_indices_[i][n]);
  }

  Vec3 s_i = montecarlo::get_spin(i);

  return dot(s_i, cross(ds_x, ds_y));
}

double jams::CVarTopologicalChargeFiniteDiff::topological_charge_difference(int index,
                                                                  const Vec3 &spin_initial,
                                                                  const Vec3 &spin_final) const {
  // We calculate the difference in the topological charge between the initial
  // and final spin states. When one spin is changed, the topological charge on
  // all sites connected to it through the finite difference stencil also changes.
  // We therefore calculate the difference of the topological charge of the whole
  // stencil.

  montecarlo::set_spin(index, spin_initial);

  double initial_charge = local_topological_charge(index);
  // Loop over neighbouring sites in the stencil
  for (int n : stencil_neighbour_indices_[index]) {
    initial_charge += local_topological_charge(n);
  }

  montecarlo::set_spin(index, spin_final);

  double final_charge = local_topological_charge(index);
  // Loop over neighbouring sites in the stencil
  for (int n : stencil_neighbour_indices_[index]) {
    final_charge += local_topological_charge(n);
  }


  montecarlo::set_spin(index, spin_initial);

  return (final_charge - initial_charge) / (4.0 * kPi);
}