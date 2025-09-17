// cvar_topological_charge_finite_diff.cc                              -*-C++-*-
#include <jams/metadynamics/cvars/cvar_topological_charge_finite_diff.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>

jams::CVarTopologicalChargeFiniteDiff::CVarTopologicalChargeFiniteDiff(
    const libconfig::Setting &settings) {

  // true if a and b are equal to the lattice a and b vectors.
  auto lattice_equal = [&](Vec3 a, Vec3 b) {
    return approximately_equal(globals::lattice->a1(), a, jams::defaults::lattice_tolerance)
    && approximately_equal(globals::lattice->a2(), b, jams::defaults::lattice_tolerance);
  };

  enum class LatticeType {Unsupported, Square, Hexagonal};
  LatticeType lattice_type = LatticeType::Unsupported;
  if (lattice_equal({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0})) {
    lattice_type = LatticeType::Square;
  } else if (lattice_equal({1.0, 0.0, 0.0}, {0.5, sqrt(3) / 2, 0.0})) {
    lattice_type = LatticeType::Hexagonal;
  }

  if (lattice_type == LatticeType::Unsupported) {
    throw std::runtime_error("Metadynamics CV 'topological_charge_finite_diff' "
                             "requires the lattice to be either square or hexagonal.");
  }

  enum class FiniteDifferenceStencil {Unsupported, Square8, Hexagonal4, Hexagonal6, Hexagonal12};
  FiniteDifferenceStencil stencil = FiniteDifferenceStencil::Unsupported;

  // Set defaults for backwards compatibility
  if (lattice_type == LatticeType::Square) {
    stencil = FiniteDifferenceStencil::Square8;
  } else if (lattice_type == LatticeType::Hexagonal) {
    stencil = FiniteDifferenceStencil::Hexagonal6;
  }

  if(settings.exists("stencil")) {
    std::string stencil_name = settings["stencil"].c_str();

    // if there is a stencil setting then reset the stencil here so that we can
    // check later if an unsupported stencil/lattice combination was chosen

    stencil = FiniteDifferenceStencil::Unsupported;
    if (stencil_name == "Square8" &&
        lattice_type == LatticeType::Square) {
      stencil = FiniteDifferenceStencil::Square8;
    } else if (stencil_name == "Hexagonal6" &&
               lattice_type == LatticeType::Hexagonal) {
      stencil = FiniteDifferenceStencil::Hexagonal6;
    } else if (stencil_name == "Hexagonal4" &&
               lattice_type == LatticeType::Hexagonal) {
      stencil = FiniteDifferenceStencil::Hexagonal4;
    } else if (stencil_name ==
               "Hexagonal12" && lattice_type == LatticeType::Hexagonal) {
      stencil = FiniteDifferenceStencil::Hexagonal12;
    }
  }

  if (stencil == FiniteDifferenceStencil::Unsupported) {
    throw std::runtime_error("Metadynamics CV 'topological_charge_finite_diff' "
                             "an unsupported stencil or stencil/lattice combination was selected");
  }

  // Below we create the finite difference stencils. These are based on stencils
  // in Xu, G.,Liu, G.R. (2006-10) WSEAS Transactions on Mathematics 5 (10)
  // : 1117-1122. The figures below are from Fig. 2 in the paper and show the
  // weights for (∂u/∂x, ∂u/∂y) in the brackets.
  //
  // Square lattice stencil
  // ----------------------
  //
  //   (-1/16, 1/16)   (0, 3/8)  (1/16, 1/16)
  //               +-----+-----+
  //               |     |     |
  //               |     |     |
  //     (-3/8, 0) +-----+-----+ ( 3/8, 0)
  //               |     |     |
  //               |     |     |
  //               +-----+-----+
  //  (-1/16,-1/16)   (0,-3/8)  ( 1/16,-1/16)
  //
  //
  // Hexagonal lattice stencil
  // -------------------------
  //
  //     (-1/6, √3/6) _______ ( 1/6, √3/6)
  //                 /\     /\
  //                /  \   /  \
  //     (-1/3, 0) /____\ /____\ ( 1/3, 0)
  //               \    / \    /
  //                \  /   \  /
  //                 \/_____\/
  //     (-1/6,-√3/6)         ( 1/6,-√3/6)
  //
  // We'll generate neighbour lists from these stencils the same we do with
  // exchange interactions using the functions and classes in
  // jams/core/interactions.h. This is quite messy and overkill, but will avoid
  // bugs while we do a very quick job here.
  //
  // Usually the interaction template is read from a plain text file or the
  // config. We'll skip that step and build the template manually here.
  //

  stencil_neighbour_indices_.resize(globals::num_spins);

  // ------------------------------- ∂S/∂x -------------------------------------
  {
    // first index is interaction vector, second is the +/- sign of the
    // contribution

    std::vector<std::pair<Vec3, double>> dx_interaction_data;

    if (stencil == FiniteDifferenceStencil::Square8) {
      dx_interaction_data = {
          {Vec3{ 1, 1, 0}, 1.0/16.0},
          {Vec3{ 1,-1, 0}, 1.0/16.0},
          {Vec3{ 1, 0, 0}, 3.0/8.0},
          {Vec3{-1, 0, 0},-3.0/8.0},
          {Vec3{-1, 1, 0},-1.0/16.0},
          {Vec3{-1,-1, 0},-1.0/16.0},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal4) {
      dx_interaction_data = {
          {Vec3{ 1,-1, 0}, 1.0/2.0},
          {Vec3{ 0, 1, 0}, 1.0/2.0},
          {Vec3{-1, 1, 0},-1.0/2.0},
          {Vec3{ 0,-1, 0},-1.0/2.0},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal6) {
      dx_interaction_data = {
          {Vec3{ 1, 0, 0}, 1.0/3.0},
          {Vec3{-1, 0, 0},-1.0/3.0},
          {Vec3{ 1,-1, 0}, 1.0/6.0},
          {Vec3{-1, 1, 0},-1.0/6.0},
          {Vec3{ 0, 1, 0}, 1.0/6.0},
          {Vec3{ 0,-1, 0},-1.0/6.0}
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal12) {
      dx_interaction_data = {
          {Vec3{ 1, 0, 0}, 1.0/2.0}, // F
          {Vec3{-1, 0, 0},-1.0/2.0}, // G
          {Vec3{ 1,-1, 0}, 1.0/4.0}, // C
          {Vec3{-1, 1, 0},-1.0/4.0}, // D
          {Vec3{ 0, 1, 0}, 1.0/4.0}, // B
          {Vec3{ 0,-1, 0},-1.0/4.0}, // E
          {Vec3{ 1, 1, 0},-1.0/12.0},
          {Vec3{ 2,-1, 0},-1.0/12.0},
          {Vec3{-2, 1, 0}, 1.0/12.0},
          {Vec3{-1,-1, 0}, 1.0/12.0}
      };
    }

    std::vector<InteractionData> interaction_template;
    for (auto &data: dx_interaction_data) {
      InteractionData J;
      J.basis_site_i = 0;
      J.basis_site_j = 0;
      J.type_i = globals::lattice->material_name(
          globals::lattice->basis_site_atom(J.basis_site_i).material_index);
      J.type_j = globals::lattice->material_name(
          globals::lattice->basis_site_atom(J.basis_site_j).material_index);
      J.interaction_vector_cart = globals::lattice->fractional_to_cartesian(data.first);
      J.interaction_value_tensor[0][0] = data.second;
      interaction_template.push_back(J);
    }

    post_process_interactions(interaction_template, {InteractionFileFormat::KKR, InteractionType::SCALAR}, CoordinateFormat::CARTESIAN, false, 0.0, 0.0, jams::defaults::lattice_tolerance);

    jams::InteractionList<Mat3,2> nbrs = neighbour_list_from_interactions(interaction_template);
    dx_indices_.resize(globals::num_spins);
    dx_values_.resize(globals::num_spins);

    for (auto n = 0; n < nbrs.size(); ++n) {
      auto i = nbrs[n].first[0];
      auto j = nbrs[n].first[1];
      auto weight = nbrs[n].second[0][0];
      stencil_neighbour_indices_[i].insert(j);
      dx_indices_[i].push_back(j);
      dx_values_[i].push_back(weight);
    }
  }

  // ------------------------------- ∂S/∂y -------------------------------------
  {
    // first index is interaction vector, second is the +/- sign of the
    // contribution
    std::vector<std::pair<Vec3, double>> dy_interaction_data;

    if (stencil == FiniteDifferenceStencil::Square8) {
      dy_interaction_data = {
          {Vec3{ 1, 1, 0}, 1.0/16.0},
          {Vec3{-1, 1, 0}, 1.0/16.0},
          {Vec3{ 0, 1, 0}, 3.0/8.0},
          {Vec3{ 0,-1, 0},-3.0/8.0},
          {Vec3{ 1,-1, 0},-1.0/16.0},
          {Vec3{-1,-1, 0},-1.0/16.0},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal4) {
      dy_interaction_data = {
          {Vec3{-1, 1, 0}, 1.0/(2*sqrt(3.0))},
          {Vec3{ 0, 1, 0}, 1.0/(2*sqrt(3.0))},
          {Vec3{ 1,-1, 0},-1.0/(2*sqrt(3.0))},
          {Vec3{ 0,-1, 0},-1.0/(2*sqrt(3.0))},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal6) {
      dy_interaction_data = {
          {Vec3{ 1,-1, 0},-sqrt(3.0)/6.0},
          {Vec3{-1, 1, 0}, sqrt(3.0)/6.0},
          {Vec3{ 0, 1, 0}, sqrt(3.0)/6.0},
          {Vec3{ 0,-1, 0},-sqrt(3.0)/6.0},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal12) {
      dy_interaction_data = {
          {Vec3{ 1,-1, 0},-sqrt(3.0)/4.0},
          {Vec3{-1, 1, 0}, sqrt(3.0)/4.0},
          {Vec3{ 0, 1, 0}, sqrt(3.0)/4.0},
          {Vec3{ 0,-1, 0},-sqrt(3.0)/4.0},
          {Vec3{-1, 2, 0}, -1/(6*sqrt(3))},
          {Vec3{ 1,-2, 0}, 1/(6*sqrt(3))},
          {Vec3{ 1, 1, 0},-1.0/(12*sqrt(3))},
          {Vec3{ 2,-1, 0}, 1.0/(12*sqrt(3))},
          {Vec3{-2, 1, 0}, -1.0/(12*sqrt(3))},
          {Vec3{-1,-1, 0}, 1.0/(12*sqrt(3))}
      };
    }

      std::vector<InteractionData> interaction_template;
    for (auto &data: dy_interaction_data) {
      InteractionData J;
      J.basis_site_i = 0;
      J.basis_site_j = 0;
      J.type_i = globals::lattice->material_name(
          globals::lattice->basis_site_atom(J.basis_site_i).material_index);
      J.type_j = globals::lattice->material_name(
          globals::lattice->basis_site_atom(J.basis_site_j).material_index);
      J.interaction_vector_cart = ::globals::lattice->fractional_to_cartesian(data.first);
      J.interaction_value_tensor[0][0] = data.second;
      interaction_template.push_back(J);
    }

    post_process_interactions(interaction_template, {InteractionFileFormat::KKR, InteractionType::SCALAR}, CoordinateFormat::CARTESIAN, false, 0.0, 0.0, jams::defaults::lattice_tolerance);


    jams::InteractionList<Mat3, 2> nbrs = neighbour_list_from_interactions(
        interaction_template);
    dy_indices_.resize(globals::num_spins);
    dy_values_.resize(globals::num_spins);

    for (auto n = 0; n < nbrs.size(); ++n) {
      auto i = nbrs[n].first[0];
      auto j = nbrs[n].first[1];
      auto weight = nbrs[n].second[0][0];
      stencil_neighbour_indices_[i].insert(j);
      dy_indices_[i].push_back(j);
      dy_values_[i].push_back(weight);
    }
  }


}

std::string jams::CVarTopologicalChargeFiniteDiff::name() {
  return name_;
}

double jams::CVarTopologicalChargeFiniteDiff::value() {
  return cached_value();
}

double jams::CVarTopologicalChargeFiniteDiff::spin_move_trial_value(int i,
                                                                    const Vec3 &spin_initial,
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