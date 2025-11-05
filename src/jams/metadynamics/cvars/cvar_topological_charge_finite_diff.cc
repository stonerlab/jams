// cvar_topological_charge_finite_diff.cc                              -*-C++-*-
#include <jams/metadynamics/cvars/cvar_topological_charge_finite_diff.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>
#include <cmath>

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
  } else if (lattice_equal({1.0, 0.0, 0.0}, {0.5, std::sqrt(3.0) / 2.0, 0.0})) {
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

  const std::size_t expected_nbrs =
    (stencil == FiniteDifferenceStencil::Square8)   ? 8 :
    (stencil == FiniteDifferenceStencil::Hexagonal4)? 4 :
    (stencil == FiniteDifferenceStencil::Hexagonal6)? 6 :
    (stencil == FiniteDifferenceStencil::Hexagonal12)?12 : 0;

  stencil_neighbour_indices_.resize(globals::num_spins);
  if (expected_nbrs > 0) {
    for (auto &nbrs : stencil_neighbour_indices_) {
      nbrs.reserve(expected_nbrs);
    }
  }

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
          {Vec3{-1, 1, 0}, 1.0/(2.0*std::sqrt(3.0))},
          {Vec3{ 0, 1, 0}, 1.0/(2.0*std::sqrt(3.0))},
          {Vec3{ 1,-1, 0},-1.0/(2.0*std::sqrt(3.0))},
          {Vec3{ 0,-1, 0},-1.0/(2.0*std::sqrt(3.0))},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal6) {
      dy_interaction_data = {
          {Vec3{ 1,-1, 0},-std::sqrt(3.0)/6.0},
          {Vec3{-1, 1, 0}, std::sqrt(3.0)/6.0},
          {Vec3{ 0, 1, 0}, std::sqrt(3.0)/6.0},
          {Vec3{ 0,-1, 0},-std::sqrt(3.0)/6.0},
      };
    }
    if (stencil == FiniteDifferenceStencil::Hexagonal12) {
      dy_interaction_data = {
          {Vec3{ 1,-1, 0},-std::sqrt(3.0)/4.0},
          {Vec3{-1, 1, 0}, std::sqrt(3.0)/4.0},
          {Vec3{ 0, 1, 0}, std::sqrt(3.0)/4.0},
          {Vec3{ 0,-1, 0},-std::sqrt(3.0)/4.0},
          {Vec3{-1, 2, 0}, -1.0/(6.0*std::sqrt(3.0))},
          {Vec3{ 1,-2, 0}, 1.0/(6.0*std::sqrt(3.0))},
          {Vec3{ 1, 1, 0},-1.0/(12.0*std::sqrt(3.0))},
          {Vec3{ 2,-1, 0}, 1.0/(12.0*std::sqrt(3.0))},
          {Vec3{-2, 1, 0}, -1.0/(12.0*std::sqrt(3.0))},
          {Vec3{-1,-1, 0}, 1.0/(12.0*std::sqrt(3.0))}
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
      J.interaction_vector_cart = globals::lattice->fractional_to_cartesian(data.first);
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

  fd_weights_.clear();
  fd_weights_.resize(globals::num_spins);

  for (int i = 0; i < globals::num_spins; ++i) {
    const auto &dx_idx = dx_indices_[i];
    const auto &dx_val = dx_values_[i];
    const auto &dy_idx = dy_indices_[i];
    const auto &dy_val = dy_values_[i];

    // Build a temporary map idx -> (wx, wy).
    // Since stencil sizes are tiny (<= 12), a simple vector search is fine.
    std::vector<FDWeights> tmp;
    tmp.reserve(dx_idx.size() + dy_idx.size());

    auto add_weight = [&](int idx, double wx, double wy) {
      for (auto &w : tmp) {
        if (w.index == idx) {
          w.wx += wx;
          w.wy += wy;
          return;
        }
      }
      tmp.push_back(FDWeights{idx, wx, wy});
    };

    for (std::size_t n = 0; n < dx_idx.size(); ++n) {
      add_weight(dx_idx[n], dx_val[n], 0.0);
    }
    for (std::size_t n = 0; n < dy_idx.size(); ++n) {
      add_weight(dy_idx[n], 0.0, dy_val[n]);
    }

    fd_weights_[i] = std::move(tmp);
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

double jams::CVarTopologicalChargeFiniteDiff::local_topological_charge(int i) const {
  const auto &weights = fd_weights_[i];

  Vec3 ds_x{0.0, 0.0, 0.0};
  Vec3 ds_y{0.0, 0.0, 0.0};

  for (const auto &w : weights) {
    const Vec3 s = montecarlo::get_spin(w.index);
    if (w.wx != 0.0) {
      ds_x = fma(w.wx, s, ds_x);
    }
    if (w.wy != 0.0) {
      ds_y = fma(w.wy, s, ds_y);
    }
  }

  const Vec3 s_i = montecarlo::get_spin(i);
  return scalar_triple_product(s_i, ds_x, ds_y);
}

double jams::CVarTopologicalChargeFiniteDiff::local_topological_charge_difference_for_site(
    const std::vector<jams::FDWeights> &weights,
    int site_index,
    int moving_index,
    const Vec3 &spin_initial,
    const Vec3 &spin_final) const {

  Vec3 ds_x_init{0.0, 0.0, 0.0};
  Vec3 ds_y_init{0.0, 0.0, 0.0};
  Vec3 ds_x_final{0.0, 0.0, 0.0};
  Vec3 ds_y_final{0.0, 0.0, 0.0};

  for (const auto &w : weights) {
    Vec3 s_cur;
    Vec3 s_init;
    Vec3 s_final;

    if (w.index == moving_index) {
      // For the moving spin, use the provided initial/final values.
      s_init = spin_initial;
      s_final = spin_final;
    } else {
      // For all other spins, the configuration is unchanged.
      s_cur = montecarlo::get_spin(w.index);
      s_init = s_cur;
      s_final = s_cur;
    }

    if (w.wx != 0.0) {
      ds_x_init  = fma(w.wx, s_init,  ds_x_init);
      ds_x_final = fma(w.wx, s_final, ds_x_final);
    }
    if (w.wy != 0.0) {
      ds_y_init  = fma(w.wy, s_init,  ds_y_init);
      ds_y_final = fma(w.wy, s_final, ds_y_final);
    }
  }

  // Central spin at this site.
  Vec3 s_i_init;
  Vec3 s_i_final;

  if (site_index == moving_index) {
    s_i_init  = spin_initial;
    s_i_final = spin_final;
  } else {
    const Vec3 s_i = montecarlo::get_spin(site_index);
    s_i_init  = s_i;
    s_i_final = s_i;
  }

  const double q_init  = scalar_triple_product(s_i_init,  ds_x_init,  ds_y_init);
  const double q_final = scalar_triple_product(s_i_final, ds_x_final, ds_y_final);

  return q_final - q_init;
}


double jams::CVarTopologicalChargeFiniteDiff::topological_charge_difference(int index,
                                                                  const Vec3 &spin_initial,
                                                                  const Vec3 &spin_final) const {
  // We calculate the difference in the topological charge between the initial
  // and final spin states. When one spin is changed, the topological charge on
  // all sites connected to it through the finite difference stencil also changes.
  // We therefore calculate the difference of the topological charge of the whole
  // stencil, but without modifying the global spin configuration.

  double delta_charge = 0.0;

  // Contribution from the site whose spin is being changed.
  delta_charge += local_topological_charge_difference_for_site(
      fd_weights_[index], index, index, spin_initial, spin_final);

  // Contributions from all stencil neighbours that depend on this spin.
  const auto &stencil_nbrs = stencil_neighbour_indices_[index];
  for (int n : stencil_nbrs) {
    delta_charge += local_topological_charge_difference_for_site(
        fd_weights_[n], n, index, spin_initial, spin_final);
  }

  return delta_charge / (4.0 * kPi);
}