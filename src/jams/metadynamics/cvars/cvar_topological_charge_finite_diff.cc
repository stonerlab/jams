// cvar_topological_charge_finite_diff.cc                              -*-C++-*-
#include <jams/metadynamics/cvars/cvar_topological_charge_finite_diff.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>

jams::CVarTopologicalChargeFiniteDiff::CVarTopologicalChargeFiniteDiff(
    const libconfig::Setting &settings) {
  // In all of this code, we assume stacking along z, so the motif offset can also be
  // used to index the layers.
  int total_layers = globals::lattice->num_motif_atoms();

  if (settings.exists("material")) {
    std::string material = config_optional<std::string>(settings, "material", "all");
    if (!globals::lattice->material_exists(material)) {
      throw std::runtime_error("Invalid material specified in topological charge collective variable.");
    }
    selected_material_id_ = globals::lattice->material_id(material);

    num_selected_layers_ = 0;
    for (auto i = 0; i < total_layers; ++i) {
      // iterate over layers, determining whether each motif atom isq
      // of the specified material or not.
      if (globals::lattice->motif_atom(i).material_index == selected_material_id_) {
        num_selected_layers_ += 1;
      }
    }

  } else {
    selected_material_id_ = -1;
    num_selected_layers_ = globals::lattice->num_motif_atoms();
  }

  assert((selected_material_id_ >= 0 && selected_material_id_ < globals::lattice->num_materials()) || selected_material_id_ == -1);
  assert(num_selected_layers_ > 0);

  // true if a and b are equal to the lattice a and b vectors.
  auto lattice_equal = [&](Vec3 a, Vec3 b) {
    return approximately_equal(globals::lattice->a(), a, defaults::lattice_tolerance)
    && approximately_equal(globals::lattice->b(), b, defaults::lattice_tolerance);
  };

  // Detect if we have a square or hexagonal lattice in the plane (all other
  // lattices are unsupported)
  enum class LatticeShape {Unsupported, Square, HexagonalAcute, HexagonalObtuse};

  LatticeShape lattice_shape = LatticeShape::Unsupported;
  if (lattice_equal({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0})) {
    lattice_shape = LatticeShape::Square;
  } else if (lattice_equal({1.0, 0.0, 0.0}, {0.5, sqrt(3)/2, 0.0})) {
    lattice_shape = LatticeShape::HexagonalAcute;
  } else if (lattice_equal({1.0, 0.0, 0.0}, {-0.5, sqrt(3)/2, 0.0})) {
    lattice_shape = LatticeShape::HexagonalObtuse;
  } else {
    throw std::runtime_error("Metadynamics CV 'topological_charge_finite_diff' "
                             "requires the lattice to be either square or triangular.");
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

    if (lattice_shape == LatticeShape::Square) {
      dx_interaction_data = {
          {Vec3{ 1, 1, 0}, 1.0/16.0},
          {Vec3{ 1,-1, 0}, 1.0/16.0},
          {Vec3{ 1, 0, 0}, 3.0/8.0},
          {Vec3{-1, 0, 0},-3.0/8.0},
          {Vec3{-1, 1, 0},-1.0/16.0},
          {Vec3{-1,-1, 0},-1.0/16.0},
      };
    }
    if (lattice_shape == LatticeShape::HexagonalAcute) {
      dx_interaction_data = {
          {Vec3{ 1,-1, 0}, 1.0/6.0},
          {Vec3{-1, 1, 0},-1.0/6.0},
          {Vec3{ 0, 1, 0}, 1.0/6.0},
          {Vec3{ 0,-1, 0},-1.0/6.0},
          {Vec3{ 1, 0, 0}, 1.0/3.0},
          {Vec3{-1, 0, 0},-1.0/3.0}
      };
    }
    if (lattice_shape == LatticeShape::HexagonalObtuse) {
      dx_interaction_data = {
          {Vec3{0,1,0}, -1.0/6.0},
          {Vec3{1,1,0}, 1.0/6.0},
          {Vec3{1,0,0}, 1.0/3.0},
          {Vec3{0,-1,0}, 1.0/6.0},
          {Vec3{-1,-1,0}, -1.0/6.0},
          {Vec3{-1,0,0}, -1.0/3.0},
      };
    }

    std::vector<InteractionData> interaction_template;
    for (auto &data: dx_interaction_data) {
      InteractionData J;
      J.unit_cell_pos_i = 0;
      J.unit_cell_pos_j = 0;
      J.type_i = globals::lattice->material_name(
          globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
      J.type_j = globals::lattice->material_name(
          globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);
      J.r_ij = globals::lattice->fractional_to_cartesian(data.first);
      J.J_ij[0][0] = data.second;
      interaction_template.push_back(J);
    }

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

    if (lattice_shape == LatticeShape::Square) {
      dy_interaction_data = {
          {Vec3{ 1, 1, 0}, 1.0/16.0},
          {Vec3{-1, 1, 0}, 1.0/16.0},
          {Vec3{ 0, 1, 0}, 3.0/8.0},
          {Vec3{ 0,-1, 0},-3.0/8.0},
          {Vec3{ 1,-1, 0},-1.0/16.0},
          {Vec3{-1,-1, 0},-1.0/16.0},
      };
    }
    if (lattice_shape == LatticeShape::HexagonalAcute) {
      dy_interaction_data = {
          {Vec3{ 1,-1, 0},-sqrt(3.0)/6.0},
          {Vec3{-1, 1, 0}, sqrt(3.0)/6.0},
          {Vec3{ 0, 1, 0}, sqrt(3.0)/6.0},
          {Vec3{ 0,-1, 0},-sqrt(3.0)/6.0},
      };
    }
    if (lattice_shape == LatticeShape::HexagonalObtuse) {
      dy_interaction_data = {
          {Vec3{0,1,0}, sqrt(3.0)/6.0},
          {Vec3{1,1,0}, sqrt(3.0)/6.0},
          {Vec3{0,-1,0}, -sqrt(3.0)/6.0},
          {Vec3{-1,-1,0}, -sqrt(3.0)/6.0},
      };
    }

      std::vector<InteractionData> interaction_template;
    for (auto &data: dy_interaction_data) {
      InteractionData J;
      J.unit_cell_pos_i = 0;
      J.unit_cell_pos_j = 0;
      J.type_i = globals::lattice->material_name(
          globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
      J.type_j = globals::lattice->material_name(
          globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);
      J.r_ij = ::globals::lattice->fractional_to_cartesian(data.first);
      J.J_ij[0][0] = data.second;
      interaction_template.push_back(J);
    }
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
  // check if the spin is of relevant material
  if (selected_material_id_==-1 || globals::lattice->atom_material_id(i)==selected_material_id_) {
    const double trial_value = cached_value() + topological_charge_difference(i, spin_initial, spin_trial);
    set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);
    return trial_value;
  }

  set_cache_values(i, spin_initial, spin_trial, cached_value(), cached_value());

  return cached_value();  //Used in CollectiveVariable
}



double jams::CVarTopologicalChargeFiniteDiff::calculate_expensive_cache_value() {

  double topological_charge = 0.0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    // include the spin if it is of relevant material, or if we consider all
    // spins in the system.
    if (selected_material_id_==-1 || globals::lattice->atom_material_id(i)==selected_material_id_) {
      topological_charge += local_topological_charge(i);
    }

  }

  return topological_charge / (4.0 * kPi * num_selected_layers_);
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

  return (final_charge - initial_charge) / (4.0 * kPi * num_selected_layers_);
}