// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H

extern "C" {
#include "spglib/spglib.h"
}


#include <map>
#include <string>
#include <vector>
#include <iosfwd>
#include <cmath>
#include <cassert>
#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/neartree.h"
#include "jblib/containers/array.h"
#include "jams/core/material.h"
#include "jams/core/cell.h"

class Lattice {
public:
    Lattice() = default;

    ~Lattice();

    void init_from_config(const libconfig::Config &pConfig);

    inline int size(int i) const;      // number of unitcell in each dimension
    inline double parameter() const;   // [m]
    inline double volume() const;      // [m^3]

    inline Vec3 a() const;
    inline Vec3 b() const;
    inline Vec3 c() const;

    inline Vec3 displacement(const int i, const int j) const;
    inline Vec3 displacement(const Vec3 &r_i, const Vec3 &r_j) const;

    inline bool is_periodic(int i) const;

    inline int num_motif_positions() const;
    inline Vec3 motif_position_frac(int i) const;
    inline Vec3 motif_position_cart(int i) const;
    inline Material motif_material(int i) const;

    inline int num_materials() const;
    inline string material_name(int uid);
    inline int material_id(const string &name);

    inline int atom_material(const int i) const;
    inline Vec3 atom_position(const int i) const;
    inline void atom_neighbours(const int i, const double r_cutoff, std::vector<Atom> &neighbours) const;

    double max_interaction_radius() const;

    // TODO: remove rmax
    inline Vec3 rmax() const;

    Vec3 generate_position(const Vec3 unit_cell_frac_pos, const Vec3i translation_vector) const;

    Vec3 generate_image_position(const Vec3 unit_cell_cart_pos, const Vec3i image_vector) const;

    Vec3 generate_fractional_position(const Vec3 unit_cell_frac_pos, const Vec3i translation_vector) const;

    std::vector<Vec3> generate_symmetric_points(const Vec3 &r, const double tolerance) const;

    inline Vec3 cartesian_to_fractional(const Vec3 &r_cart) const;

    inline Vec3 fractional_to_cartesian(const Vec3 &r_frac) const;

    bool is_a_symmetry_complete_set(const std::vector<Vec3> &points, const double tolerance) const;

    // lookup the site index but unit cell integer coordinates and motif offset
    inline int site_index_by_unit_cell(const int i, const int j, const int k, const int m) const;

    bool apply_boundary_conditions(Vec3i &pos) const;

    bool apply_boundary_conditions(int &a, int &b, int &c) const;

    bool apply_boundary_conditions(Vec4i &pos) const;

    const Vec3i &super_cell_pos(const int i) const;

    const Vec3i &kspace_size() const;

    void load_spin_state_from_hdf5(std::string &filename);

private:
    void read_materials_from_config(const libconfig::Setting &settings);

    void read_unitcell_from_config(const libconfig::Setting &settings);

    void read_lattice_from_config(const libconfig::Setting &settings);

    void read_motif_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format);

    void read_motif_from_file(const std::string &filename, CoordinateFormat coordinate_format);

    void init_unit_cell(const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings);

    void init_lattice_positions(const libconfig::Setting &lattice_settings);

    void global_rotation(const Mat3 &rotation_matrix);

    void global_reorientation(const Vec3 &reference, const Vec3 &vector);

    void calc_symmetry_operations();

    using NeartreeFunctorType = std::function<double(const Atom &, const Atom &)>;
    NearTree<Atom, NeartreeFunctorType> *neartree_ = nullptr;


    bool symops_enabled_;

    Cell unitcell;
    Cell supercell;
    double lattice_parameter;

    Vec3i lattice_dimensions;
    Vec3b lattice_periodic;

    std::vector<Atom> motif_;
    std::vector<Atom> atoms_;

    std::vector<int> num_of_material_;
    std::map<string, Material> material_name_map_;
    std::map<int, Material> material_id_map_;

    std::vector<std::string> lattice_materials_;
    jblib::Array<Vec3i, 1> lattice_super_cell_pos_;
    jblib::Array<int, 4> lattice_map_;

    jblib::Array<int, 3> kspace_map_;
    Vec3i kspace_size_;
    std::vector<Vec3> lattice_positions_;
    Vec3 rmax_;

    SpglibDataset *spglib_dataset_ = nullptr;
    std::vector<Mat3> rotations_;

};

inline double Lattice::parameter() const {
  return lattice_parameter;
}

inline double Lattice::volume() const {
  return ::volume(supercell) * pow3(lattice_parameter);
}

inline int Lattice::size(int i) const {
  return lattice_dimensions[i];
}

inline int Lattice::num_motif_positions() const {
  return motif_.size();
}

inline Vec3 Lattice::a() const {
  return unitcell.a();
}

inline Vec3 Lattice::b() const {
  return unitcell.b();
}

inline Vec3 Lattice::c() const {
  return unitcell.c();
}

inline Vec3
Lattice::motif_position_frac(int i) const {
  assert(i < num_motif_positions());
  return motif_[i].pos;
}

inline Vec3
Lattice::motif_position_cart(int i) const {
  assert(i < num_motif_positions());
  return supercell.matrix() * motif_[i].pos;
}

inline Material
Lattice::motif_material(int i) const {
  assert(i < motif_.size());
  return material_id_map_.at(i);
}

inline int
Lattice::num_materials() const {
  return material_name_map_.size();
}

inline std::string
Lattice::material_name(int uid) {
  return material_id_map_.at(uid).name;
}

inline int
Lattice::material_id(const string &name) {
  return material_name_map_.at(name).id;
}

inline int
Lattice::atom_material(const int i) const {
  assert(i < lattice_materials_.size());
  return atoms_[i].material;
}

inline Vec3
Lattice::atom_position(const int i) const {
  return atoms_[i].pos;
}

inline void
Lattice::atom_neighbours(const int i, const double r_cutoff, std::vector<Atom> &neighbours) const {
  neartree_->find_in_radius(r_cutoff, neighbours, {i, atoms_[i].material, atoms_[i].pos});
}

inline Vec3
Lattice::displacement(const int i, const int j) const {
  return minimum_image(supercell, atom_position(i), atom_position(j));
}

inline Vec3
Lattice::displacement(const Vec3 &r_i, const Vec3 &r_j) const {
  return minimum_image(supercell, r_i, r_j);
}

inline Vec3
Lattice::cartesian_to_fractional(const Vec3 &r_cart) const {
  return unitcell.inverse_matrix() * r_cart;
}

inline Vec3
Lattice::fractional_to_cartesian(const Vec3 &r_frac) const {
  return unitcell.matrix() * r_frac;
}

inline Vec3
Lattice::rmax() const {
  return rmax_;
};

inline int Lattice::site_index_by_unit_cell(const int i, const int j, const int k, const int m) const {
  assert(i < lattice_dimensions[0]);
  assert(i >= 0);
  assert(j < lattice_dimensions[1]);
  assert(j >= 0);
  assert(k < lattice_dimensions[2]);
  assert(k >= 0);
  assert(m < num_motif_positions());
  assert(m >= 0);

  return lattice_map_(i, j, k, m);
}

inline bool Lattice::is_periodic(int i) const {
  return lattice_periodic[i];
}

inline const Vec3i &Lattice::super_cell_pos(const int i) const {
  return lattice_super_cell_pos_(i);
}

inline const Vec3i &Lattice::kspace_size() const {
  return kspace_size_;
}

#endif // JAMS_CORE_LATTICE_H
