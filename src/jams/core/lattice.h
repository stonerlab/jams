// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H

extern "C" {
#include "spglib.h"
}


#include <map>
#include <string>
#include <vector>
#include <iosfwd>
#include <cmath>
#include <cassert>
#include <libconfig.h++>
#include "jams/containers/name_id_map.h"
#include "jams/containers/multiarray.h"

#include "jams/core/types.h"
#include "jams/core/base.h"
#include "jams/containers/neartree.h"
#include "jams/containers/material.h"
#include "jams/containers/cell.h"

struct Impurity {
    size_t   material;
    double   fraction;
};

class DisplacementCalculator {
public:
    DisplacementCalculator() = default;
    DisplacementCalculator(const Cell& cell) : cell_(cell) {};


    inline void insert(const Vec3& cartesian_position) {
      fractional_positions_.push_back(cell_.inverse_matrix() * cartesian_position);
    }

    __attribute__((hot))
    inline Vec3 operator()(const unsigned& i, const unsigned& j) const {
      Vec3 dr = (fractional_positions_[j] - fractional_positions_[i]);

      if (cell_.periodic(0)) {
        dr[0] = dr[0] - std::trunc(2.0 * dr[0]);
      }

      if (cell_.periodic(1)) {
        dr[1] = dr[1] - std::trunc(2.0 * dr[1]);
      }

      if (cell_.periodic(2)) {
        dr[2] = dr[2] - std::trunc(2.0 * dr[2]);
      }

      return cell_.matrix() * dr;
    }

    inline Vec3 operator()(const Vec3& r_cart_i, const Vec3& r_cart_j) const {
      Vec3 dr = cell_.inverse_matrix() * (r_cart_j - r_cart_i);

      if (cell_.periodic(0)) {
        dr[0] = dr[0] - std::trunc(2.0 * dr[0]);
      }

      if (cell_.periodic(1)) {
        dr[1] = dr[1] - std::trunc(2.0 * dr[1]);
      }

      if (cell_.periodic(2)) {
        dr[2] = dr[2] - std::trunc(2.0 * dr[2]);
      }

      return cell_.matrix() * dr;
    }

private:
    Cell cell_ = kIdentityMat3;
    std::vector<Vec3> fractional_positions_;
};

class Lattice : public Base {
public:
    using MaterialMap = NameIdMap<Material>;
    using ImpurityMap = std::map<size_t, Impurity>;

    Lattice() = default;

    ~Lattice();

    void init_from_config(const libconfig::Config &pConfig);

    int size(int i) const;      // number of unitcell in each dimension
    Vec3i size() const;

    double parameter() const;   // [m]
    double volume() const;      // [m^3]

    Vec3 a() const;
    Vec3 b() const;
    Vec3 c() const;

    const Cell& get_supercell();
    const Cell& get_unitcell();
    const Mat3& get_global_rotation_matrix();

    Vec3 displacement(const Vec3 &r_i, const Vec3 &r_j) const;

    Vec3 displacement(const unsigned& i, const unsigned&j) const;

    bool is_periodic(int i) const;
    Vec3b periodic_boundaries() const;

    const Atom& motif_atom(const int &i) const;
    int num_motif_atoms() const;

    int num_materials() const;
    const Material &material(const int &i) const;
    std::string material_name(int uid);
    int material_id(const std::string &name);
    bool material_exists(const std::string &name);

    int atom_material_id(const int &i) const;           // integer index of the material
    Vec3 atom_position(const int &i) const;             // cartesian position in the supercell
    unsigned atom_motif_position(const int &i) const;   // integer index within the motif
    void atom_neighbours(const int &i, const double &r_cutoff, std::vector<Atom> &neighbours) const;
    constexpr int atom_unitcell(const int &i) const;


    // supercell

    constexpr unsigned num_cells() const {
      return cell_offsets_.size();
    }

    const Vec3i& cell_offset(const int &i) const {
      return cell_offsets_[atom_to_cell_lookup_[i]];
    }

    const int& cell_containing_atom(const int &i) const {
      return atom_to_cell_lookup_[i];
    }

    const Vec3& cell_center(const int &i) const {
      return cell_centers_[i];
    }


    double max_interaction_radius() const;

    bool has_impurities() const;

    // TODO: remove rmax
    Vec3 rmax() const;

    Vec3 generate_cartesian_lattice_position_from_fractional(const Vec3 &unit_cell_frac_pos,
                                                             const Vec3i &translation_vector) const;

    Vec3 generate_image_position(const Vec3 &unit_cell_cart_pos, const Vec3i &image_vector) const;

    std::vector<Vec3> generate_symmetric_points(const Vec3 &r, const double &tolerance) const;

    Vec3 cartesian_to_fractional(const Vec3 &r_cart) const;

    Vec3 fractional_to_cartesian(const Vec3 &r_frac) const;

    bool is_a_symmetry_complete_set(const std::vector<Vec3> &points, const double &tolerance) const;

    // lookup the site index but unit cell integer coordinates and motif offset
    int site_index_by_unit_cell(const int &i, const int &j, const int &k, const int &m) const;

    bool apply_boundary_conditions(Vec3i &pos) const;

    bool apply_boundary_conditions(int &a, int &b, int &c) const;

    bool apply_boundary_conditions(Vec4i &pos) const;


    const Vec3i &kspace_size() const;

private:
    void read_materials_from_config(const libconfig::Setting &settings);
    ImpurityMap read_impurities_from_config(const libconfig::Setting &settings);

    void read_unitcell_from_config(const libconfig::Setting &settings);

    void read_lattice_from_config(const libconfig::Setting &settings);

    void read_motif_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format);

    void read_motif_from_file(const std::string &filename, CoordinateFormat coordinate_format);

    void init_unit_cell(const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings);

    void generate_supercell(const libconfig::Setting &lattice_settings);

    void global_rotation(const Mat3 &rotation_matrix);

    void global_reorientation(const Vec3 &reference, const Vec3 &vector);

    void calc_symmetry_operations();

    using NeartreeFunctorType = std::function<double(const Atom &, const Atom &)>;
    NearTree<Atom, NeartreeFunctorType> *neartree_ = nullptr;


    bool symops_enabled_;

    Mat3 global_orientation_matrix_ = kIdentityMat3;

    Cell unitcell;
    Cell supercell;
    double lattice_parameter;

    Vec3i lattice_dimensions;
    Vec3b lattice_periodic;

    std::vector<Atom> motif_;
    std::vector<Atom> atoms_;

    std::vector<int>   atom_to_cell_lookup_;     // index is a spin and the data is the unit cell that spin belongs to
    std::vector<Vec3>  cell_centers_;
    std::vector<Vec3i> cell_offsets_;

    DisplacementCalculator displacement_calculator;

    MaterialMap       materials_;
    unsigned          impurity_seed_;
    ImpurityMap       impurity_map_;
    jams::MultiArray<int, 4> lattice_map_;

    jams::MultiArray<int, 3> kspace_map_;
    Vec3i kspace_size_;
    Vec3 rmax_;

    SpglibDataset *spglib_dataset_ = nullptr;
    std::vector<Mat3> rotations_;

};

#endif // JAMS_CORE_LATTICE_H
