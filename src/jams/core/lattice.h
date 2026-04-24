// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H


/// The definitions and nomenclature follow Kittel's Introduction to Solid State Physics.
///
/// The *basis* is made of three crystal axes a1, a2, a3 and a set of basis sites with a Cartesian position
/// r_i = x_i a1 + y_i a2 + z_i a3 where 0 <= x_i, y_i, z_i < 1. The coordinate (x_i, y_i, z_i) is called a
/// *fractional coordinate*.
///
/// The lattice is formed by repeated translation of the basis by translation vectors
///
/// T = u1 a1 + u2 a2 + u3 a3, where u1, u2, u3 are integers. (u1, u2, u3) is therefore a 
///
///

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

class Lattice : public Base {
public:
    using MaterialMap = NameIdMap<Material>;
    using ImpurityMap = std::map<size_t, Impurity>;

    Lattice() = default;

    ~Lattice();

    void init_from_config(const libconfig::Config &pConfig);

    int size(int dimension) const;      // number of unitcell in each dimension
    Vec<int, 3> size() const;

    /// @brief Lattice parameter in meters
    double parameter() const;

    Vec<double, 3> a1() const;
    Vec<double, 3> a2() const;
    Vec<double, 3> a3() const;

    const Cell& get_supercell();
    const Cell& get_unitcell();
    const Mat<double, 3, 3>& get_global_rotation_matrix();

    Vec<double, 3> displacement(const Vec<double, 3> &position_i_cart, const Vec<double, 3> &position_j_cart) const;

    Vec<double, 3> displacement(const unsigned& lattice_site_i, const unsigned&lattice_site_j) const;

    bool is_periodic(int dimension) const;
    const Vec<bool, 3> & periodic_boundaries() const;

    /// Return the Atom data for a basis site
    const Atom& basis_site_atom(const int &basis_site_index) const;

    /// Return the number of sites in the basis
    int num_basis_sites() const;

    /// Return the number of materials
    int num_materials() const;

    /// Return the Material data for a given material index
    const Material &material(const int &material_index) const;

    /// Return the Material name for a given material index
    std::string material_name(int material_index) const;
    int material_index(const std::string &material_name) const;
    bool material_exists(const std::string &material_name) const;

    int lattice_site_material_id(int lattice_site_index) const;           // integer index of the material
    std::string lattice_site_material_name(int lattice_site_index) const; // name of the material of atom lattice_site_index
    const Vec<double, 3> & lattice_site_position_cart(int lattice_site_index) const;             // cartesian position in the supercell
    const Vec<double, 3> & lattice_site_vector_frac(int lattice_site_index) const;             // fractional position in the supercell

    const std::vector<Vec<double, 3>>& lattice_site_positions_cart() const;

    /// Return the basis site index of the given lattice site
    unsigned lattice_site_basis_index(int lattice_site_index) const;

    /// Return a vector of symmetry operations (rotation matrices) of the point group at the given lattice site
    const std::vector<Mat<double, 3, 3>>& lattice_site_point_group_symops(int lattice_site_index);

    // supercell

    unsigned num_cells() const {
      return cell_offsets_.size();
    }

    const Vec<int, 3>& cell_offset(int lattice_site_index) const {
      return cell_offsets_[lattice_site_to_cell_lookup_[lattice_site_index]];
    }

    int cell_containing_atom(int lattice_site_index) const {
      return lattice_site_to_cell_lookup_[lattice_site_index];
    }

    const Vec<double, 3>& cell_center(int cell_index) const {
      return cell_centers_[cell_index];
    }


    double max_interaction_radius() const;

    bool has_impurities() const;

    // TODO: remove rmax
    const Vec<double, 3> & rmax() const;

    Vec<double, 3> generate_cartesian_lattice_position_from_fractional(const Vec<double, 3> &basis_site_position_frac,
                                                             const Vec<int, 3> &lattice_translation_vector) const;

    Vec<double, 3> generate_image_position(const Vec<double, 3> &unit_cell_cart_pos, const Vec<int, 3> &image_vector) const;

    // Generates a list of points symmetric to r_cart based on the local point group symmetry of basis_site_index
    std::vector<Vec<double, 3>> generate_symmetric_points(int basis_site_index, const Vec<double, 3> &r_cart, const double &tolerance);

    Vec<double, 3> cartesian_to_fractional(const Vec<double, 3> &r_cart) const;

    Vec<double, 3> fractional_to_cartesian(const Vec<double, 3> &r_frac) const;

    // Returns true if the points are a symmetry complete set (the symmetry operations on each point, generate
    // a point in the set), based on the local point group operations of the given motif position.
    bool is_a_symmetry_complete_set(int motif_index, const std::vector<Vec<double, 3>> &points, const double &tolerance);

    // lookup the site index but unit cell integer coordinates and motif offset
    int site_index_by_unit_cell(const int &i, const int &j, const int &k, const int &m) const;

    bool apply_boundary_conditions(Vec<int, 3> &pos) const;

    bool apply_boundary_conditions(int &a, int &b, int &c) const;

    const Vec<int, 3> &kspace_size() const;

private:
    void read_materials_from_config(const libconfig::Setting &settings);
    ImpurityMap read_impurities_from_config(const libconfig::Setting &settings);

    void read_unitcell_from_config(const libconfig::Setting &settings);

    void read_lattice_from_config(const libconfig::Setting &settings);

    void read_basis_sites_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format);

    void read_basis_sites_from_file(const std::string &filename, CoordinateFormat coordinate_format);

    void init_unit_cell(const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings);

    void generate_supercell(const libconfig::Setting &lattice_settings);

    void global_rotation(const Mat<double, 3, 3> &rotation_matrix);

    void global_reorientation(const Vec<double, 3> &reference, const Vec<double, 3> &vector);

    void calc_symmetry_operations();

    bool symops_enabled_;

    Mat<double, 3, 3> global_orientation_matrix_ = kIdentityMat3;

    Cell unitcell;
    Cell supercell;
    double lattice_parameter;

    Vec<int, 3> lattice_dimensions_;
    Vec<bool, 3> lattice_periodic;

    std::vector<Atom> basis_sites_;
    std::vector<Atom> lattice_sites_;

    // store both cartesian and fractional to avoid matrix multiplication when we want fractional
    std::vector<Vec<double, 3>> lattice_site_positions_cart_;
    std::vector<Vec<double, 3>> lattice_site_positions_frac_;

    std::vector<int>   lattice_site_to_cell_lookup_;     // index is a spin and the data is the unit cell that spin belongs to
    std::vector<Vec<double, 3>>  cell_centers_;
    std::vector<Vec<int, 3>> cell_offsets_;

    MaterialMap       materials_;
    unsigned          impurity_seed_;
    ImpurityMap       impurity_map_;
    jams::MultiArray<int, 4> lattice_map_;

    jams::MultiArray<int, 3> kspace_map_;
    Vec<int, 3> kspace_size_;
    Vec<double, 3> rmax_;

    SpglibDataset *spglib_dataset_ = nullptr;

    std::vector<Mat<double, 3, 3>> sym_rotations_;
    std::vector<Vec<double, 3>> sym_translations_;

    std::vector<std::vector<Mat<double, 3, 3>>> basis_site_point_group_symops_;
};

namespace jams {
    //
    // Returns the maximum interaction length of the parallelepiped described by
    // vectors a1, a2, a3 were periodic boundaries may be defined.
    //
    // - 3D periodic returns the radius of a1 sphere
    // - 2D periodic returns the radius of a1 cylinder in the periodic plane
    // - 1D periodic returns half the length along a1 line in the periodic direction
    // - Non periodic returns the maximum length across the parallelepiped
    //
    double maximum_interaction_length(const Vec<double, 3>& a1, const Vec<double, 3>& a2, const Vec<double, 3>& a3, const Vec<bool, 3>& periodic_boundaries);
}

#endif // JAMS_CORE_LATTICE_H
