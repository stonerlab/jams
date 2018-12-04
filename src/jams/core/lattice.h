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
#include "jams/containers/name_id_map.h"

#include "jams/core/types.h"
#include "jams/core/base.h"
#include "jams/containers/neartree.h"
#include "jblib/containers/array.h"
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

    int size(int i) const;      // number of unitcell in each dimension
    double parameter() const;   // [m]
    double volume() const;      // [m^3]

    Vec3 a() const;
    Vec3 b() const;
    Vec3 c() const;

    const Cell& get_supercell();
    const Cell& get_unitcell();
    const Mat3& get_global_rotation_matrix();

    Vec3 displacement(const Vec3 &r_i, const Vec3 &r_j) const;

    bool is_periodic(int i) const;

    const Atom& motif_atom(const int &i) const;
    int motif_size() const;

    int num_materials() const;
    const Material &material(const int &i) const;
    string material_name(int uid);
    int material_id(const string &name);
    bool material_exists(const string &name);

    int atom_material_id(const int &i) const;
    Vec3 atom_position(const int &i) const;
    void atom_neighbours(const int &i, const double &r_cutoff, std::vector<Atom> &neighbours) const;

    double max_interaction_radius() const;

    // TODO: remove rmax
    Vec3 rmax() const;

    Vec3 generate_position(const Vec3 &unit_cell_frac_pos, const Vec3i &translation_vector) const;

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

    const Vec3i &supercell_index(const int &i) const;

    const Vec3i &kspace_size() const;

    void load_spin_state_from_hdf5(std::string &filename);

private:
    void read_materials_from_config(const libconfig::Setting &settings);
    ImpurityMap read_impurities_from_config(const libconfig::Setting &settings);

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

    Mat3 global_orientation_matrix_ = kIdentityMat3;

    Cell unitcell;
    Cell supercell;
    double lattice_parameter;

    Vec3i lattice_dimensions;
    Vec3b lattice_periodic;

    std::vector<Atom> motif_;
    std::vector<Atom> atoms_;
    MaterialMap       materials_;
    unsigned          impurity_seed_;
    ImpurityMap       impurity_map_;
    std::vector<Vec3i> supercell_indicies_;
    jblib::Array<int, 4> lattice_map_;

    jblib::Array<int, 3> kspace_map_;
    Vec3i kspace_size_;
    Vec3 rmax_;

    SpglibDataset *spglib_dataset_ = nullptr;
    std::vector<Mat3> rotations_;

};


#endif // JAMS_CORE_LATTICE_H