// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H

extern "C"{
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

namespace jams {
    Mat3 unit_cell_matrix(const Vec3& a1, const Vec3& a2,  const Vec3& a3);
    Mat3 inverse_unit_cell_matrix(const Mat3& unit_cell_matrix);
}

class DistanceMetric {

public:
  DistanceMetric(const Mat3 unit_cell, const Vec3i cell_count, const Vec3b cell_pbc)
  : super_cell_pbc_(cell_pbc)
  {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            super_unit_cell_[i][j] = unit_cell[i][j] * cell_count[j];
        }
    }

    super_unit_cell_inv_ = jams::inverse_unit_cell_matrix(super_unit_cell_);
  }

  inline Vec3 minimum_image(const Vec3& r_i, const Vec3& r_j) const {
    Vec3 dr(super_unit_cell_inv_ * (r_i - r_j));

    #pragma unroll
    for (int n = 0; n < 3; ++n) {
      if (super_cell_pbc_[n]) {
        // W. Smith, CCP5 Information Quarterly for Computer Simulation of Condensed Phases (1989).
        dr[n] = dr[n] - trunc(2.0 * dr[n]);
      }
    }

    return (super_unit_cell_ * dr);
  }

  inline Vec3 displacement(const Atom& a, const Atom& b) const {
    return minimum_image(a.pos, b.pos);
  }

  inline double operator()(const Atom& a, const Atom& b) const {
    return abs(displacement(a, b));
  }

private:
    const Vec3b super_cell_pbc_;
    Mat3 super_unit_cell_;
    Mat3 super_unit_cell_inv_;
};


class Lattice {
  public:
    Lattice();
    ~Lattice();

    void init_from_config(const libconfig::Config& pConfig);

    inline double  parameter() const;      ///< lattice parameter (m)
    inline double  volume() const;      ///< volume (m^3)

    inline int     size(const int i) const;           ///< integer number of unitcell in each lattice vector
    inline int     num_unit_cells() const; ///< number of unit cells in the whole lattice
    inline int     num_unit_cell_positions() const;         ///< number atomic positions in the unit cell
    inline Vec3    unit_cell_vector(const int i) const;     ///< lattice vectors
    inline Vec3    inv_unit_cell_vector(const int i) const;     ///< lattice vectors
    inline Vec3    unit_cell_position(const int i) const;   ///< position i in fractional coordinates
    inline Vec3    unit_cell_position_cart(const int i) const;   ///< position i in fractional coordinates
    inline Material     unit_cell_material(const int i) ;     ///< uid of material at position i
    inline int     unit_cell_material_uid(const int i);     ///< uid of material at position i
    inline string  unit_cell_material_name(const int i);     ///< uid of material at position i

    inline int     num_materials() const;                           ///< number of unique materials in lattice
    inline int     num_of_material(const int i) const;               ///< number of lattice sites of material i
    inline string  material_name(const int uid);  ///< material string name from uid
    inline int     material_id(const string &name);  ///< material string name from uid

    inline int     atom_material(const int i) const;  ///< uid of material at lattice site i
    inline Vec3    atom_position(const int i) const;
    inline void    atom_neighbours(const int i, const double r_cutoff, std::vector<Atom> &neighbours) const;
//           void    atom_nearest_neighbours(const int i, const double r_cutoff, std::vector<Atom> &neighbours) const;

    inline double  distance(const int i, const int j) const;
    inline Vec3    displacement(const int i, const int j) const;
    inline Vec3    minimum_image(const Vec3& r_i, const Vec3& r_j) const;
           double  maximum_interaction_radius() const; ///< the maximum radius which can be used that avoids self interaction
    inline Vec3 rmax() const;
    inline Vec3 rmin() const;
    Vec3 generate_position(const Vec3 unit_cell_frac_pos, const Vec3i translation_vector) const;
    Vec3 generate_image_position(const Vec3 unit_cell_cart_pos, const Vec3i image_vector) const;
    Vec3 generate_fractional_position(const Vec3 unit_cell_frac_pos, const Vec3i translation_vector) const;
    std::vector<Vec3> generate_symmetric_points(const Vec3& r, const double tolerance) const;

    inline Vec3 cartesian_to_fractional(const Vec3& r_cart) const;
    inline Vec3 fractional_to_cartesian(const Vec3& r_frac) const;

    inline int      num_sym_ops() const;
          Vec3      sym_rotation(const int i, const Vec3 r_frac) const;
          bool      is_a_symmetry_complete_set(const std::vector<Vec3>& points, const double tolerance) const;

    // --------------------------------------------------------------------------
    // lattice vector functions
    // --------------------------------------------------------------------------


    // --------------------------------------------------------------------------
    // lattice vector functions
    // --------------------------------------------------------------------------
    inline int num_unit_cells(const int i) const {
        assert(i < 3);
        return super_cell.size[i];
    }

    // lookup the site index but unit cell integer coordinates and motif offset
    inline int site_index_by_unit_cell(const int i, const int j, const int k, const int m) const {
        assert(i < num_unit_cells(0));
        assert(i >= 0);
        assert(j < num_unit_cells(1));
        assert(j >= 0);
        assert(k < num_unit_cells(2));
        assert(k >= 0);
        assert(m < num_unit_cell_positions());
        assert(m >= 0);

        return lattice_map_(i, j, k, m);
    }

    inline bool is_bulk_system() const {
        return (super_cell.periodic[0] && super_cell.periodic[1] && super_cell.periodic[2]);
    }

    inline bool is_open_system() const {
        return (!super_cell.periodic[0] && !super_cell.periodic[1] && !super_cell.periodic[2]);
    }

    inline bool is_periodic(const int i) const {
        assert(i < 3);
        return super_cell.periodic[i];
    }

    bool apply_boundary_conditions(Vec3i& pos) const;
    bool apply_boundary_conditions(int &a, int &b, int &c) const;
    bool apply_boundary_conditions(Vec4i& pos) const;

    const Vec3i& super_cell_pos(const int i) const {
        return lattice_super_cell_pos_(i);
    }

    // --------------------------------------------------------------------------
    // kspace functions
    // --------------------------------------------------------------------------
    const Vec3i& kspace_size() const {
        return kspace_size_;
    }

    void load_spin_state_from_hdf5(std::string &filename);

  private:
    void read_motif_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format);
    void read_motif_from_file(const std::string &filename, CoordinateFormat coordinate_format);


    void init_unit_cell(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings);
    void init_lattice_positions(const libconfig::Setting &lattice_settings);
    void init_nearest_neighbour_list(const double r_cutoff, const bool prune = false);
    void calc_symmetry_operations();
    void set_spacegroup(const int hall_number);

    bool is_debugging_enabled_;
    bool symops_enabled_;

    SuperCell                       super_cell;
    DistanceMetric*                 metric_;
    NearTree<Atom, DistanceMetric>* neartree_;

    std::vector<Atom> motif_;
    std::vector<Atom> atoms_;

    std::vector<int>            num_of_material_;
    std::map<string, Material>  material_name_map_;
    std::map<int, Material>     material_id_map_;

    std::vector<std::string>    lattice_materials_;
    jblib::Array<Vec3i, 1>        lattice_super_cell_pos_;
    jblib::Array<int, 4>        lattice_map_;

    jblib::Array<int, 3>        kspace_map_;
    Vec3i            kspace_size_;
    jblib::Array<int, 2>        kspace_inv_map_;
    std::vector< Vec3 > lattice_positions_;
    std::vector< Vec3 > lattice_frac_positions_;
    std::vector<int>            lattice_material_num_;
    Vec3         rmax_;
    Vec3         rmin_;

    // spglib
    SpglibDataset *spglib_dataset_;
    std::vector< Mat3 > rotations_;

};

inline double
Lattice::parameter() const {
    return super_cell.parameter;
}

inline double
Lattice::volume() const {
    return std::abs(determinant(super_cell.unit_cell))*std::pow(super_cell.parameter, 3);
}

inline int
Lattice::size(const int i) const {
    return super_cell.size[i];
}

inline int
Lattice::num_unit_cells() const {
    return product(super_cell.size);
}

inline int
Lattice::num_unit_cell_positions() const {
    return motif_.size();
}

inline Vec3
Lattice::unit_cell_vector(const int i) const {
    assert(i < 3 && i >= 0);
    // vectors are columns of the unit cell matrix
    return {super_cell.unit_cell[0][i], super_cell.unit_cell[1][i], super_cell.unit_cell[2][i]};
}

inline Vec3
Lattice::inv_unit_cell_vector(const int i) const {
    assert(i < 3 && i >= 0);
    // vectors are columns of the unit cell matrix
    return {super_cell.unit_cell_inv[0][i], super_cell.unit_cell_inv[1][i], super_cell.unit_cell_inv[2][i]};
}

inline Vec3
Lattice::unit_cell_position(const int i) const {
    assert(i < num_unit_cell_positions());
    return motif_[i].pos;
}

inline Vec3
Lattice::unit_cell_position_cart(const int i) const {
    assert(i < num_unit_cell_positions());
    return super_cell.unit_cell * motif_[i].pos;
}

inline Material
Lattice::unit_cell_material(const int i) {
    assert(i < motif_.size());
    return material_id_map_[motif_[i].material];
}


inline int
Lattice::unit_cell_material_uid(const int i) {
    assert(i < motif_.size());
    return motif_[i].material;
}

inline std::string
Lattice::unit_cell_material_name(const int i) {
    assert(i < motif_.size());
    return material_id_map_[motif_[i].material].name;
}

inline int
Lattice::num_materials() const {
    return material_name_map_.size();
}

inline int
Lattice::num_of_material(const int i) const {
    return num_of_material_[i];
}

inline std::string
Lattice::material_name(const int uid) {
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

inline double
Lattice::distance(const int i, const int j) const {
    return (*metric_)(atoms_[i], atoms_[j]);
}

inline Vec3
Lattice::displacement(const int i, const int j) const {
    return (*metric_).displacement(atoms_[i], atoms_[j]);
}

inline Vec3
Lattice::minimum_image(const Vec3& r_i, const Vec3& r_j) const {
    return (*metric_).minimum_image(r_i, r_j);
}


inline Vec3
Lattice::cartesian_to_fractional(const Vec3& r_cart) const {
    return super_cell.unit_cell_inv*r_cart;
}

inline Vec3
Lattice::fractional_to_cartesian(const Vec3& r_frac) const {
    return super_cell.unit_cell*r_frac;
}

inline int
Lattice::num_sym_ops() const {
    if (symops_enabled_) {
        return spglib_dataset_->n_operations;
    } else {
        return 0;
    }
}

inline Vec3
Lattice::rmax() const {
    return rmax_;
};


inline Vec3
Lattice::rmin() const {
    return rmin_;
};




#endif // JAMS_CORE_LATTICE_H
