//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_CELL_H
#define JAMS_CELL_H

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"

class Cell;

namespace jams {
    enum class LatticeSystem {
        triclinic,
        monoclinic,
        orthorhombic,
        tetragonal,
        rhombohedral,
        hexagonal,
        cubic
    };
}

double volume(const Cell& cell);
Cell   scale(const Cell& cell, const jams::Vec<int, 3>& size);
Cell   rotate(const Cell& cell, const jams::Mat<double, 3, 3>& rotation_matrix);

class Cell {
public:
    inline Cell() = default;
    inline Cell(const jams::Mat<double, 3, 3> &basis, const jams::Vec<bool, 3> pbc = {{true, true, true}});
    inline Cell(const jams::Vec<double, 3> &a1, const jams::Vec<double, 3> &a2, const jams::Vec<double, 3> &a3, const jams::Vec<bool, 3> pbc = {{true, true, true}});

    inline jams::Vec<double, 3> a1() const { return {matrix_[0][0], matrix_[1][0], matrix_[2][0]}; }
    inline jams::Vec<double, 3> a2() const { return {matrix_[0][1], matrix_[1][1], matrix_[2][1]}; }
    inline jams::Vec<double, 3> a3() const { return {matrix_[0][2], matrix_[1][2], matrix_[2][2]}; }

    inline jams::Vec<double, 3> b1() const { return inverse_matrix_[0]; }
    inline jams::Vec<double, 3> b2() const { return inverse_matrix_[1]; }
    inline jams::Vec<double, 3> b3() const { return inverse_matrix_[2]; }

    inline double alpha() const {return rad_to_deg(jams::angle(a2(), a3()));}
    inline double beta() const {return rad_to_deg(jams::angle(a3(), a1()));}
    inline double gamma() const {return rad_to_deg(jams::angle(a1(), a2()));}

    inline bool has_orthogonal_basis() const { return has_orthogonal_basis_; };
    inline jams::LatticeSystem lattice_system() const { return lattice_system_; };

    inline jams::Vec<bool, 3> periodic() const {return periodic_;}
    inline bool periodic(int n) const {return periodic_[n];}

    inline jams::Mat<double, 3, 3> matrix() const {return matrix_;}
    inline jams::Mat<double, 3, 3> inverse_matrix() const {return inverse_matrix_;}

    inline jams::Vec<double, 3> cartesian_to_fractional(jams::Vec<double, 3> xyz) const {return inverse_matrix_ * xyz;}
    inline jams::Vec<double, 3> fractional_to_cartesian(jams::Vec<double, 3> hkl) const {return matrix_ * hkl;}

    inline jams::Vec<double, 3> inv_cartesian_to_fractional(jams::Vec<double, 3> inv_xyz) const {return transpose(matrix_) * inv_xyz;}
    inline jams::Vec<double, 3> inv_fractional_to_cartesian(jams::Vec<double, 3> inv_hkl) const {return transpose(inverse_matrix_) * inv_hkl;}


protected:
    bool classify_orthogonal_basis() const;
    jams::LatticeSystem classify_lattice_system(const double& angle_eps = 1e-5) const;

    jams::Mat<double, 3, 3>  matrix_ = kIdentityMat3;
    jams::Mat<double, 3, 3>  inverse_matrix_= kIdentityMat3;
    jams::Vec<bool, 3> periodic_ = {{true, true, true}};

    bool has_orthogonal_basis_ = true;
    jams::LatticeSystem lattice_system_ = jams::LatticeSystem::cubic;
};

inline Cell::Cell(const jams::Mat<double, 3, 3> &basis, jams::Vec<bool, 3> pbc)
        : matrix_(basis),
          inverse_matrix_(inverse(matrix_)),
          periodic_(pbc),
          has_orthogonal_basis_(classify_orthogonal_basis()),
          lattice_system_(classify_lattice_system())
{}

inline Cell::Cell(const jams::Vec<double, 3> &a, const jams::Vec<double, 3> &b, const jams::Vec<double, 3> &c, jams::Vec<bool, 3> pbc)
        : matrix_(matrix_from_cols(a, b, c)),
          inverse_matrix_(inverse(matrix_)),
          periodic_(pbc),
          has_orthogonal_basis_(classify_orthogonal_basis()),
          lattice_system_(classify_lattice_system())
{}


#endif //JAMS_CELL_H
