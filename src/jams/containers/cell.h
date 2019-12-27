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

Vec3 minimum_image(const Cell& cell, const Vec3& r_cart_i, const Vec3& r_cart_j);
double volume(const Cell& cell);
Cell   scale(const Cell& cell, const Vec3i& size);
Cell   rotate(const Cell& cell, const Mat3& rotation_matrix);

class Cell {
public:
    inline Cell() = default;
    inline Cell(const Mat3 &basis, const Vec3b pbc = {{true, true, true}});
    inline Cell(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3b pbc = {{true, true, true}});

    inline Vec3 a() const { return {matrix_[0][0], matrix_[1][0], matrix_[2][0]}; }
    inline Vec3 b() const { return {matrix_[0][1], matrix_[1][1], matrix_[2][1]}; }
    inline Vec3 c() const { return {matrix_[0][2], matrix_[1][2], matrix_[2][2]}; }

    inline double alpha() const {return rad_to_deg(angle(b(), c()));}
    inline double beta() const {return rad_to_deg(angle(c(), a()));}
    inline double gamma() const {return rad_to_deg(angle(a(), b()));}

    inline bool has_orthogonal_basis() const { return has_orthogonal_basis_; };
    inline jams::LatticeSystem lattice_system() const { return lattice_system_; };

    inline Vec3b periodic() const {return periodic_;}
    inline bool periodic(int n) const {return periodic_[n];}

    inline Mat3 matrix() const {return matrix_;}
    inline Mat3 inverse_matrix() const {return inverse_matrix_;}

protected:
    bool classify_orthogonal_basis() const;
    jams::LatticeSystem classify_lattice_system(const double& angle_eps = 1e-5) const;

    Mat3  matrix_ = kIdentityMat3;
    Mat3  inverse_matrix_= kIdentityMat3;
    Vec3b periodic_ = {{true, true, true}};

    bool has_orthogonal_basis_ = true;
    jams::LatticeSystem lattice_system_ = jams::LatticeSystem::cubic;
};

inline Cell::Cell(const Mat3 &basis, Vec3b pbc)
        : matrix_(basis),
          inverse_matrix_(inverse(matrix_)),
          periodic_(pbc),
          has_orthogonal_basis_(classify_orthogonal_basis()),
          lattice_system_(classify_lattice_system())
{}

inline Cell::Cell(const Vec3 &a, const Vec3 &b, const Vec3 &c, Vec3b pbc)
        : matrix_(matrix_from_cols(a, b, c)),
          inverse_matrix_(inverse(matrix_)),
          periodic_(pbc),
          has_orthogonal_basis_(classify_orthogonal_basis()),
          lattice_system_(classify_lattice_system())
{}


#endif //JAMS_CELL_H
