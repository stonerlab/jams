//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_CELL_H
#define JAMS_CELL_H
#include "jams/core/types.h"

class Cell;

Vec3   minimum_image(const Cell& cell, const Vec3& r_i, const Vec3& r_j);
double volume(const Cell& cell);
Cell   scale(const Cell& cell, const Vec3i& size);
Cell   rotate(const Cell& cell, const Mat3& rotation_matrix);

class Cell {
public:
    inline Cell() = default;
    inline Cell(const Mat3 &basis, const Vec3b pbc = {true, true, true});
    inline Cell(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3b pbc = {true, true, true});

    inline Vec3 a() const;
    inline Vec3 b() const;
    inline Vec3 c() const;

    inline Vec3b periodic() const;
    inline bool periodic(int n) const;

    inline Mat3 matrix() const;
    inline Mat3 inverse_matrix() const;

protected:
    Mat3  matrix_ = kIdentityMat3;
    Mat3  inverse_matrix_= kIdentityMat3;
    Vec3b periodic_ = {true, true, true};
};

inline Cell::Cell(const Mat3 &basis, Vec3b pbc)
        : matrix_(basis),
          inverse_matrix_(inverse(matrix_)),
          periodic_(pbc)
{}

inline Cell::Cell(const Vec3 &a, const Vec3 &b, const Vec3 &c, Vec3b pbc)
        : matrix_(matrix_from_cols(a, b, c)),
          inverse_matrix_(inverse(matrix_)),
          periodic_(pbc)
{}

inline Vec3 Cell::a() const {
  return {matrix_[0][0], matrix_[1][0], matrix_[2][0]};
}

inline Vec3 Cell::b() const {
  return {matrix_[0][1], matrix_[1][1], matrix_[2][1]};
}

inline Vec3 Cell::c() const {
  return {matrix_[0][2], matrix_[1][2], matrix_[2][2]};
}

inline bool Cell::periodic(const int n) const {
  return periodic_[n];
}

inline Mat3 Cell::matrix() const {
  return matrix_;
}

inline Mat3 Cell::inverse_matrix() const {
  return inverse_matrix_;
}

Vec3b Cell::periodic() const {
  return periodic_;
}

#endif //JAMS_CELL_H
