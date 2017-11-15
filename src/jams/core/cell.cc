//
// Created by Joe Barker on 2017/11/15.
//

#include "jams/core/cell.h"

Vec3 minimum_image(const Cell& cell, const Vec3& r_i, const Vec3& r_j) {
  // W. Smith, CCP5 Information Quarterly for Computer Simulation of Condensed Phases (1989).
  Vec3 dr = cell.inverse_matrix() * (r_j - r_i);
  for (auto n = 0; n < 3; ++n) {
    if (cell.periodic(n)) {
      dr[n] = dr[n] - trunc(2.0 * dr[n]);
    }
  }
  return cell.matrix() * dr;
}

double volume(const Cell& cell) {
  return std::abs(determinant(cell.matrix()));
}

Cell scale(const Cell &cell, const Vec3i& size) {
  Vec3 new_a = cell.a() * double(size[0]);
  Vec3 new_b = cell.b() * double(size[1]);
  Vec3 new_c = cell.c() * double(size[2]);
  return Cell(cell.a(), cell.b(), cell.c(), cell.periodic());
}

Cell rotate(const Cell &cell, const Mat3& rotation_matrix) {
  Vec3 new_a = rotation_matrix * cell.a();
  Vec3 new_b = rotation_matrix * cell.b();
  Vec3 new_c = rotation_matrix * cell.c();

  return Cell(new_a, new_b, new_c, cell.periodic());
}