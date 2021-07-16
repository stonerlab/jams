//
// Created by Joe Barker on 2017/11/15.
//

#include "jams/containers/cell.h"

__attribute__((hot))
Vec3 minimum_image_orthogonal_basis(const Cell& cell, const Vec3& r_cart_i, const Vec3& r_cart_j) {
  //   W. Smith, CCP5 Information Quarterly for Computer Simulation of Condensed Phases (1989).
  Vec3 dr = cell.inverse_matrix() * (r_cart_j - r_cart_i);

  for (auto n = 0; n < 3; ++n) {
    if (cell.periodic(n)) {
      dr[n] = dr[n] - std::trunc(2.0 * dr[n]);
    }
  }
  return cell.matrix() * dr;
}

__attribute__((hot))
Vec3 minimum_image_bruteforce(const Cell& cell, const Vec3& r_i_cart, const Vec3& r_j_cart) {
  const auto r_ij = r_j_cart - r_i_cart;

  const auto N_a = cell.periodic(0) ? 1 : 0;
  const auto N_b = cell.periodic(1) ? 1 : 0;
  const auto N_c = cell.periodic(2) ? 1 : 0;

  Vec3 r_ij_min = r_ij;
  for (auto h = -N_a; h < N_a + 1; ++h) {
    for (auto k = -N_b; k < N_b + 1; ++k) {
      for (auto l = -N_c; l < N_c + 1; ++l) {
        auto ds = r_ij + h * cell.a() + k * cell.b() + l * cell.c();
        if (norm_sq(ds) < norm_sq(r_ij_min)) {
          r_ij_min = ds;
        }
      }
    }
  }

  return r_ij_min;
}

Vec3 minimum_image(const Cell& cell, const Vec3& r_cart_i, const Vec3& r_cart_j) {
  if (cell.has_orthogonal_basis()) {
    return minimum_image_orthogonal_basis(cell, r_cart_i, r_cart_j);
  }

  return minimum_image_bruteforce(cell, r_cart_i, r_cart_j);
}

double volume(const Cell& cell) {
  return std::abs(determinant(cell.matrix()));
}

Cell scale(const Cell &cell, const Vec3i& size) {
  Vec3 new_a = cell.a() * double(size[0]);
  Vec3 new_b = cell.b() * double(size[1]);
  Vec3 new_c = cell.c() * double(size[2]);
  return Cell(new_a, new_b, new_c, cell.periodic());
}

Cell rotate(const Cell &cell, const Mat3& rotation_matrix) {
  Vec3 new_a = rotation_matrix * cell.a();
  Vec3 new_b = rotation_matrix * cell.b();
  Vec3 new_c = rotation_matrix * cell.c();

  return Cell(new_a, new_b, new_c, cell.periodic());
}

bool Cell::classify_orthogonal_basis() const {
  return approximately_zero(dot(a(), b()), DBL_EPSILON)
      && approximately_zero(dot(b(), c()), DBL_EPSILON)
      && approximately_zero(dot(c(), a()), DBL_EPSILON);
}

jams::LatticeSystem Cell::classify_lattice_system(const double& angle_eps) const {
  if (all_equal(a(), b(), c()) && all_equal(alpha(), beta(), gamma()) && approximately_equal(alpha(), 90.0, angle_eps)) {
    return jams::LatticeSystem::cubic;
  }

  if (all_equal(a(), b(), c()) && all_equal(alpha(), beta(), gamma()) && !approximately_equal(alpha(), 90.0, angle_eps)) {
    return jams::LatticeSystem::rhombohedral;
  }

  if (only_two_equal(a(), b(), c()) && all_equal(alpha(), beta(), gamma()) &&
      (approximately_equal(alpha(), 120.0, angle_eps) || approximately_equal(beta(), 120.0, angle_eps) || approximately_equal(gamma(), 120.0, angle_eps)))  {
    return jams::LatticeSystem::hexagonal;
  }

  if (only_two_equal(a(), b(), c()) && all_equal(alpha(), beta(), gamma()) && approximately_equal(alpha(), 90.0, angle_eps)) {
    return jams::LatticeSystem::tetragonal;
  }

  if (none_equal(a(), b(), c()) && all_equal(alpha(), beta(), gamma()) && approximately_equal(alpha(), 90.0, angle_eps)) {
    return jams::LatticeSystem::orthorhombic;
  }

  if ((a() != c() || c() != b() || c() != a()) &&
      (!approximately_equal(alpha(), 90.0, angle_eps) || !approximately_equal(beta(), 90.0, angle_eps) || !approximately_equal(gamma(), 90.0, angle_eps))) {
    return jams::LatticeSystem::monoclinic;
  }

  // any case not already matched
  return jams::LatticeSystem::triclinic;
}
