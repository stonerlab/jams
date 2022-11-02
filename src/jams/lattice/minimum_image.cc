#include <jams/lattice/minimum_image.h>

#include <jams/maths/parallelepiped.h>
#include <jams/containers/mat3.h>

#include <cassert>
#include <tuple>

namespace {
    /// Returns true if all periodic boundaries are false
    inline bool is_open_system(const Vec3b &pbc) {
      return !pbc[0] && !pbc[1] && !pbc[2];
    }
};

Vec3 jams::minimum_image(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                         const Vec3b &pbc, const Vec3 &r_i, const Vec3 &r_j, const double& epsilon) {
  // If the vectors a, b, c lie in a plane then the minimum image will
  // probably not work
  assert(!approximately_zero(jams::maths::parallelepiped_volume(a, b, c), epsilon));

  if (is_open_system(pbc)) {
    // if there are no periodic boundaries then return the only solution
    return r_i - r_j;
  }

  Vec3 r_ij = minimum_image_smith_method(a, b, c, pbc, r_i, r_j);

//  if ((dot(a, b) == 0 && dot(b, c) == 0 && dot(c, a) == 0) ||
//      definately_less_than(norm(r_ij), maths::parallelepiped_inradius(a, b, c), epsilon)) {
//    return r_ij;
//  }

  // For Smith's method we accept only if |r_ij| < r_inradius. For orthogonal
  // systems you can in principle always use Smith's method (the inradius check
  // guards against skewness) but degenerate points of the unit cell
  // are mapped into neighbouring cells i.e. the minimum image for
  // r_i = (0, 0, 0) r_j = (0.5, 0.5, 0.5) is r_ij = (-0.5, -0.5, -0.5). In
  // principle this makes no difference, but we generally want the r_ij found
  // with Smith's method to be the same as the r_ij found with a brute force
  // method. In the brute force method, which of the degenerate positions
  // we identify will depend on the order in which we check them (i.e. if we
  // check one distance and then only accept shorter distances then the first
  // cell we hit with the minimum distance will be the r_ij). So the simplest
  // behaviour is to check the central cell first and then loop over neighbours.
  // For Smith's method we can ensure degenerate points are found in the central
  // cell only by applying the cutoff radius to all cells such that generate
  // points always fall back to the brute force algorithm--even though cubic
  // cells look simpler. It may be possible to solve the issue in other ways
  // with a more complex algorithm.
  if (definately_less_than(norm(r_ij), maths::parallelepiped_inradius(a, b, c), epsilon)) {
    return r_ij;
  }

  // If r_ij is not inside the inradius we have to do a bruteforce check
  // algorithm which should always give the shortest r_ij but it is much
  // more costly.
  return minimum_image_bruteforce(a, b, c, pbc, r_i, r_j, epsilon);
}


Vec3 jams::minimum_image_bruteforce_explicit_depth(const Vec3 &a, const Vec3 &b,
                                                   const Vec3 &c,
                                                   const Vec3b &pbc,
                                                   const Vec3 &r_i,
                                                   const Vec3 &r_j,
                                                   const Vec3i &offset_depth,
                                                   const double& epsilon) {
  // If the cell is not periodic along a vector (a, b or c) then set the
  // offset_depth in that direction to zero
  const Vec3i N{
      pbc[0] ? offset_depth[0] : 0,
      pbc[1] ? offset_depth[1] : 0,
      pbc[2] ? offset_depth[2] : 0};

  // calculate the displacement between r_i and r_j in the central cell
  Vec3 r_ij = r_i - r_j;
  // search over repeated offset cells to look for a smaller displacement than
  // the one currently found

  for (auto h = -N[0]; h < N[0] + 1; ++h) {
    for (auto k = -N[1]; k < N[1] + 1; ++k) {
      for (auto l = -N[2]; l < N[2] + 1; ++l) {
        // calculate the displacement between r_i and and r_j in the
        // offset cell
        auto r_ik = r_i - ((h * a + k * b + l * c) + r_j);

        if (definately_less_than(norm_squared(r_ik), norm_squared(r_ij), epsilon)) {
          r_ij = r_ik;
        }
      }
    }
  }

  return r_ij;
}

Vec3 jams::minimum_image_bruteforce(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                                    const Vec3b &pbc, const Vec3 &r_i,
                                    const Vec3 &r_j, const double& epsilon) {
  // calculate the displacement between r_i and r_j
  Vec3 r_ij = r_i - r_j;

  // if there are no periodic boundaries then return the only solution
  if (is_open_system(pbc)) {
    return r_ij;
  }

  // Although the maximum possible distance is the longest diagonal, we only
  // need to search cells which are within |r_ij| because we only care about
  // shorter distances.
  const auto r_max = norm(r_ij);

  // if the cell is periodic along a vector (a, b or c) then set the number of
  // offset repeats in that direction to search over
  int N_a = ceil(r_max / jams::maths::parallelepiped_height(b, c, a));
  int N_b = ceil(r_max / jams::maths::parallelepiped_height(c, a, b));
  int N_c = ceil(r_max / jams::maths::parallelepiped_height(a, b, c));

  return minimum_image_bruteforce_explicit_depth(a, b, c, pbc, r_i, r_j,
                                                 {N_a, N_b, N_c}, epsilon);
}

Vec3 jams::minimum_image_smith_method(const Mat3 &cell_matrix,
                                      const Mat3 &cell_inv_matrix,
                                      const Vec3b &pbc,
                                      const Vec3 &r_i, const Vec3 &r_j) {
  Vec3 r_ij = r_i - r_j;

  // if there are no periodic boundaries then return the only solution
  if (is_open_system(pbc)) {
    return r_ij;
  }

  // transform the real space r_ij into fractional lattice coordinates
  Vec3 s_ij = cell_inv_matrix * r_ij;

  // In Smith's paper he uses the function INT(A). Presumably this is the
  // fortran function INT which in gcc is describes as:
  //
  // "If A is of type REAL and |A| < 1, INT(A) equals 0. If |A| \geq 1, then
  // INT(A) is the integer whose magnitude is the largest integer that does not
  // exceed the magnitude of A and whose sign is the same as the sign of A."
  //
  //
  // This is NOT the same as casting to an int in C++ (i.e. int(A)) which
  // simply removes the decimal part. The correct equivalent function is
  // std::trunc.
  //
  for (auto n = 0; n < 3; ++n) {
    if (pbc[n]) {
      s_ij[n] = s_ij[n] - std::trunc(2.0 * s_ij[n]);
    }
  }

  // transform back into real space
  return cell_matrix * s_ij;
}

Vec3 jams::minimum_image_smith_method(const Vec3 &a, const Vec3 &b,
                                      const Vec3 &c,
                                      const Vec3b &pbc,
                                      const Vec3 &r_i,
                                      const Vec3 &r_j) {
  auto T = matrix_from_cols(a, b, c);
  return minimum_image_smith_method(T, inverse(T), pbc, r_i, r_j);
}
