#ifndef JAMS_LATTICE_MINIMUM_IMAGE_H
#define JAMS_LATTICE_MINIMUM_IMAGE_H

///
/// @file
/// These functions are used for calculating the minimum image of a vector in a
/// parallelepiped cell which has periodic boundaries.
///
/// The parallelepiped cell is defined by three vectors (a, b, c) and periodic
/// boundary conditions (pbc) may exist along each direction (a, b, c)
/// independently. For a source point r_i and target point r_j the minimum image
/// is the shortest vector r_ij which connects r_i to an equivalent r_j in any
/// adjacent cell across the periodic boundaries. These equivalent positions are
/// r_ij' = r_i - ((h * a + k * b + l * c) + r_j), where h, k, l are integers.
/// If a,b,c defines a cube then the solution is a simple mapping across the
/// periodic boundaries, but for skew cells the solution is more complex.
///
/// Determining the minimum image is typically needed for calculating distances
/// between interacting particles or working out where to translate moving
/// particles in periodic system.
///
/// A fast method is described in "W. Smith, CCP5 Information Quarterly for
/// Computer Simulation of Condensed Phases (1989)" and we refer to it here as
/// "Smith's method". This transforms the points r_i and r_j into 'lattice'
/// coordinates, removing the anisotropy of skew cells and allowing a simple
/// solution of the problem as though it were a cube. However the method only
/// guarantees the shortest r_ij is found if |r_ij| is less than the inradius of
/// the cell. If it is larger we must fall back to a bruteforce search.
///
/// Bruteforce searches are also not trivial. For very skew cells it may be that
/// closer equivalent r_ij vectors are not found in the immediately adjacent
/// offset cells (i.e. h, k, l = +/- 1) but can be at more distant offsets. To
/// guarantee the shortest r_ij is found it is required to search all offset
/// cells within |r_ij| distance of r_i.
///

#include <jams/containers/vec3.h>
#include <jams/containers/mat3.h>

namespace jams {

    /// General function for calculating the vector displacement r_ij = r_i -
    /// r_j obeying the minimum image convention within a cell defined by
    /// vectors @p a, @p b, @p c with optional periodic boundaries along each
    /// vector. This function is be safe in all cases, always returning the
    /// minimum distance.
    ///
    /// @param a   cell vector a
    /// @param b   cell vector b
    /// @param c   cell vector c
    /// @param pbc Vec3 of booleans for (a, b ,c), true if lattice direction is
    /// periodic
    /// @param r_i Vec3 point i
    /// @param r_j Vec3 point j
    ///
    /// @return r_ij minimum image displacement
    ///
    Vec3
    minimum_image(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3b &pbc,
                  const Vec3 &r_i, const Vec3 &r_j, const double& epsilon);

    /// Bruteforce method to calculate the vector displacement r_ij = r_i - r_j
    /// obeying the minimum image convention within a cell defined by vectors a,
    /// b, c with optional periodic boundaries along each vector.
    ///
    /// @param a   cell vector a
    /// @param b   cell vector b
    /// @param c   cell vector c
    /// @param pbc Vec3 of booleans for (a, b ,c), true if lattice direction is
    /// periodic
    /// @param r_i Vec3 point i
    /// @param r_j Vec3 point j
    ///
    /// @return r_ij minimum image displacement
    ///
    /// @details This algorithm does a bruteforce search over adjacent offset
    /// cells for the shortest distance between @p r_i and @p r_j. The depth of
    /// adjacent cells to check is determined by the initial r_ij within the
    /// cell. For very skew cells it is possible that a long distance must be
    /// checked. In the worst case the maximum distance in the cell is the
    /// longest diagonal and this must be compared to the height of the
    /// parallelepiped of the cell in each direction.
    ///
    Vec3 minimum_image_bruteforce(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                                  const Vec3b &pbc, const Vec3 &r_i,
                                  const Vec3 &r_j, const double& epsilon);

    /// Bruteforce method to calculate the vector displacement r_ij = r_i - r_j
    /// obeying the minimum image convention within a cell defined by vectors a,
    /// b, c with optional periodic boundaries along each vector. The depth of
    /// adjacent cell offsets to search is specified by 'offset_depth'.
    ///
    /// @param a   cell vector a
    /// @param b   cell vector b
    /// @param c   cell vector c
    /// @param pbc Vec3 of booleans for (a, b ,c), true if lattice direction is
    /// periodic
    /// @param r_i Vec3 point i
    /// @param r_j Vec3 point j
    /// @param offset_depth Vec3 of integers giving the search +/- offset depth
    /// along (a, b, c)
    ///
    /// @return r_ij minimum image displacement
    ///
    /// @details This algorithm does a bruteforce search over adjacent offset
    /// cells for the shortest distance between r_i and r_j it is essentially
    /// the same as the 'minimum_image_bruteforce' function but the offset_depth
    /// is manually specified.
    ///
    Vec3 minimum_image_bruteforce_explicit_depth(const Vec3 &a, const Vec3 &b,
                                                 const Vec3 &c,
                                                 const Vec3b &pbc,
                                                 const Vec3 &r_i,
                                                 const Vec3 &r_j,
                                                 const Vec3i &offset_depth,
                                                 const double& epsilon);

    /// Calculates the vector displacement r_ij = r_i - r_j obeying the minimum
    /// image convention using Smith's algorithm within a cell defined by
    /// vectors a, b, c with optional periodic boundaries along each vector.
    ///
    /// @warning The method guarantees to give the minimum image only if the
    /// displacement found (r_ij) is less than the inradius of the cell (the
    /// radius of the smallest sphere which can be completely contained in the
    /// parallelepiped a,b,c). If this is not satisfied then an alternative
    /// method must be used. This function does not check the inradius
    /// condition.
    ///
    /// @param a   cell vector a
    /// @param b   cell vector b
    /// @param c   cell vector c
    /// @param pbc Vec3 of booleans for (a, b,c), true if lattice direction is
    /// periodic
    /// @param r_i Vec3 point i
    /// @param r_j Vec3 point j
    ///
    /// @return r_ij minimum image displacement
    ///
    /// @details This is a specialised version of the minimum image using the
    /// method described in "W. Smith, CCP5 Information Quarterly for Computer
    /// Simulation of Condensed Phases (1989)" whereby the minimum image is
    /// solved in the ' lattice space' rather than real space. It should be much
    /// more efficient than bruteforce searches of image cells and can also
    /// handle non-orthogonal cells with no extra performance cost. The method
    /// guarantees to give the minimum image only if the radius of the
    /// displacement found (|r_ij|) is less than the inradius of the cell (the
    /// radius of the smallest sphere which can be completely contained in the
    /// cell). This function does NOT check this condition and so it MUST be
    /// already guaranteed that any r_ij greater than the inradius will be
    /// rejected. See the example below
    ///
    /// @example
    /// In this example we use @c minimum_image_smith_method_no_radius_check to
    /// find the minimum image vector with the understanding that we will be
    /// rejecting (continuing the loop) for any |r_ij| which is larger than @c
    /// r_cutoff which we have already guaranteed is smaller than the inradius
    /// of our cell.
    /// @code
    ///   for (auto j = 0; j < N; ++j) {
    ///     auto r_i = position[i];
    ///     for (auto j = 0; j < N; ++j) {
    ///         auto r_j = position[j];
    ///
    ///         auto r_ij = minimum_image_smith_method(a, b, c, pbc, r_i, r_j);
    ///
    ///         // here we guarantee r_cutoff is less than the inradus of the
    ///         // cell (a,b,c)
    ///         if (definately_greater_than(norm(r_ij), r_cutoff)) continue;
    ///
    ///         ... do some calculation ...
    ///     }
    ///   }
    /// @endcode
    ///
    Vec3 minimum_image_smith_method(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                                    const Vec3b &pbc, const Vec3 &r_i,
                                    const Vec3 &r_j);

    /// Calculates the vector displacement r_ij = r_i - r_j obeying the minimum
    /// image convention using Smith's algorithm within a cell defined by
    /// vectors a, b, c with optional periodic boundaries along each vector.
    ///
    /// @warning The method guarantees to give the minimum image only if the
    /// displacement found (r_ij) is less than the inradius of the cell (the
    /// radius of the smallest sphere which can be completely contained in the
    /// parallelepiped a,b,c). If this is not satisfied then an alternative
    /// method must be used. This function does not check the inradius
    /// condition.
    ///
    /// @param cell_matrix Mat3 where columns are (a, b, c) vectors defining the
    /// unit cell
    /// @param cell_inv_matrix Mat3 inverse of cell_matrix
    /// @param r_i Vec3 point i
    /// @param r_j Vec3 point j
    ///
    /// @return r_ij minimum image displacement
    Vec3 minimum_image_smith_method(const Mat3 &cell_matrix,
                                    const Mat3 &cell_inv_matrix,
                                    const Vec3b &pbc,
                                    const Vec3 &r_i, const Vec3 &r_j);

}

#endif //JAMS_LATTICE_MINIMUM_IMAGE_H
