#ifndef JAMS_PARALLELEPIPED_H
#define JAMS_PARALLELEPIPED_H

#include <array>
#include <jams/containers/vec3.h>

namespace jams {
namespace maths {

    template <typename T>
    inline void require_fp()
    {
        static_assert(std::is_floating_point<T>::value,
                      "This function requires a floating-point type");
    }

    ///
    /// Returns the area of the parallelogram defined by the vectors a and b
    ///
    template <typename T>
    inline T parallelogram_area(const Vec<T,3>& a, const Vec<T,3>& b)
    {
        require_fp<T>();
        return jams::norm(jams::cross(a, b));
    }

    ///
    /// Returns the height of the parallelogram defined by the vectors a
    /// and b where b is taken as the base.
    ///
    /// \verbatim
    ///
    ///      +-----------+..^
    ///     /           /   |
    ///  b /           /    | height
    ///   /           /     |
    ///  +-----------+......v
    ///        a
    /// \endverbatim
    template <typename T>
    inline T parallelogram_height(const Vec<T,3>& a, const Vec<T,3>& b)
    {
        require_fp<T>();
        return parallelogram_area(a, b) / jams::norm(b);
    }

    ///
    /// Returns the smallest inradius of the parallelogram defined by the
    /// vectors a and b. The inradius is the radius of the largest circle
    /// which fits within the parallelogram.
    ///
    /// NOTE: The inradius is only the same as for the inscribed circle if
    /// the parallelogram is a rhombus (i.e. equal length edges). For a
    /// general parallelogram the circle will not touch all of the edges.
    ///
    template <typename T>
    inline T parallelogram_inradius(const Vec<T,3>& a, const Vec<T,3>& b)
    {
        require_fp<T>();
        return T(0.5) * std::min(parallelogram_height(a, b),
                                 parallelogram_height(b, a));
    }

    ///
    /// Returns the volume of the parallelepiped defined by the vectors
    /// a, b and c.
    ///
    template <typename T>
    inline T parallelepiped_volume(const Vec<T,3>& a,
                                   const Vec<T,3>& b,
                                   const Vec<T,3>& c)
    {
        require_fp<T>();
        return std::abs(jams::dot(jams::cross(a, b), c));
    }

    ///
    /// Returns the height of the parallelepiped which is the distance
    /// between the base (a, b) and the vector c
    ///
    /// \verbatim
    ///
    ///        +------------+....\
    ///       / \         /  \    \
    ///      /    +-----------+....\
    ///     /    /c     /    /      ^
    ///    /    /      /    /       |
    ///   +----/------+..../....\   | height
    /// b  \  /         \ /      \  |
    ///      +-----------+........\ v
    ///            a
    /// \endverbatim
    template <typename T>
    inline T parallelepiped_height(const Vec<T,3>& a,
                                   const Vec<T,3>& b,
                                   const Vec<T,3>& c)
    {
        require_fp<T>();
        return parallelepiped_volume(a, b, c) / jams::norm(jams::cross(a, b));
    }


    ///
    /// Returns the smallest inradius of the parallelepiped defined by the
    /// vectors a, b and c. The inradius is the radius of the largest sphere
    /// which fits within the parallelepiped.
    ///
    /// NOTE: The inradius is only the same as for the inscribed sphere if
    /// the parallelogram is a rhombohedron (i.e. equal length edges and
    /// angles). For a general parallelepiped the circle will not touch all
    /// of the edges.
    ///
    template <typename T>
    inline T parallelepiped_inradius(const Vec<T,3>& a,
                                     const Vec<T,3>& b,
                                     const Vec<T,3>& c)
    {
        require_fp<T>();
        return T(0.5) * std::min({
            parallelepiped_height(a, b, c),
            parallelepiped_height(c, a, b),
            parallelepiped_height(b, c, a)
        });
    }


    ///
    /// Returns the longest body diagonal of the parallelepiped defined by
    /// vectors a, b and c
    ///
    template <typename T>
    inline T parallelepiped_longest_diagonal(const Vec<T,3>& a,
                                             const Vec<T,3>& b,
                                             const Vec<T,3>& c)
    {
        require_fp<T>();
        return std::max({
            jams::norm(a + b + c),
            jams::norm(-a + b + c),
            jams::norm(a - b + c),
            jams::norm(a + b - c)
        });
    }
}
}

#endif //JAMS_PARALLELEPIPED_H