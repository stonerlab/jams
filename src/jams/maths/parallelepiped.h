#ifndef JAMS_PARALLELEPIPED_H
#define JAMS_PARALLELEPIPED_H

#include <array>
#include <jams/containers/vec3.h>

namespace jams {
    namespace maths {
        //
        // Returns the area of the parallelogram defined by the vectors a and b
        //
        inline constexpr double parallelogram_area(const Vec3 &a, const Vec3 &b) {
            return norm(cross(a, b));
        }

        //
        // Returns the height of the parallelogram defined by the vectors a
        // and b where b is taken as the base.
        //
        inline constexpr double parallelogram_height(const Vec3 &a, const Vec3 &b) {
            return parallelogram_area(a, b) / norm(b);
        }

        //
        // Returns the smallest inradius of the parallelogram defined by the
        // vectors a and b. The inradius is the radius of the largest circle
        // which fits within the parallelogram.
        //
        // NOTE: The inradius is only the same as for the inscribed circle if
        // the parallelogram is a rhombus (i.e. equal length edges). For a
        // general parallelogram the circle will not touch all of the edges.
        //
        inline double parallelogram_inradius(const Vec3 &a, const Vec3 &b) {
            return 0.5 * parallelogram_area(a, b) / std::max(norm(a), norm(b));
        }

        //
        // Returns the volume of the parallelepiped defined by the vectors
        // a, b and c.
        //
        inline double
        parallelepiped_volume(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
            return std::abs(dot(cross(a, b), c));
        }

        //
        // Returns the height of the parallelepiped which is the distance
        // between the base (a, b) and the vector c
        //
        inline double
        parallelepiped_height(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
            return parallelepiped_volume(a, b, c) / norm(cross(a, b));
        }

        //
        // Returns the smallest inradius of the parallelepiped defined by the
        // vectors a, b and c. The inradius is the radius of the largest sphere
        // which fits within the parallelepiped.
        //
        // NOTE: The inradius is only the same as for the inscribed sphere if
        // the parallelogram is a rhombohedron (i.e. equal length edges and
        // angles). For a general parallelepiped the circle will not touch all
        // of the edges.
        //
        inline double
        parallelepiped_inradius(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
            return 0.5 * parallelepiped_volume(a, b, c)
            / sqrt(std::max({cross_norm_sq(a, b), cross_norm_sq(b, c), cross_norm_sq(c, a)}));
        }

        //
        // Returns the longest body diagonal of the parallelepiped defined by
        // vectors a, b and c
        //
        inline double parallelepiped_longest_diagonal(const Vec3 &a, const Vec3 &b,
                                               const Vec3 &c) {
            return std::max({norm(a + b + c), norm(-a + b + c), norm(a - b + c),
                             norm(a + b - c)});
        }
    }
}

#endif //JAMS_PARALLELEPIPED_H
