#ifndef JAMS_PARALLELEPIPED_H
#define JAMS_PARALLELEPIPED_H

#include <array>

namespace jams {
    namespace maths {
        using Vec3 = std::array<double, 3>;

        //
        // Returns the area of the parallelogram defined by the vectors a and b
        //
        double parallelogram_area(const Vec3& a, const Vec3& b);

        //
        // Returns the height of the parallelogram defined by the vectors a
        // and b where b is taken as the base.
        //
        double parallelogram_height(const Vec3& a, const Vec3& b);

        //
        // Returns the smallest inradius of the parallelogram defined by the
        // vectors a and b. The inradius is the radius of the largest circle
        // which fits within the parallelogram.
        //
        // NOTE: The inradius is only the same as for the inscribed circle if
        // the parallelogram is a rhombus (i.e. equal length edges). For a
        // general parallelogram the circle will not touch all of the edges.
        //
        double parallelogram_inradius(const Vec3& a, const Vec3& b);

        //
        // Returns the volume of the parallelepiped defined by the vectors
        // a, b and c.
        //
        double parallelepiped_volume(const Vec3& a, const Vec3& b, const Vec3& c);

        //
        // Returns the height of the parallelepiped which is the distance
        // between the base (a, b) and the vector c
        //
        double parallelepiped_height(const Vec3& a, const Vec3& b, const Vec3& c);

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
        double parallelepiped_inradius(const Vec3& a, const Vec3& b, const Vec3& c);

        //
        // Returns the longest body diagonal of the parallelepiped defined by
        // vectors a, b and c
        //
        double parallelepiped_longest_diagonal(const Vec3& a, const Vec3& b, const Vec3& c);
    }
}

#endif //JAMS_PARALLELEPIPED_H
