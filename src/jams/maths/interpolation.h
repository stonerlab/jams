//
// Created by Work on 2020/11/12.
//

#ifndef JAMS_MATHS_INTERPOLATION_H
#define JAMS_MATHS_INTERPOLATION_H

#include <cassert>

namespace jams {
    namespace maths {
        /// 1D linear interpolation to find y at point x given lower coordinate
        /// (x_lower, y_lower) and upper coordinate (x_upper, y_upper)
        inline double linear_interpolation(const double &x,const double &x_lower, const double &y_lower, const double &x_upper, const double &y_upper) {
     //     assert(x_lower < x_upper); in 2d potential this is not always true
          assert(x > x_lower || approximately_equal(x, x_lower));
          assert(x < x_upper || approximately_equal(x, x_upper));

          return y_lower + (x - x_lower) * (y_upper - y_lower) / (x_upper - x_lower);
        }
    }
}

#endif //JAMS_INTERPOLATION_H
