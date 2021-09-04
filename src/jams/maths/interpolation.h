#ifndef INCLUDED_JAMS_MATHS_INTERPOLATION
#define INCLUDED_JAMS_MATHS_INTERPOLATION

namespace jams {
    namespace maths {
        ///
        /// Interpolate the value at x from samples at x1 and x2
        ///
        double linear_interpolation(double x, double x1, double f_x1, double x2,
                                    double f_x2);

        ///
        /// Interpolate the value at the point (x,y) from samples
        /// on the corners of the rectangle (x1,y1), (x1,y2), (x2, y1),
        /// (x2, y2).
        ///
        /// \verbatim
        ///    ^
        ///    |   f_12      R2   f_22
        /// y2 -   o- - - - -+- - o
        ///    |
        ///    |             |
        /// y  -             * f(x,y)
        ///    |             |
        /// y1 -   o- - - - -+- - o
        ///    |   f_11      R1   f_21
        ///    |
        ///    ----|---------|----|---->
        ///        x1        x    x2
        /// \endverbatim
        ///
        double bilinear_interpolation(double x, double y, double x1, double y1,
                                      double x2,
                                      double y2, double f_11, double f_12,
                                      double f_21,
                                      double f_22);
        }
}

#endif // INCLUDED_JAMS_MATHS_INTERPOLATION
