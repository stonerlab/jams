// init_skyrmion.h                                                    -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_SKYRMION
#define INCLUDED_JAMS_INIT_SKYRMION

/// @class jams::InitSkyrmion
///
/// Initialises a Skyrmion with a given charge, vorticity, helicity and
/// 'c' and 'w' parameters which control width and domain wall sharpness.
///
/// @details
/// @p This class is designed for initialising a Neel skyrmion with given
/// properties and center. We follow the definitions given here:
/// https://juspin.de/skyrmion-radius/. Examples of the charge, vorticity,
/// helicity are given here: https://iopscience.iop.org/article/10.1088/1361-648X/ab5488.
///
/// The coordinates r are related to polar coordinates
/// @f[
///     r = (r \cos(\phi), r \sin(\phi), 0)
/// @f]
///
/// The spin orientation will be determined from spherical coordinates
/// @f[
///     m = (\sin(\theta) \cos(\phi), \sin(\theta) \cos(\phi), \cos(\theta))
/// @f]
///
/// The angles theta are given by
/// @f[
///     \theta(r) = \sum_{\pm} \arcsin( \tanh(-\frac{r \pm c}{w / 2}) ) + \pi
/// @f]
/// Where 'c' and 'w' determine the sharpness of the domain walls and the radius
/// of the skyrmion.
///
/// @note The equations only allow the creation of Neel type skyrmions.
/// this could be changed in future (see https://juspin.de/skyrmion-radius/).
///
/// @setting `initializer.coordinate_format` coordinate format ("fractional" or
///          "cartesian") to use for the skyrmion centre (default "fractional")
///
/// @setting `initializer.center` position of the skyrmion center as a 2D vector
///          in the x-y plane. If using fractional coordinates then these are
///          relative to the super cell i.e. [0.5, 0.5] is the center of the
///          whole system at z = 0. (default [0.5, 0.5] in fractional coordinates)
///
/// @setting `initializer.polarity` skyrmion core polarity (+1 or -1), where -1
///           means a ferromagnet in the +z direction with a skyrmion core in
///           the -z direction (default -1)
///
/// @setting `initializer.vorticity` skrymion vorticity (default 1.0)
///
/// @setting `initializer.helicity` skyrmion helicity (default 0.0)
///
/// @setting `initializer.w` skyrmion domain wall parameter 'w' (default 5.0)
///
/// @setting `initializer.c` skyrmion domain wall parameter 'c' (default 5.0)
///
/// @example Example config:
/// @code{.unparsed}
/// initializer : {
///     module = "skyrmion";
///     coordinate_format = "fractional";
///     center = [0.5, 0.5];
///     polarity = -1.0;
///     vorticity = 1.0;
///     helicity = 0.0;
///     w = 5.0;
///     c = 5.0;
/// };
/// @endcode


#include <jams/interface/config.h>

namespace jams {
class InitSkyrmion {
public:
  static void execute(const libconfig::Setting &settings);
};
}

#endif //INCLUDED_JAMS_INIT_SKYRMION

// ----------------------------- END-OF-FILE ----------------------------------
