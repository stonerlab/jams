// init_skyrmion.h                                                    -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_SKYRMION
#define INCLUDED_JAMS_INIT_SKYRMION

/// @class jams::InitSkyrmion
///
/// Initialises a Skyrmion with a given radius, vorticity and helicity.
///
/// @details
/// @p This class is designed for initialising a Bloch domain wall. The domain
/// wall is placed in the x-direction and the spins are aligned im the -/+ z
/// directions at the left/right ends of the wall. The center and width can be
/// specified.
///
/// The general equation is
/// @f[
///     S_x(x) = 0; S_y(x) = 1/cosh(pi (x-center) / width); S_z = tanh(pi (x-center) / width)
/// @f]
/// This assumes the system is much wider than the domain wall. See
/// [Kazantseva, Phys. Rev. Lett. 94, 037206 (2005)](https://dx.doi.org/10.1103/physrevlett.94.037206)
/// for a derivation and more details of constrained domain walls.
///
/// @attention We are using the convention for width which includes a factor of
/// pi. i.e. = pi \sqrt{A/K}. This factor of pi is an arbitrary choice in the
/// definition of width. Some authors use it some don't. Including the pi gives
/// are much better feeling for the extent of the domain wall when trying to fit
/// it into a finite size simulation box.
///
/// @setting `initializer.width` (required) domain wall width in units of lattice constants
/// @setting `initializer.center` (required) domain wall width in units of lattice constants
///
/// @example Example config:
/// @code{.unparsed}
/// initializer : {
///     module = "domain_wall";
///     width = 10.0;
///     center = 64.0;
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
