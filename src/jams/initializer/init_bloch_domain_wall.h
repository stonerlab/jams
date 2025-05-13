// init_bloch_domain_wall.h                                            -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_BLOCH_DOMAIN_WALL
#define INCLUDED_JAMS_INIT_BLOCH_DOMAIN_WALL

/// @class jams::InitBlochDomainWall
///
/// Initialises a Bloch domain wall in the spin system with a given center and
/// width.
///
/// @details
/// @p This class is designed for initialising a Bloch domain wall. The domain
/// wall is placed in the x-direction and the spins are aligned im the -/+ z
/// directions at the left/right ends of the wall. The center and width can be
/// specified.
///
/// The general equation is
/// @f[
///     m_x(x) = 0; m_y(x) = 1/cosh(pi (x-center) / width); m_z = tanh(pi (x-center) / width)
/// @f]
/// This assumes the system is much wider than the domain wall. See
/// [Kazantseva, Phys. Rev. Lett. 94, 037206 (2005)](https://dx.doi.org/10.1103/physrevlett.94.037206)
/// for a derivation and more details of constrained domain walls.
///
/// The spin state is set by rotating the spins with respect the mx, my, mz of the domain wall. This means,
/// for example, if there is a ferrimagnetic ground state already set, the ground state is rotated, rather than
/// converting the system to a ferromagnetic state. Additional settings `normal` and `domain` allow the
/// direction and orientation of the domain wall to be chosen.
///
/// @attention We are using the convention for width which includes a factor of
/// pi. i.e. = pi \sqrt{A/K}. This factor of pi is an arbitrary choice in the
/// definition of width. Some authors use it some don't. Including the pi gives
/// are much better feeling for the extent of the domain wall when trying to fit
/// it into a finite size simulation box.
///
/// @setting `initializer.width` (required) domain wall width in units of lattice constants
/// @setting `initializer.center` (required) domain wall width in units of lattice constants
/// @setting `initializer.normal` (optional) propagation direction normal to the domain wall in cartesian space,
///                               default is x-axis (1, 0, 0)
/// @setting `initializer.domain` (optional) domain direction in cartesian space, the domain wall will be between
///                               +domain and -domain, default is z-axis (0, 0, 1)
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
class InitBlochDomainWall {
public:
  static void execute(const libconfig::Setting &settings);
};
}

#endif //INCLUDED_JAMS_INIT_BLOCH_DOMAIN_WALL

// ----------------------------- END-OF-FILE ----------------------------------
