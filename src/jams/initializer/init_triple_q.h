// init_triple_q.h                                                     -*-C++-*-
#ifndef INCLUDED_JAMS_INIT_TRIPLE_Q
#define INCLUDED_JAMS_INIT_TRIPLE_Q

/// @class jams::InitTripleQ
///
/// Initialises the spin system with the specified triple Q configuration.
///
/// @details
/// @p This class allows complex alternating spin textures such as
/// antiferromagnetic states to be initialised even when all atoms have the
/// same material name.
///
/// The spins are initialised using
/// @f[
///     \vec{S} = \vec{S}'\exp{\left(2\pi i (h\vec{K}_1 + k\vec{K}_2 + l\vec{K}_3)\cdot\vec{r} \right)}
/// @f]
/// where \vec{S}' is a reference spin direction, K_1, K_2, K_3 are reciprocal
/// lattice vectors, h,k,l are decimal multiplies and r is the position of the
/// spin. h,k,l can be chosen such that the spins alternate between \vec{S}' and
/// -\vec{S}' along different lattice directions. They must be chosen carefully
/// though such that the phase is purely real on each site. The class will check
/// for this and raise an exception if the choice of h,k,l leads to imaginary
/// phases.
///
/// @setting `initializer.h` (required) K_1 multiplier
/// @setting `initializer.k` (required) K_2 multiplier
/// @setting `initializer.l` (required) K_3 multiplier
/// @setting `initializer.spin` (required) reference spin direction
/// @setting `initializer.material` (optional) limit the initializer to a single material
///
/// @example Example config:
/// @code{.unparsed}
/// initializer : {
///     module = "triple-q";
///     h = 0.5;
///     k = 0.5;
///     l = 0.5;
///     spin = [0, 0, 1];
///     material = "Fe";
/// };
/// @endcode

#include <jams/interface/config.h>

namespace jams {
class InitTripleQ {
public:
    static void execute(const libconfig::Setting &settings);
};
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------