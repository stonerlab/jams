// collective_variable.h                                               -*-C++-*-
#ifndef INCLUDED_JAMS_COLLECTIVE_VARIABLE
#define INCLUDED_JAMS_COLLECTIVE_VARIABLE

#include <jams/containers/vec3.h>

#include <string>

namespace jams {

/// Implements a collective variable which returns a value based on the current
/// state of the spin system.
class CollectiveVariable {
public:
    virtual ~CollectiveVariable() = default;

    /// Returns the name of the collective variable.
    virtual std::string name() = 0;

    /// Returns the current value of the collective variable.
    virtual double value() = 0;

    /// Returns the value of the collective variable after a trial
    /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
    virtual double spin_move_trial_value(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) = 0;

    /// Notify the CollectiveVariable that a trial spin move has been accepted.
    virtual void spin_move_accepted(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) = 0;
};

}


#endif //INCLUDED_JAMS_COLLECTIVE_VARIABLE
