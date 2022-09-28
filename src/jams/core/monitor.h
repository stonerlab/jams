// monitor.h                                                           -*-C++-*-

#ifndef JAMS_CORE_MONITOR_H
#define JAMS_CORE_MONITOR_H

///
/// @purpose
///     Defines the interface for monitor classes to implement
///
/// @classes
///   Monitor: virtual class for monitors which are observers of the simulation
///
/// @description
///     This class defines the interface of a monitor which is a
///     passive observer of the simulation. Monitors analyse the simulated system
///     and are typically told to update at intervals by the solver.
///
///     Monitors should never alter the simulated system. This is currently not
///     strictly enforced (do to the overuse of global variables), but this may
///     change in the future.
///
/// Usage
/// -----
///
/// @example
/// @code{.cpp}
///
///     class MyMonitor : public Monitor {
///     public:
///         explicit MyMonitor(const libconfig::Setting &settings);
///
///         ~MyMonitor() override = default;
///
///         void update(Solver *solver) override;
///         void post_process() override {};
///
///         bool convergence_status() override;
///     };
///
/// @endcode

#include <libconfig.h++>

#include "jams/core/base.h"

class Solver;

//==============================================================================
// class Monitor
//==============================================================================

class Monitor : public Base {
public:

    enum class ConvergenceStatus {
        kDisabled,
        kConverged,
        kNotConverged,
    };

    ///
    /// Construct the monitor using any config values provided in `settings`.
    ///
    explicit Monitor(const libconfig::Setting &settings);

    virtual ~Monitor() = default;

    ///
    /// Request the monitor to update.
    ///
    /// This will usually be called by a solver to notify the monitor that an
    /// iteration is complete and the monitor can do some calculations based
    /// on the current state of the system if needed.
    ///
    virtual void update(Solver *solver) = 0;

    ///
    /// Runs any post processing.
    ///
    /// This is called when no more updates are expected, usually when the
    /// solving is completed. Typical us is when a monitor is used to gather a
    /// series of data and then the `post_process` performs some calculations
    /// on the series and outputs the final result.
    ///
    virtual void post_process() = 0;

    virtual ConvergenceStatus convergence_status() { return convergence_status_; };

    bool is_updating(const int &iteration) const;

    ///
    /// Factory which creates different Monitors which are derived from
    /// this class. The `settings` will be used to determined which class is
    /// created and the `settings` will be forwarded to the constructor of that
    /// class.
    ///
    static Monitor *create(const libconfig::Setting &settings);

protected:
    int output_step_freq_;
    ConvergenceStatus convergence_status_;
    double convergence_tolerance_;
    double convergence_stderr_;
    double convergence_burn_time_;
};

#endif  // JAMS_CORE_MONITOR_H
// ----------------------------- END-OF-FILE ----------------------------------