// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_MONITOR_H
#define JAMS_CORE_MONITOR_H

#include <libconfig.h++>
#include <iosfwd>

// forward declarations
class Solver;

class Monitor {
 public:
  Monitor(const libconfig::Setting &settings);

  virtual ~Monitor() {}
  virtual void update(Solver * solver) = 0;
  virtual bool is_converged() = 0;
  bool is_updating (const int &iteration) const;

  static Monitor* create(const libconfig::Setting &settings);

  std::string name() const {return name_; }

  protected:
    std::string name_;

    int     output_step_freq_;

    // variables for convergence testing in monitor
    bool    convergence_is_on_;
    double  convergence_tolerance_;
    double  convergence_stderr_;
    double  convergence_burn_time_;  // amount of time to discard before calculating convegence stats

};

#endif  // JAMS_CORE_MONITOR_H
