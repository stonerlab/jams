// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_MONITOR_H
#define JAMS_CORE_MONITOR_H

#include <libconfig.h++>

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

  protected:
    bool is_equilibration_monitor_;
    int  output_step_freq_;

};

#endif  // JAMS_CORE_MONITOR_H
