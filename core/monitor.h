// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_MONITOR_H
#define JAMS_CORE_MONITOR_H

#include <libconfig.h++>

#include "jblib/containers/vec.h"

class Monitor {
 public:
  Monitor(const libconfig::Setting &settings);

  virtual ~Monitor() {}
  virtual void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) = 0;
  bool is_updating (const int &iteraction) const;

  static Monitor* create(const libconfig::Setting &settings);

  protected:
    bool is_equilibration_monitor_;
    int  output_step_freq_;

};

#endif  // JAMS_CORE_MONITOR_H
