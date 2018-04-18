// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_PHYSICS_H
#define JAMS_CORE_PHYSICS_H

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/base.h"

class Physics : public Base {
 public:
  Physics(const libconfig::Setting &settings);
  virtual ~Physics() {}

  virtual void update(const int &iterations, const double &time, const double &dt) = 0;

  inline double temperature() const { return temperature_; }
  inline void set_temperature(const double t) { temperature_ = t; }

  inline const Vec3& applied_field() const { return applied_field_; }
  inline double applied_field(const int i) const { return applied_field_[i]; }

  inline void set_applied_field(const Vec3 &field) { applied_field_ = field; }

  static Physics* create(const libconfig::Setting &settings);

 protected:
  double              temperature_;
  Vec3 applied_field_;
  int                 output_step_freq_;
};

#endif  // JAMS_CORE_PHYSICS_H
