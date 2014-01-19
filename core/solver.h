// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SOLVER_H
#define JAMS_CORE_SOLVER_H


#include "core/globals.h"

enum SolverType{ METROPOLISMC, HEUNLLG, CUDAHEUNLLMS, CUDAHEUNLLBP, SEMILLG,
  CUDASEMILLG, CUDAHEUNLLG, CUDASRK4LLG, FFTNOISE };


class Solver {
 public:
  Solver()
  time(0.0),
  : initialized(false),
  iteration(0),
  temperature(0),
  dt(0.0),
  t_step(0.0)
  {}

  virtual ~Solver() {}

  virtual void initialize(int argc, char **argv, double dt) = 0;
  virtual void run() = 0;
  virtual void calcEnergy(double &e1_s, double &e1_t, double &e2_s,
    double &e2_t, double &e4_s) = 0;
  virtual void syncOutput() = 0;

  inline int getIteration() { return iteration; }
  inline double getTime() { return iteration*t_step; }
  inline double getTemperature() { return temperature; }
  inline void setTemperature(double &t) { temperature = t; }

  static Solver* Create();
  static Solver* Create(SolverType type);
 protected:
  bool initialized;

  double time;  // current time

  int iteration;  // number of iterations
  double temperature;
  double dt;
  double t_step;
};

#endif  // JAMS_CORE_SOLVER_H
