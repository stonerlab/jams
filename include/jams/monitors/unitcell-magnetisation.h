// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_UNITCELL_MAGNETISATION_H
#define JAMS_MONITOR_UNITCELL_MAGNETISATION_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "H5Cpp.h"

#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jams/core/stats.h"

#include "jblib/containers/array.h"

class Solver;

class UnitcellMagnetisationMonitor : public Monitor {
 public:
  UnitcellMagnetisationMonitor(const libconfig::Setting &settings);
  ~UnitcellMagnetisationMonitor();

  void update(Solver * solver);
  bool is_converged() {return true;};
  std::string name() const {return "unitcell-magnetisation";}


 private:
  void write_h5_file(const std::string &h5_file_name, const H5::PredType float_type);
  void new_xdmf_file(const std::string &xdmf_file_name);
  void update_xdmf_file(const std::string &h5_file_name, const H5::PredType float_type);

  jblib::Array<double, 4> mag;
  FILE*        xdmf_file_;
};

#endif  // JAMS_MONITOR_UNITCELL_MAGNETISATION_H

