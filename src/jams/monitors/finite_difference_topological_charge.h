//
// Created by ioannis charalampidis on 02/08/2022.
//

#ifndef JAMS_SRC_JAMS_MONITORS_FINITE_DIFFERENCE_TOPOLOGICAL_CHARGE_H_
#define JAMS_SRC_JAMS_MONITORS_FINITE_DIFFERENCE_TOPOLOGICAL_CHARGE_H_

#include <fstream>
#include <vector>
#include <libconfig.h++>
#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include <jams/helpers/montecarlo.h>

class TopChargeMonitor : public Monitor {
 public:
  TopChargeMonitor(const libconfig::Setting &settings);
  ~TopChargeMonitor();

  void update(Solver * solver) override;
  void post_process() override {};

  bool is_converged() override;

 private:
  static std::string tsv_header();

  double local_topological_charge(const int i) const;


  std::string name_ = "topological_charge_finite_diff";
  std::ofstream outfile;

  // basically a CSR matrix
  std::vector<std::vector<int>> dx_indices_;
  std::vector<std::vector<double>> dx_values_;

  // basically a CSR matrix
  std::vector<std::vector<int>> dy_indices_;
  std::vector<std::vector<double>> dy_values_;

  double monitor_top_charge_cache_ = 0.0;

  double max_tolerance_threshold_ = 1.0;
  double min_tolerance_threshold_ = -1.0;
  double tolerance_value_ = 0.09;
};

#endif //JAMS_SRC_JAMS_MONITORS_FINITE_DIFFERENCE_TOPOLOGICAL_CHARGE_H_



