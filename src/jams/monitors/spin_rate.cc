#include "jams/monitors/spin_rate.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/interface/config.h"

SpinRateMonitor::SpinRateMonitor(const libconfig::Setting& settings)
    : Monitor(settings) {
  if (!globals::solver->supports_deterministic_spin_rate()) {
    throw std::runtime_error("spin-rate is not supported by solver '" + globals::solver->name() + "'");
  }
  if (globals::num_magnetic_spins == 0) {
    throw std::runtime_error("spin-rate requires at least one magnetic spin");
  }
  if (output_step_freq_ <= 0 || start_delay_steps_ < 0) {
    throw std::runtime_error("spin-rate requires output_steps > 0 and start_delay_steps >= 0");
  }
  const int samples = jams::config_optional<int>(settings, "consecutive_samples", 1);
  if (samples <= 0) throw std::runtime_error("consecutive_samples must be positive");
  if (convergence_status_ != ConvergenceStatus::kDisabled) {
    threshold_.emplace(convergence_tolerance_, samples);
  }
  auto columns = globals::solver->monitor_coordinate_columns();
  columns.push_back({"max_ds_dt", "ps^-1"});
  tsv_.open(jams::output::monitor_filename(name(), "tsv"), std::move(columns),
            jams::config_optional<int>(settings, "precision", 8));
}

void SpinRateMonitor::update(Solver& solver) {
  const double rate = solver.max_deterministic_spin_rate();
  if (!jams::is_finite(rate) || rate < 0.0) {
    throw std::runtime_error("invalid deterministic spin rate");
  }
  auto values = solver.monitor_coordinates();
  values.push_back(rate);
  tsv_.write_row(values);

  if (!threshold_) return;
  const bool was_satisfied = threshold_->satisfied();
  if (solver.iteration() < solver.min_steps()) {
    threshold_->reset();
  } else {
    threshold_->update(rate);
  }
  convergence_status_ = threshold_->satisfied()
      ? ConvergenceStatus::kConverged : ConvergenceStatus::kNotConverged;
  if (!was_satisfied && threshold_->satisfied()) {
    std::ostringstream message;
    message << std::scientific << std::setprecision(8)
            << "spin-rate '" << name() << "' satisfied at iteration " << solver.iteration()
            << ": max_ds_dt = " << rate << " ps^-1 < " << convergence_tolerance_ << " ps^-1\n";
    std::cout << message.str();
  }
}
