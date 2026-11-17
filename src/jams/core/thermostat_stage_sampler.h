#ifndef JAMS_CORE_THERMOSTAT_STAGE_SAMPLER_H
#define JAMS_CORE_THERMOSTAT_STAGE_SAMPLER_H

#include <cstdint>
#include <stdexcept>

#include "jams/core/thermostat.h"

namespace jams {

// A solver supplies its tableau's stage ticks. The thermostat publishes one
// field per tick on its native grid, including ticks the tableau skips.
class ThermostatStageSampler {
 public:
  explicit ThermostatStageSampler(int ticks_per_step) : ticks_per_step_(ticks_per_step) {}

  void reset() { current_tick_ = -1; }

  const Real* sample(Thermostat& bath, Real temperature, int step, int stage_tick) {
    advance_to(bath, temperature, target_tick(step, stage_tick));
    return bath.data();
  }

#if HAS_CUDA
  // The caller records consumption on the solver stream after the last
  // kernel using each distinct sample.
  const Real* sample_device(Thermostat& bath, Real temperature, int step,
                            int stage_tick, cudaStream_t stream) {
    const auto target = target_tick(step, stage_tick);
    if (target > current_tick_) {
      bath.wait_for_consumed();
      advance_to(bath, temperature, target);
      bath.record_done();
    }
    bath.wait_on(stream);
    return bath.device_data();
  }
#endif

 private:
  std::int64_t target_tick(int step, int stage_tick) const {
    const auto target = std::int64_t{step} * ticks_per_step_ + stage_tick;
    if (step < 0 || stage_tick < 0 || stage_tick > ticks_per_step_ || target < current_tick_) {
      throw std::logic_error("invalid thermostat stage tick");
    }
    return target;
  }

  void advance_to(Thermostat& bath, Real temperature, std::int64_t target) {
    if (target == current_tick_) return;
    if (bath.has_uniform_temperature()) {
      bath.set_temperature(temperature);
    }
    if (current_tick_ < 0) {
      bath.sample_initial();
      current_tick_ = 0;
    }
    while (current_tick_ < target) {
      bath.update();
      ++current_tick_;
    }
  }

  const int ticks_per_step_;
  std::int64_t current_tick_ = -1;
};

}  // namespace jams

#endif  // JAMS_CORE_THERMOSTAT_STAGE_SAMPLER_H
