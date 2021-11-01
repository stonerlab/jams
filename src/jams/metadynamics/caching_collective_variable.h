// caching_collective_variable.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CACHING_COLLECTIVE_VARIABLE
#define INCLUDED_JAMS_CACHING_COLLECTIVE_VARIABLE

#include <jams/metadynamics/collective_variable.h>

#include <limits>

namespace jams {

// Implements a CollectiveVariable with caching internals. This can massively
// speed up Monte Carlo based metadynamics with computationally expensive
// CollectiveVariables.
//
// Deriving from this allows us to store the value of a CollectiveVariable in a
// cache when doing trial moves for Monte Carlo. Then on spin move acceptance we
// replace the cache value with the trial value. The expensive full calculation
// of the CollectiveVariable should be a function which overrides
// `calculate_expensive_value()`.
//
// Example implementation as a concrete CollectiveVariable
// -------------------------------------------------------
//
// Deriving classes would implement `value()` like:
//
//  double jams::CVarConcrete::value() {
//    return cached_value();
//  }
//
// The expensive calculation should be implemented in
// `calculate_expensive_value()`
//
//   double jams::CVarConcrete::calculate_expensive_value() {
//     double value = 0.0;
//     // DO SOMETHING EXPENSIVE HERE
//     return value;
//   }
//
// New cache variables must be set in `spin_move_trial_value()`
//
//   double jams::CVarConcrete::spin_move_trial_value(
//     int i, const Vec3 &spin_initial, const Vec3 &spin_trial) {
//
//     const double trial_value = cached_value() + // CHANGE IN VALUE;
//
//     // set new cache value for trial_value
//     set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);
//
//     return trial_value;
//   }
//

class CachingCollectiveVariable : public CollectiveVariable {
private:
    bool cache_initialised_ = false;
    double cached_value_ = std::numeric_limits<double>::signaling_NaN();
    double cached_trial_value_ = std::numeric_limits<double>::signaling_NaN();
    int cached_i_ = -1;
    Vec3 cached_spin_initial_ ;
    Vec3 cached_spin_trial_;

protected:
    inline bool cache_is_initialised() const;

    inline bool can_use_cache_values(int i, const Vec3 &spin_initial,
                                     const Vec3 &spin_trial) const;

    inline double cached_value();

    inline void
    set_cache_values(int i, const Vec3 &spin_initial, const Vec3 &spin_trial,
                     const double value, const double trial_value);

public:
    inline double value() override;

    virtual double calculate_expensive_value() = 0;

    inline void spin_move_accepted(int i, const Vec3 &spin_initial,
                            const Vec3 &spin_trial) override;


};

}

// INLINE FUNCTIONS

bool jams::CachingCollectiveVariable::cache_is_initialised() const {
  return cache_initialised_;
}

bool jams::CachingCollectiveVariable::can_use_cache_values(int i,
                                                           const Vec3 &spin_initial,
                                                           const Vec3 &spin_trial) const {
  return cache_initialised_ && i == cached_i_
         && spin_initial == cached_spin_initial_
         && spin_trial == cached_spin_trial_;
}

double jams::CachingCollectiveVariable::cached_value() {
  if (!cache_initialised_) {
    cached_value_ = calculate_expensive_value();
    cache_initialised_ = true;
  }
  return cached_value_;
}

void jams::CachingCollectiveVariable::set_cache_values(int i,
                                                       const Vec3 &spin_initial,
                                                       const Vec3 &spin_trial,
                                                       const double value,
                                                       const double trial_value) {
  cached_i_ = i;
  cached_spin_initial_ = spin_initial;
  cached_spin_trial_ = spin_trial;
  cached_value_ = value;
  cached_trial_value_ = trial_value;
  cache_initialised_ = true;
}

double jams::CachingCollectiveVariable::value() {
  return cached_value();
}

void jams::CachingCollectiveVariable::spin_move_accepted(int i,
                                                         const Vec3 &spin_initial,
                                                         const Vec3 &spin_trial) {
  if (can_use_cache_values(i, spin_initial, spin_trial)) {
    cached_value_ = cached_trial_value_;
  } else {
    cached_value_ = calculate_expensive_value();
    cache_initialised_ = true;
  }
}



#endif //INCLUDED_JAMS_CACHING_COLLECTIVE_VARIABLE
