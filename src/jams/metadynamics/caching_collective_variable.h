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

template <class CacheType>
class CachingCollectiveVariable : public CollectiveVariable {
private:
    bool cache_initialised_ = false;
    CacheType cached_value_;
    CacheType cached_trial_value_;
    int cached_i_ = -1;
    Vec3 cached_spin_initial_;
    Vec3 cached_spin_trial_;

protected:
    inline bool cache_is_initialised() const;

    inline bool can_use_cache_values(int i, const Vec3 &spin_initial,
                                     const Vec3 &spin_trial) const;

    inline CacheType cached_value();

    inline void
    set_cache_values(int i, const Vec3 &spin_initial, const Vec3 &spin_trial,
                     const CacheType value, const CacheType trial_value);

public:
    virtual inline double value() override = 0;

    virtual CacheType calculate_expensive_cache_value() = 0;

    inline void spin_move_accepted(int i, const Vec3 &spin_initial,
                            const Vec3 &spin_trial) override;


};

}

// INLINE FUNCTIONS
template <class CacheType>
bool jams::CachingCollectiveVariable<CacheType>::cache_is_initialised() const {
  return cache_initialised_;
}

template <class CacheType>
bool jams::CachingCollectiveVariable<CacheType>::can_use_cache_values(int i,
                                                           const Vec3 &spin_initial,
                                                           const Vec3 &spin_trial) const {
  return cache_initialised_ && i == cached_i_
         && spin_initial == cached_spin_initial_
         && spin_trial == cached_spin_trial_;
}

template <class CacheType>
CacheType jams::CachingCollectiveVariable<CacheType>::cached_value() {
  if (!cache_initialised_) {
    cached_value_ = calculate_expensive_cache_value();
    cache_initialised_ = true;
  }
  return cached_value_;
}

template <class CacheType>
void jams::CachingCollectiveVariable<CacheType>::set_cache_values(int i,
                                                       const Vec3 &spin_initial,
                                                       const Vec3 &spin_trial,
                                                       const CacheType value,
                                                       const CacheType trial_value) {
  cached_i_ = i;
  cached_spin_initial_ = spin_initial;
  cached_spin_trial_ = spin_trial;
  cached_value_ = value;
  cached_trial_value_ = trial_value;
  cache_initialised_ = true;
}

template <class CacheType>
void jams::CachingCollectiveVariable<CacheType>::spin_move_accepted(int i,
                                                         const Vec3 &spin_initial,
                                                         const Vec3 &spin_trial) {
  if (can_use_cache_values(i, spin_initial, spin_trial)) {
    cached_value_ = cached_trial_value_;
  } else {
    cached_value_ = calculate_expensive_cache_value();
    cache_initialised_ = true;
  }
}



#endif //INCLUDED_JAMS_CACHING_COLLECTIVE_VARIABLE
