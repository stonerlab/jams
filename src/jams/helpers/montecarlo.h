#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

#include <random>

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "random.h"

enum class MonteCarloMoveType {
    REFLECTION,
    UNIFORM,
    ANGLE
};

class MonteCarloMove {
public:
    virtual Vec3 operator()(Vec3 spin) = 0;
};

class MonteCarloReflectionMove : public MonteCarloMove {
public:
    inline Vec3 operator()(Vec3 spin) {
      return -spin;
    }
};

template <class RNG>
class MonteCarloUniformMove : public MonteCarloMove {
public:
    explicit MonteCarloUniformMove(RNG * gen) :
            gen_(gen)
            {}

    inline Vec3 operator()(Vec3 spin) {
      return uniform_random_sphere(*gen_);
    }
private:
    RNG * gen_;
};

template <class RNG>
class MonteCarloAngleMove : public MonteCarloMove {
public:
    MonteCarloAngleMove(RNG * gen, const double sigma) :
            gen_(gen),
            sigma_(sigma){}

    inline Vec3 operator()(Vec3 spin) {
      return normalize(spin + uniform_random_sphere(*gen_) * sigma_);
    }

  private:
    double sigma_ = 0.5;
    RNG * gen_;
  };

inline Vec3 mc_spin_as_vec(int i) {
    return {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
}

inline void mc_set_spin_as_vec(int i, const Vec3& spin) {
  globals::s(i, 0) = spin[0];
  globals::s(i, 1) = spin[1];
  globals::s(i, 2) = spin[2];
}

#endif  // JAMS_CORE_MONTECARLO_H