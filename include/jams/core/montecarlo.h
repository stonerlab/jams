#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

#include <random>

#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/rand.h"
#include "jblib/containers/vec.h"

namespace {

    template <class RNG>
    Vec3 random_uniform_sphere(RNG &gen) {
      std::uniform_real_distribution<> dist;
      double v1, v2, s, ss;

      do {
        v1 = -1.0 + 2.0 * dist(gen);
        v2 = -1.0 + 2.0 * dist(gen);
        s = (v1 * v1) + (v2 * v2);
      } while (s > 1.0);

      ss = sqrt(1.0 - s);

      return {
              2.0 * v1 * ss,
              2.0 * v2 * ss,
              1.0 - 2.0 * s
      };
    }
}

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
    Vec3 operator()(Vec3 spin) {
      return -spin;
    }
};

template <class RNG>
class MonteCarloUniformMove : public MonteCarloMove {
public:
    explicit MonteCarloUniformMove(RNG * gen) :
            gen_(gen)
            {}

    Vec3 operator()(Vec3 spin) {
      return random_uniform_sphere(*gen_);
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

    Vec3 operator()(Vec3 spin) {
      spin = spin + random_uniform_sphere(*gen_) * sigma_;
      return spin / abs(spin);
    }

  private:
    double sigma_ = 0.5;
    RNG * gen_;
  };

// Trial steps as defined in Hinzke Comput. Phys. Commun. 1999
// RTS
inline Vec3 mc_reflection_trial_step(Vec3 spin) {
    return -spin;
}

// UTS
inline Vec3 mc_uniform_trial_step(Vec3 spin) {
    return rng->sphere();
}

// STS
inline Vec3 mc_angle_trial_step(Vec3 spin) {
    spin = spin + mc_uniform_trial_step(spin)*0.5;
    return spin / abs(spin);
}

// 90deg rotation with random inplane angle PTS
inline Vec3 mc_perpendicular_trial_step(Vec3 spin) {
    const double phi = rng->uniform()*kTwoPi;
    return {spin[2], sin(phi)*spin[0] + cos(phi)*spin[1], -cos(phi)*spin[0] + sin(phi)*spin[1]};
}


inline Vec3 mc_spin_as_vec(const int i) {
    return {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
}

inline void mc_set_spin_as_vec(const int i, const Vec3 spin) {
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        globals::s(i, j) = spin[j];
    }
}

inline double mc_boltzmann_probability(const double &energy, const double &beta) {
  return exp(std::min(0.0, -energy * beta));
}

inline double mc_percolation_probability(const double &energy, const double &beta) {
  return 1.0 - exp(std::min(0.0, energy * beta));
}

#endif  // JAMS_CORE_MONTECARLO_H