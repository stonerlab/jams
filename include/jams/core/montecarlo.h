#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/rand.h"
#include "jblib/containers/vec.h"

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

class MonteCarloUniformMove : public MonteCarloMove {
public:
    Vec3 operator()(Vec3 spin) {
      return rng->sphere();
    }
};

class MonteCarloAngleMove : public MonteCarloMove {
public:
    MonteCarloAngleMove(const double sigma) :
            sigma_(sigma){}

    Vec3 operator()(Vec3 spin) {
      spin = spin + rng->sphere() * sigma_;
      return spin / abs(spin);
    }

  private:
    double sigma_ = 0.5;
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