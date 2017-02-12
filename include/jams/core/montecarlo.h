#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

#include "jams/core/consts.h"
#include "jams/core/rand.h"
#include "jblib/containers/vec.h"

// Trial steps as defined in Hinzke Comput. Phys. Commun. 1999
// RTS
inline jblib::Vec3<double> mc_reflection_trial_step(jblib::Vec3<double> spin) {
    return -spin;
}

// UTS
inline jblib::Vec3<double> mc_uniform_trial_step(jblib::Vec3<double> spin) {
    rng->sphere(spin.x, spin.y, spin.z);
    return spin;
}

// STS
inline jblib::Vec3<double> mc_small_trial_step(jblib::Vec3<double> spin) {
    spin = spin + mc_uniform_trial_step(spin)*0.5;
    return spin / abs(spin);
}

// 90deg rotation with random inplane angle PTS
inline jblib::Vec3<double> mc_perpendicular_trial_step(jblib::Vec3<double> spin) {
    const double phi = rng->uniform()*kTwoPi;
    return jblib::Vec3<double>(spin.z, sin(phi)*spin.x + cos(phi)*spin.y, -cos(phi)*spin.x + sin(phi)*spin.y);
}


inline jblib::Vec3<double> mc_spin_as_vec(const int i) {
    return jblib::Vec3<double>(globals::s(i,0), globals::s(i,1), globals::s(i,2));
}

inline void mc_set_spin_as_vec(const int i, const jblib::Vec3<double> spin) {
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