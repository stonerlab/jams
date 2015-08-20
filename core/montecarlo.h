#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

// Trial steps as defined in Hinzke Comput. Phys. Commun. 1999
// RTS
inline jblib::Vec3<double> mc_reflection_trial_step(jblib::Vec3<double> spin) {
    return -spin;
}

// UTS
inline jblib::Vec3<double> mc_uniform_trial_step(jblib::Vec3<double> spin) {
    rng.sphere(spin.x, spin.y, spin.z);
    return spin;
}

// STS
inline jblib::Vec3<double> mc_small_trial_step(jblib::Vec3<double> spin) {
    spin = spin + mc_uniform_trial_step(spin)*0.1;
    return spin / abs(spin);
}

inline jblib::Vec3<double> mc_spin_as_vec(const int i) {
    return jblib::Vec3<double>(globals::s(i,0), globals::s(i,1), globals::s(i,2));
}

inline void mc_set_spin_as_vec(const int i, const jblib::Vec3<double> spin) {
    for (int j = 0; j < 3; ++j) {
        globals::s(i, j) = spin[j];
    }
}

#endif  // JAMS_CORE_MONTECARLO_H