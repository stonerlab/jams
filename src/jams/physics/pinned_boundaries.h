#ifndef JAMS_PINNED_BOUNDARIES_PHYSICS_H
#define JAMS_PINNED_BOUNDARIES_PHYSICS_H

#include <jams/core/physics.h>
#include <jams/interface/config.h>
#include <jams/containers/multiarray.h>

class PinnedBoundariesPhysics : public Physics {
public:
    PinnedBoundariesPhysics(const libconfig::Setting &settings);
    ~PinnedBoundariesPhysics() = default;
    void update(const int &iterations, const double &time, const double &dt);
private:
    bool initialized;

    Vec3 left_pinned_magnetisation_ = {0.0, 0.0, -1.0};
    Vec3 right_pinned_magnetisation_ = {0.0, 0.0, 1.0};

    jams::MultiArray<int,1> left_pinned_spins_;
    jams::MultiArray<int,1> right_pinned_spins_;
};

#endif  // JAMS_PINNED_BOUNDARIES_PHYSICS_H
