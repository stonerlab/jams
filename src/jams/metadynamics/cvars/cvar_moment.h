// cvar_moment.h                                                       -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_MOMENT
#define INCLUDED_JAMS_CVAR_MOMENT

#include <jams/metadynamics/caching_collective_variable.h>

#include <jams/containers/vec3.h>

#include <vector>

namespace jams {
class CVarMoment : public CachingCollectiveVariable<double> {
public:
    CVarMoment() = default;
    explicit CVarMoment(const libconfig::Setting &settings);

    std::string name() override;

    double value() override;

    /// Returns the value of the collective variable after a trial
    /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
    double spin_move_trial_value(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

    double calculate_expensive_cache_value() override;

private:
    std::string name_ = "moment";
    int selected_material_id_;
    double  total_selected_moments_;
};
}


#endif //INCLUDED_JAMS_CVAR_MOMENT
