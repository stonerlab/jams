// cvar_reduced_mz.h                                                   -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_REDUCED_MZ
#define INCLUDED_JAMS_CVAR_REDUCED_MZ
/// @brief:
///
/// @details: This component...
///
/// Usage
/// -----

#include <jams/metadynamics/collective_variable.h>

namespace jams {
class CVarReducedMz : public CollectiveVariable {
public:
    CVarReducedMz() = default;
    explicit CVarReducedMz(const libconfig::Setting &settings);

    std::string name() override;

    double value() override;

    inline const jams::MultiArray<double, 2>& derivatives() override {
      throw std::runtime_error("unimplemented function");
    };

    double spin_move_trial_value(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

    void spin_move_accepted(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

protected:

    Vec3 calculate_magnetisation();

    std::string name_ = "reduced_mz";

    bool cache_initialised_ = false;
    Vec3 cached_magnetisation_ = {std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN()};
    Vec3 cached_trial_magnetisation_ = {std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN()};
    int cached_i_ = -1;
    Vec3 cached_spin_initial_ ;
    Vec3 cached_spin_trial_;
};
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------