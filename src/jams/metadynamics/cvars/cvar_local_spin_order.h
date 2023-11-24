// cvar_local_spin_order.h                                             -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_LOCAL_SPIN_ORDER
#define INCLUDED_JAMS_CVAR_LOCAL_SPIN_ORDER

#include <jams/metadynamics/caching_collective_variable.h>

#include <jams/containers/vector_set.h>

namespace jams {
    class CVarLocalSpinOrder : public CachingCollectiveVariable<double> {
    public:
        CVarLocalSpinOrder() = default;
        explicit CVarLocalSpinOrder(const libconfig::Setting &settings);

        std::string name() override;

        double value() override;

        /// Returns the value of the collective variable after a trial
        /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
        double spin_move_trial_value(
            int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

        double calculate_expensive_cache_value() override;

    private:

        double local_spin_order(const int i) const;
        double spin_order_difference(int index,
                                     const Vec3 &spin_initial,
                                     const Vec3 &spin_final) const;

        std::string name_ = "local_spin_order";

        // crude neighbour list
        std::vector<std::vector<int>> neighbour_indices_;
        std::vector<int> num_neighbour_;
        double num_spins_selected_;
    };
}
#endif
// ----------------------------- END-OF-FILE ----------------------------------