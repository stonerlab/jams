// magnetisation_layers.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_MONITORS_MAGNETISATION_LAYERS
#define INCLUDED_JAMS_MONITORS_MAGNETISATION_LAYERS

#include <jams/core/monitor.h>
#include <jams/interface/config.h>
#include <jams/containers/multiarray.h>

#include <vector>

class Solver;

class MagnetisationLayersMonitor : public Monitor {
public:
    explicit MagnetisationLayersMonitor(const libconfig::Setting &settings);

    ~MagnetisationLayersMonitor() override = default;

    void update(Solver *solver) override;
    inline void post_process() override {};

    inline bool is_converged() override { return false; };

private:
    int num_layers_;
    jams::MultiArray<double,2> layer_magnetisation_;
    std::vector<jams::MultiArray<int,1>>    layer_spin_indicies_;
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------