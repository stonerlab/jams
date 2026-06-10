// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <jams/helpers/output.h>
#include <jams/core/monitor.h>
#include <jams/containers/multiarray.h>
#include <jams/containers/vec3.h>
#include <jams/monitors/spin_grouping.h>

#include <fstream>
#include <memory>
#include <vector>
#include <string>

class Solver;

class MagnetisationMonitor : public Monitor {
public:
    explicit MagnetisationMonitor(const libconfig::Setting &settings);

    ~MagnetisationMonitor() override;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    jams::output::TsvWriter make_tsv_writer(const libconfig::Setting &settings);
    void append_magnetisation_values(
        std::vector<double>& values,
        const jams::Vec<double, 3>& magnetisation,
        std::size_t group_index) const;

#if HAS_CUDA
    struct CudaBackend;

    void prepare_cuda_backend();
    void accumulate_magnetisation_cuda();
#endif

    jams::monitors::SpinGrouping grouping_ = jams::monitors::SpinGrouping::MATERIALS;
    bool normalize_magnetisation_ = true;
    std::vector<jams::monitors::SpinGroup> spin_groups_;
    std::vector<double> group_normalising_factors_;
    jams::MultiArray<double, 2> group_magnetisation_;

#if HAS_CUDA
    std::unique_ptr<CudaBackend> cuda_backend_;
#endif

    jams::output::TsvWriter tsv_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H
