//
// Created by Joseph Barker on 2019-05-02.
//

#ifndef JAMS_MONITOR_UNITCELL_AVERAGE_H
#define JAMS_MONITOR_UNITCELL_AVERAGE_H

#include "jams/core/monitor.h"

class UnitcellAverageMonitor : public Monitor {
public:
    explicit UnitcellAverageMonitor(const libconfig::Setting &settings);
    ~UnitcellAverageMonitor();

    void update(Solver * solver) override;
    void post_process() override {};

private:
    void open_new_xdmf_file(const std::string &xdmf_file_name);
    void update_xdmf_file(const std::string &h5_file_name);
    void write_h5_file(const std::string &h5_file_name);

    bool         compression_enabled_ = true;
    Slice        slice_;
    FILE*        xdmf_file_;
    std::vector<Mat3> spin_transformations_;
    jams::MultiArray<double, 2> cell_centers_;
    jams::MultiArray<double, 2> cell_mag_;
    jams::MultiArray<double, 2> cell_neel_;
};

#endif //JAMS_MONITOR_UNITCELL_AVERAGE_H
