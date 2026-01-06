//
// Created by Joseph Barker on 06/01/2026.
//

#include "jams/monitors/stability.h"

#include <iomanip>
#include <ios>
#include <ostream>

#include "jams/containers/vec3.h"
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

StabilityMonitor::StabilityMonitor(const libconfig::Setting& settings)
: Monitor(settings),
    tsv_file(jams::output::full_path_filename("stability.tsv"))
{
    tsv_file.setf(std::ios::right);
    tsv_file << tsv_header();
}


std::string StabilityMonitor::tsv_header() {
    std::stringstream ss;
    ss.width(12);

    ss << "time ";
    ss << "max_S_err" << " ";
    ss << "mean_S_err" << " ";
    ss << "rms_S_err" << " ";
    ss << std::endl;

    return ss.str();
}

void StabilityMonitor::update(Solver& solver)
{
    double max_S_err = 0.0;
    double S_err_mean = 0.0;
    double S2_err_mean = 0.0;
    for (auto i = 0; i < globals::num_spins; ++i) {
        const Vec3 s = {globals::s(i,0), globals::s(i, 1), globals::s(i,2)};
        double S_err = norm_squared(s) - 1;
        S_err_mean += S_err;
        S2_err_mean += S_err * S_err;
        if (std::abs(S_err) > max_S_err) max_S_err = std::abs(S_err);
    }

    S_err_mean /= globals::num_spins;
    S2_err_mean /= globals::num_spins;

    tsv_file.width(16);
    tsv_file << solver.time() << " ";
    tsv_file << std::scientific << std::setprecision(8) << max_S_err << " ";
    tsv_file << std::scientific << std::setprecision(8) << S_err_mean << " ";
    tsv_file << std::scientific << std::setprecision(8) << std::sqrt(S2_err_mean) << " ";
    tsv_file << std::endl;
}
