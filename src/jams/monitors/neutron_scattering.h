// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_NEUTRON_SCATTERING_H
#define JAMS_MONITOR_NEUTRON_SCATTERING_H

#include <fstream>
#include <complex>
#include <vector>

#include <libconfig.h++>

#include "jams/interface/fft.h"
#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jams/monitors/spectrum_base.h"

class Solver;

/**
 * Monitor for calculating neutron scattering cross-sections
 *
 * Neutron scattering cross-sections contain several factors which we don't
 * want in a 'pure' spin wave spectrum. For example:
 *
 *  - the magnetic structure factor which causes forbidden reflections depending
 * on the Brillouin zone
 *  - the polarization factor where only components perpendicular to Q can be
 * measured
 *  - extended zone scheme so we are not limited to the first Brillouin zone /
 * reduced zone scheme
 *
 * This monitor includes all of these effects allowing a direct
 * comparison to neutron scattering measurments.
 *
 */
class NeutronScatteringMonitor : public SpectrumBaseMonitor {
 public:
    using Complex = std::complex<double>;

    explicit NeutronScatteringMonitor(const libconfig::Setting &settings);
    ~NeutronScatteringMonitor() override = default;

    void post_process() override {};
    void update(Solver *solver) override;

private:
    void configure_form_factors(libconfig::Setting& settings);
    void configure_polarizations(libconfig::Setting& settings);

    void output_neutron_cross_section();

    jams::MultiArray<Complex, 2> calculate_unpolarized_cross_section(const jams::MultiArray<Vec3cx, 3>& spectrum);
    jams::MultiArray<Complex, 3> calculate_polarized_cross_sections(const jams::MultiArray<Vec3cx, 3>& spectrum, const std::vector<Vec3>& polarizations);

    jams::MultiArray<double, 2> neutron_form_factors_;
    std::vector<Vec3>           neutron_polarizations_;
    jams::MultiArray<Complex,2> total_unpolarized_neutron_cross_section_;
    jams::MultiArray<Complex,3> total_polarized_neutron_cross_sections_;



};

#endif  // JAMS_MONITOR_NEUTRON_SCATTERING_H

