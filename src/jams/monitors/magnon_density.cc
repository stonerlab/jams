//
// Created by Joseph Barker on 2019-08-01.
//

#include <jams/monitors/magnon_density.h>

#include <jams/helpers/output.h>
#include "jams/core/globals.h"

#include <cmath>

#include "jams/core/lattice.h"

MagnonDensityMonitor::MagnonDensityMonitor(const libconfig::Setting& settings)
    : SpectrumBaseMonitor(settings, KSamplingMode::FullGrid)
{
    enable_cuda_time_fft_backend_();
    require_negative_frequencies_();
    validate_cuda_time_fft_backend_support_();

    const int time_points = periodogram_length();
    zero(cumulative_magnon_density_.resize(time_points));

    auto channel_map = raise_lower_channel_map();
    channel_map.output_channels = 1; // S+ only
    set_channel_map(channel_map);

    print_info();
}

void MagnonDensityMonitor::update(Solver& solver)
{
    const auto& spins = globals::s;
    store_sk_snapshot(spins);

    if (periodogram_window_complete())
    {
        accumulate_magnon_density();
        output_magnon_density();
        advance_periodogram_window();
    }
}

void MagnonDensityMonitor::output_magnon_density()
{
    jams::output::TsvWriter tsv(
        jams::output::monitor_filename(name(), "tsv"),
        {{"f_THz", "THz", jams::output::ColFmt::Fixed},
         {"E_meV", "meV", jams::output::ColFmt::Fixed},
         {"magnon_density_two_sided_meV^-1_m^-3", "meV^-1 m^-3"},
         {"magnon_density_positive_folded_meV^-1_m^-3", "meV^-1 m^-3"}});

    const int time_points = periodogram_length();

    const double df_thz = frequency_resolution_thz();
    const double v = volume(globals::lattice->get_supercell()) * pow3(globals::lattice->parameter());
    const double prefactor = 1.0 / (v * periodogram_window_count() * df_thz * kTHz2meV);
    const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
    assert(cumulative_magnon_density_.size() >= static_cast<std::size_t>(time_points));
    for (auto i = 0; i < time_points; ++i)
    {
        const auto f = (freq_start + i) % time_points;
        const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                       : static_cast<int>(f) - static_cast<int>(time_points);
        const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
        const double two_sided_density = prefactor * cumulative_magnon_density_(static_cast<std::size_t>(f));

        double folded_density = 0.0;
        if (freq_index == 0
            || ((time_points % 2) == 0 && f == time_points / 2))
        {
            folded_density = two_sided_density;
        }
        else if (freq_index > 0)
        {
            const auto negative_f = static_cast<std::size_t>(time_points - f);
            folded_density = prefactor
                * (cumulative_magnon_density_(static_cast<std::size_t>(f))
                   + cumulative_magnon_density_(negative_f));
        }

        tsv.write_row_values(
            freq_thz,
            freq_thz * kTHz2meV,
            two_sided_density,
            folded_density);
    }
}

void MagnonDensityMonitor::accumulate_magnon_density()
{
    if (accumulate_magnon_density_cuda(cumulative_magnon_density_))
    {
        return;
    }

    for (auto k = 0; k < num_k_points(); ++k)
    {
        for_each_frequency_spectrum_at_k(
            k,
            [this](const CmplxMappedSlice& sw, const double taper_weight) {
                const auto time_points = periodogram_length();

                assert(cumulative_magnon_density_.size() >= static_cast<std::size_t>(time_points));

                for (auto a = 0; a < num_basis_atoms(); ++a)
                {
                    const double inv_spin_length = 1.0 / basis_spin_length_(a);
                    for (auto f = 0; f < time_points; ++f)
                    {
                        cumulative_magnon_density_(f) += taper_weight * inv_spin_length * std::norm(sw(a, f, 0));
                    }
                }
            });
    }
}
