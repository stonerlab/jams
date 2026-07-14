//
// Created by Joseph Barker on 2019-08-01.
//

#include <jams/monitors/magnon_density.h>

#include <jams/helpers/output.h>
#include "jams/core/globals.h"
#include "jams/interface/config.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "jams/core/lattice.h"

MagnonDensityMonitor::MagnonDensityMonitor(const libconfig::Setting& settings)
    : SpectrumBaseMonitor(settings, KSamplingMode::FullGrid)
{
    enable_cuda_time_fft_backend_();
    require_negative_frequencies_();
    validate_cuda_time_fft_backend_support_();

    const auto circular_channels = lowercase(
        jams::config_optional<std::string>(settings, "circular_channels", "plus"));
    auto channel_map = raise_lower_channel_map();
    if (circular_channels == "plus")
    {
        channel_map.output_channels = 1; // S+ only
    }
    else if (circular_channels == "both")
    {
        channel_map.output_channels = 2; // S+, S-
    }
    else
    {
        throw std::runtime_error("magnon-density circular_channels must be one of: plus, both");
    }
    set_channel_map(channel_map);

    const int time_points = periodogram_length();
    zero(cumulative_magnon_density_.resize(time_points, num_channels()));

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
    const bool output_both_channels = num_channels() == 2;
    std::vector<jams::output::ColDef> columns{
        {"f_THz", "THz", jams::output::ColFmt::Fixed},
        {"E_meV", "meV", jams::output::ColFmt::Fixed}};
    if (output_both_channels)
    {
        columns.push_back({"magnon_density_S+_two_sided_meV^-1_m^-3", "meV^-1 m^-3"});
        columns.push_back({"magnon_density_S-_two_sided_meV^-1_m^-3", "meV^-1 m^-3"});
        columns.push_back({"magnon_density_S+_positive_folded_meV^-1_m^-3", "meV^-1 m^-3"});
        columns.push_back({"magnon_density_S-_positive_folded_meV^-1_m^-3", "meV^-1 m^-3"});
    }
    else
    {
        columns.push_back({"magnon_density_two_sided_meV^-1_m^-3", "meV^-1 m^-3"});
        columns.push_back({"magnon_density_positive_folded_meV^-1_m^-3", "meV^-1 m^-3"});
    }
    jams::output::TsvWriter tsv(jams::output::monitor_filename(name(), "tsv"), columns);

    const int time_points = periodogram_length();

    const double df_thz = frequency_resolution_thz();
    const double v = volume(globals::lattice->get_supercell()) * pow3(globals::lattice->parameter());
    const double prefactor = 1.0 / (v * periodogram_window_count() * df_thz * kTHz2meV);
    const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
    assert(cumulative_magnon_density_.extent(0) >= static_cast<std::size_t>(time_points));
    assert(cumulative_magnon_density_.extent(1) >= static_cast<std::size_t>(num_channels()));

    const auto two_sided_density = [this, prefactor](const std::size_t f, const std::size_t c) {
        return prefactor * cumulative_magnon_density_(f, c);
    };
    const auto folded_density = [this, prefactor, time_points](
        const std::size_t f,
        const int freq_index,
        const std::size_t c) {
        if (freq_index == 0
            || ((time_points % 2) == 0 && f == static_cast<std::size_t>(time_points / 2)))
        {
            return prefactor * cumulative_magnon_density_(f, c);
        }
        if (freq_index > 0)
        {
            const auto negative_f = static_cast<std::size_t>(time_points) - f;
            return prefactor * (cumulative_magnon_density_(f, c)
                + cumulative_magnon_density_(negative_f, c));
        }
        return 0.0;
    };

    for (auto i = 0; i < time_points; ++i)
    {
        const auto f = (freq_start + i) % time_points;
        const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                       : static_cast<int>(f) - static_cast<int>(time_points);
        const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
        const auto f_index = static_cast<std::size_t>(f);

        if (output_both_channels)
        {
            tsv.write_row_values(
                freq_thz,
                freq_thz * kTHz2meV,
                two_sided_density(f_index, 0),
                two_sided_density(f_index, 1),
                folded_density(f_index, freq_index, 0),
                folded_density(f_index, freq_index, 1));
        }
        else
        {
            tsv.write_row_values(
                freq_thz,
                freq_thz * kTHz2meV,
                two_sided_density(f_index, 0),
                folded_density(f_index, freq_index, 0));
        }
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

                assert(cumulative_magnon_density_.extent(0) >= static_cast<std::size_t>(time_points));
                assert(cumulative_magnon_density_.extent(1) >= static_cast<std::size_t>(num_channels()));

                for (auto a = 0; a < num_basis_atoms(); ++a)
                {
                    const double inv_spin_length = 1.0 / basis_spin_length_(a);
                    for (auto f = 0; f < time_points; ++f)
                    {
                        for (auto c = 0; c < num_channels(); ++c)
                        {
                            cumulative_magnon_density_(f, c) +=
                                taper_weight * inv_spin_length * std::norm(sw(a, f, c));
                        }
                    }
                }
            });
    }
}
