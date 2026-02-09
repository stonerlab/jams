//
// Created by Joseph Barker on 2019-08-01.
//

#include <jams/monitors/magnon_density.h>

#include <jams/helpers/output.h>
#include "jams/core/globals.h"

#include <cmath>

MagnonDensityMonitor::MagnonDensityMonitor(const libconfig::Setting& settings)
    : SpectrumBaseMonitor(settings, KSpacePathMode::FullOnly)
{
    const int time_points = num_periodogram_samples();
    const int freq_bins = keep_negative_frequencies() ? time_points
                                                      : (time_points / 2 + 1);
    zero(cumulative_magnon_density_.resize(freq_bins));

    auto channel_map = raise_lower_channel_map();
    channel_map.output_channels = 1; // S+ only
    set_channel_map(channel_map);

    print_info();
}

void MagnonDensityMonitor::update(Solver& solver)
{
    fourier_transform_to_kspace_and_store(globals::s);

    if (do_periodogram_update())
    {
        accumulate_magnon_density();
        output_magnon_density();
        shift_periodogram();
    }
}

void MagnonDensityMonitor::output_magnon_density()
{
    std::ofstream ofs(jams::output::full_path_filename("magnon_density.tsv"));
    ofs << jams::fmt::decimal << "f_THz";
    ofs << jams::fmt::decimal << "E_meV";
    ofs << jams::fmt::decimal << "density";
    ofs << std::endl;

    const int time_points = num_periodogram_samples();
    const double prefactor = (num_kpoints() > 0)
      ? (sample_time_interval() / (num_periodogram_periods() * static_cast<double>(num_kpoints())))
      : 0.0;

    const auto freq_end = keep_negative_frequencies() ? time_points : (time_points / 2) + 1;
    const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
    assert(cumulative_magnon_density_.size() >= static_cast<std::size_t>(freq_end));
    for (auto i = 0; i < freq_end; ++i)
    {
        const auto f = keep_negative_frequencies() ? (freq_start + i) % time_points : i;

        double wfreq = 1.0;
        if (!keep_negative_frequencies()) {
          const bool is_dc = (f == 0);
          const bool is_nyquist = ((time_points % 2) == 0) && (f == (time_points / 2));
          wfreq = (is_dc || is_nyquist) ? 1.0 : 2.0;
        }

        const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                       : static_cast<int>(f) - static_cast<int>(time_points);
        const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();

        ofs << jams::fmt::decimal << freq_thz;
        ofs << jams::fmt::decimal << freq_thz * 4.135668;
        ofs << jams::fmt::sci << (prefactor * wfreq) * cumulative_magnon_density_(static_cast<std::size_t>(f));
        ofs << "\n";
    }

    ofs.close();
}

void MagnonDensityMonitor::accumulate_magnon_density()
{
    for (auto k = 0; k < num_kpoints(); ++k)
    {
        const auto& sw = fft_sk_timeseries_to_skw(k);
        const auto time_points = num_periodogram_samples();
        const auto freq_end = keep_negative_frequencies() ? time_points : (time_points / 2) + 1;
        const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);

        assert(cumulative_magnon_density_.size() >= static_cast<std::size_t>(freq_end));

        for (auto a = 0; a < num_motif_atoms(); ++a)
        {
            for (auto i = 0; i < freq_end; ++i)
            {
                const auto f = keep_negative_frequencies() ? (freq_start + i) % time_points : i;
                cumulative_magnon_density_(f) += std::real(sw(a, f, 0) * conj(sw(a, f, 0)));
            }
        }
    }
}
