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
    const int time_points = periodogram_length();
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
    store_sk_snapshot(globals::s);

    if (periodogram_window_complete())
    {
        accumulate_magnon_density();
        output_magnon_density();
        advance_periodogram_window();
    }
}

void MagnonDensityMonitor::output_magnon_density()
{
    std::ofstream ofs(jams::output::full_path_filename("magnon_density.tsv"));
    ofs << jams::fmt::decimal << " f_THz";
    ofs << jams::fmt::decimal << " E_meV";
    ofs << jams::fmt::decimal << " magnon_density_meV^-1_m^-3";
    ofs << std::endl;

    const int time_points = periodogram_length();

    // ---- Clean physical normalisation ----
    // Convert |S+|^2 → magnon occupation
    // Make spectrum integrate to magnon number
    // Output per unit real-space volume

    const double Nt = static_cast<double>(periodogram_length());
    (void)Nt;

    // For a one-sided spectrum, we apply wfreq below. Here we normalise to magnons per (m^3 * meV).
    // The discrete FFT bins have width df (THz). To convert a per-bin sum into a density, divide by df.
    const double df_thz = frequency_resolution_thz();
    const double inv_df_thz = 1.0 / df_thz;

    // Total real-space volume of the simulated supercell in m^3
    const double v = volume(globals::lattice->get_supercell()) * pow3(globals::lattice->parameter());

    // Average spin length S = mu/g across basis sites.
    // globals::mus is indexed by material, so look up via basis-site material_index.
    double avg_S = 0.0;
    for (int a = 0; a < num_basis_atoms(); ++a)
    {
        const double mu = globals::mus(a);        // Bohr magnetons
        const double S  = mu / kElectronGFactor;    // dimensionless spin length
        avg_S += S;
    }
    avg_S /= static_cast<double>(num_basis_atoms());

    // Prefactor converts accumulated |S+(q, f)|^2 into magnon number density per meV per m^3:
    //  - divide by avg_S to convert |S+|^2 -> occupation (a^\dagger a)
    //  - average over periodograms and k-points
    //  - divide by volume to get per m^3
    //  - divide by df to get per THz, then divide by kTHz2meV at output to get per meV
    const double prefactor = inv_df_thz / (avg_S * v * periodogram_window_count());

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
        ofs << jams::fmt::decimal << freq_thz * kTHz2meV;
        ofs << jams::fmt::sci << (prefactor * wfreq / kTHz2meV) * cumulative_magnon_density_(static_cast<std::size_t>(f));
        ofs << "\n";
    }

    ofs.close();
}

void MagnonDensityMonitor::accumulate_magnon_density()
{
    for (auto k = 0; k < num_k_points(); ++k)
    {
        const auto& sw = compute_frequency_spectrum_at_k(k);
        const auto time_points = periodogram_length();
        const auto freq_end = keep_negative_frequencies() ? time_points : (time_points / 2) + 1;
        const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);

        assert(cumulative_magnon_density_.size() >= static_cast<std::size_t>(freq_end));

        for (auto a = 0; a < num_basis_atoms(); ++a)
        {
            for (auto i = 0; i < freq_end; ++i)
            {
                const auto f = keep_negative_frequencies() ? (freq_start + i) % time_points : i;
                cumulative_magnon_density_(f) += std::real(sw(a, f, 0) * conj(sw(a, f, 0)));
            }
        }
    }
}
