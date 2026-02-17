//
// Created by Joseph Barker on 2019-08-01.
//

#include <jams/containers/mat3.h>

#include <jams/monitors/magnon_spectrum.h>


#include <jams/helpers/output.h>
#include "jams/core/lattice.h"
#include "jams/core/globals.h"

#include <complex>
#include <cmath>
#include <stdexcept>

MagnonSpectrumMonitor::MagnonSpectrumMonitor(const libconfig::Setting& settings) : SpectrumBaseMonitor(settings)
{
    // Size frequency dimension consistently with keep_negative_frequencies()
    const int time_points = periodogram_length();
    const int freq_bins = keep_negative_frequencies() ? time_points
                                                      : (time_points / 2 + 1);
    zero(cumulative_magnon_spectrum_.resize(freq_bins, num_k_points()));

    do_magnon_spectrum_output_ = jams::config_optional<bool>(settings, "output_magnon_spectrum", do_magnon_spectrum_output_);
    do_site_resolved_output_ = jams::config_optional<bool>(settings, "site_resolved", do_site_resolved_output_);

    set_channel_map(raise_lower_channel_map());
    if (num_channels() < 3)
    {
        throw std::runtime_error("MagnonSpectrumMonitor requires at least 3 output channels");
    }

    print_info();
}

void MagnonSpectrumMonitor::update(Solver& solver)
{

    store_sk_snapshot(globals::s);

    if (periodogram_window_complete())
    {
        if (do_magnon_spectrum_output_)
        {
            accumulate_magnon_spectrum();
        }

        // if (do_site_resolved_output_)
        // {
        //     output_site_resolved_magnon_spectrum();
        // }

        if (do_magnon_spectrum_output_)
        {
            output_total_magnon_spectrum();
        }

        advance_periodogram_window();

    }
}

void MagnonSpectrumMonitor::output_total_magnon_spectrum()
{
    for (auto n = 0; n < k_segment_offsets_.size() - 1; ++n)
    {
        std::ofstream ofs(jams::output::full_path_filename_series("magnon_spectrum_path.tsv", n, 1));
        ofs << jams::fmt::integer << "index";
        ofs << jams::fmt::decimal << "q_total";
        ofs << jams::fmt::decimal << "h" << jams::fmt::decimal << "k" << jams::fmt::decimal << "l";
        ofs << jams::fmt::decimal << "qx" << jams::fmt::decimal << "qy" << jams::fmt::decimal << "qz";
        ofs << jams::fmt::decimal << "f_THz";
        ofs << jams::fmt::decimal << "E_meV";
        ofs << jams::fmt::sci << "sqw_+-";
        ofs << jams::fmt::sci << "sqw_-+";
        ofs << jams::fmt::sci << "sqw_zz";
        ofs << std::endl;

        // sample time is here because the fourier transform in time is not an integral
        // but a discrete sum
        auto prefactor = (sample_time_interval() / (periodogram_window_count()));
        auto time_points = periodogram_length();

        auto path_begin = k_segment_offsets_[n];
        auto path_end = k_segment_offsets_[n + 1];
        const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
        for (auto i = 0; i < num_frequencies(); ++i)
        {
            const auto f = keep_negative_frequencies() ? (freq_start + i) % time_points : i;
            const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                           : static_cast<int>(f) - static_cast<int>(time_points);
            const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
            double wfreq = 1.0;
            if (!keep_negative_frequencies())
            {
                const bool is_dc = (f == 0);
                const bool is_nyquist = ((time_points % 2) == 0) && (f == (time_points / 2));
                wfreq = (is_dc || is_nyquist) ? 1.0 : 2.0;
            }
            double total_distance = 0.0;
            for (auto k = path_begin; k < path_end; ++k)
            {
                ofs << jams::fmt::integer << k;
                ofs << jams::fmt::decimal << total_distance;
                ofs << jams::fmt::decimal << k_points_[k].hkl;
                ofs << jams::fmt::decimal << k_points_[k].xyz;
                ofs << jams::fmt::decimal << freq_thz; // THz
                ofs << jams::fmt::decimal << freq_thz * 4.135668; // meV
                ofs << jams::fmt::sci << prefactor * wfreq * cumulative_magnon_spectrum_(f, k)[0];
                ofs << jams::fmt::sci << prefactor * wfreq * cumulative_magnon_spectrum_(f, k)[1];
                ofs << jams::fmt::sci << prefactor * wfreq * cumulative_magnon_spectrum_(f, k)[2];

                if (k + 1 < path_end) {
                    total_distance += jams::norm(k_points_[k].xyz - k_points_[k + 1].xyz);
                }
                ofs << "\n";
            }
            ofs << "\n";
        }

        ofs.close();
    }
}

void MagnonSpectrumMonitor::output_site_resolved_magnon_spectrum()
{
    // for (auto site = 0; site < num_basis_atoms(); ++site)
    // {
    //     for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n)
    //     {
    //         std::ofstream ofs(jams::output::full_path_filename_series(
    //             "magnon_spectrum_site_" + std::to_string(site) + "_path.tsv", n, 1));
    //
    //         ofs << "# site: " << site << " ";
    //         ofs << "material: " << globals::lattice->material_name(
    //             globals::lattice->basis_site_atom(site).material_index) << "\n";
    //         ofs << jams::fmt::integer << "index";
    //         ofs << jams::fmt::decimal << "q_total";
    //         ofs << jams::fmt::decimal << "h" << jams::fmt::decimal << "k" << jams::fmt::decimal << "l";
    //         ofs << jams::fmt::decimal << "qx" << jams::fmt::decimal << "qy" << jams::fmt::decimal << "qz";
    //         ofs << jams::fmt::decimal << "f_THz";
    //         ofs << jams::fmt::decimal << "E_meV";
    //         ofs << jams::fmt::sci << "Re_sqw_+-";
    //         ofs << jams::fmt::sci << "Im_sqw_+-";
    //         ofs << jams::fmt::sci << "Re_sqw_-+";
    //         ofs << jams::fmt::sci << "Im_sqw_-+";
    //         ofs << jams::fmt::sci << "Re_sqw_zz";
    //         ofs << jams::fmt::sci << "Im_sqw_zz";
    //         ofs << std::endl;
    //
    //         // sample time is here because the fourier transform in time is not an integral
    //         // but a discrete sum
    //         auto prefactor = (sample_time_interval() / num_periodogram_periods());
    //         auto time_points = cumulative_magnon_spectrum_.size(1);
    //
    //         auto path_begin = kspace_continuous_path_ranges_[n];
    //         auto path_end = kspace_continuous_path_ranges_[n + 1];
    //         const auto freq_end = do_output_negative_frequencies_ ? time_points : (time_points / 2) + 1;
    //         const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
    //         for (auto i = 0; i < freq_end; ++i)
    //         {
    //             const auto f = do_output_negative_frequencies_ ? (freq_start + i) % time_points : i;
    //             const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
    //                                                            : static_cast<int>(f) - static_cast<int>(time_points);
    //             const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
    //             double total_distance = 0.0;
    //             for (auto j = path_begin; j < path_end; ++j)
    //             {
    //                 ofs << jams::fmt::integer << j;
    //                 ofs << jams::fmt::decimal << total_distance;
    //                 ofs << jams::fmt::decimal << kspace_paths_[j].hkl;
    //                 ofs << jams::fmt::decimal << kspace_paths_[j].xyz;
    //                 ofs << jams::fmt::decimal << freq_thz; // THz
    //                 ofs << jams::fmt::decimal << freq_thz * 4.135668; // meV
    //                 ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[0].real();
    //                 ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[0].imag();
    //                 ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[1].real();
    //                 ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[1].imag();
    //                 ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[2].real();
    //                 ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[2].imag();
    //
    //                 if (j + 1 < path_end) {
    //                     total_distance += norm(kspace_paths_[j].xyz - kspace_paths_[j + 1].xyz);
    //                 }
    //                 ofs << "\n";
    //             }
    //             ofs << "\n";
    //         }
    //         ofs.close();
    //     }
    // }
}

void MagnonSpectrumMonitor::accumulate_magnon_spectrum()
{
    /// @brief Transverse dynamical structure factor @f$S^{+-}(\mathbf q,\omega)@f$.
    ///
    /// @details
    /// Uses fractional coordinates: @f$\mathbf q@f$ is in (hkl) and @f$\mathbf r@f$ is in (abc).
    ///
    /// The spectrum of interest is
    /// @f[
    ///   S^{+-}(\mathbf q,\omega)=\frac{1}{2\pi}\int_{-\infty}^{\infty} dt\;
    ///   e^{\,i\omega t}\,\big\langle S^{+}(\mathbf q,t)\,S^{-}(-\mathbf q,0)\big\rangle.
    /// @f]
    ///
    /// Evaluating the time correlation as a convolution in frequency space gives
    /// @f[
    ///   S^{+-}(\mathbf q,\omega)=\Big\langle S^{+}(\mathbf q,\omega)\,S^{-}(-\mathbf q,-\omega)\Big\rangle.
    /// @f]
    ///
    /// Using the identity @f$S^{-}(\mathbf r,t)=\mathrm{conj}\!\left(S^{+}(\mathbf r,t)\right)@f$, we have
    /// @f[
    ///   S^{-}(-\mathbf q,-\omega)=\mathrm{conj}\!\left(S^{+}(\mathbf q,\omega)\right),
    /// @f]
    /// hence
    /// @f[
    ///   S^{+-}(\mathbf q,\omega)=\mathrm{conj}\!\left(S^{+}(\mathbf q,\omega)\right)\,S^{+}(\mathbf q,\omega)
    ///   = \left|S^{+}(\mathbf q,\omega)\right|^2.
    /// @f]
    ///
    /// This provides a mapping
    ///
    /// S+(q,w) S+(-q,-w) => S+(q,w) conj(S-(q,w))
    /// S+(q,w) S-(-q,-w) => S+(q,w) conj(S+(q,w))
    /// S-(q,w) S+(-q,-w) => S-(q,w) conj(S-(q,w))
    /// S-(q,w) S-(-q,-w) => S-(q,w) conj(S+(q,w))
    ///
    /// and the sqw array contains (through the channel mapping) components
    /// 0: +  |  1: -  |  2: z

    for (auto k = 0; k < num_k_points(); ++k)
    {
        const auto& sw = compute_frequency_spectrum_at_k(k);
        const auto time_points = periodogram_length();
        const auto freq_end = keep_negative_frequencies() ? time_points : (time_points / 2) + 1;
        const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);

        // Defensive: check that cumulative_magnon_spectrum_ is sized to hold freq_end and kpoints
        assert(cumulative_magnon_spectrum_.size(0) >= static_cast<std::size_t>(freq_end));
        assert(cumulative_magnon_spectrum_.size(1) >= static_cast<std::size_t>(num_k_points()));

        for (auto a = 0; a < num_basis_atoms(); ++a)
        {
            for (auto i = 0; i < freq_end; ++i)
            {
                const auto f = keep_negative_frequencies() ? (freq_start + i) % time_points : i;

                // S+(q,w) S-(-q,-w) => S+(q,w) conj(S+(q,w))
                cumulative_magnon_spectrum_(f, k)[0] += std::real(sw(a, f, 0) * conj(sw(a, f, 0)));

                // S-(q,w) S+(-q,-w) => S-(q,w) conj(S-(q,w))
                cumulative_magnon_spectrum_(f, k)[1] += std::real(sw(a, f, 1) * conj(sw(a, f, 1)));

                // Sz(q,w) Sz(-q,-w) => Sz(q,w) conj(Sz(q,w))
                cumulative_magnon_spectrum_(f, k)[2] += std::real(sw(a, f, 2) * conj(sw(a, f, 2)));
            }
        }
    }
}
