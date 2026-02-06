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

MagnonSpectrumMonitor::MagnonSpectrumMonitor(const libconfig::Setting& settings) : SpectrumBaseMonitor(settings)
{
    zero(cumulative_magnon_spectrum_.resize(num_motif_atoms(), num_periodogram_samples(), num_kpoints()));
    zero(mean_sublattice_directions_.resize(globals::lattice->num_basis_sites(), num_periodogram_samples()));

    do_magnon_density_ = jams::config_optional<bool>(settings, "output_magnon_density", do_magnon_density_);
    do_site_resolved_output_ = jams::config_optional<bool>(settings, "site_resolved", do_site_resolved_output_);
    do_output_negative_frequencies_ =
        jams::config_optional<bool>(settings, "output_negative_frequencies", do_output_negative_frequencies_);

    set_channel_mapping(ChannelMapping::RaiseLower);

    print_info();
}

void MagnonSpectrumMonitor::update(Solver& solver)
{
    const auto p = periodogram_index();
    for (auto i = 0; i < globals::num_spins; ++i)
    {
        Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
        auto m = globals::lattice->lattice_site_basis_index(i);
        mean_sublattice_directions_(m, p) += spin;
    }

    fourier_transform_to_kspace_and_store(globals::s);

    if (do_periodogram_update())
    {
        jams::MultiArray<Vec3, 1> mean_directions(globals::lattice->num_basis_sites());
        zero(mean_directions);

        const double wsum = [&]()
        {
            double s = 0.0;
            for (auto n = 0; n < num_periodogram_samples(); ++n)
            {
                s += fft_window_default(n, num_periodogram_samples());
            }
            return s;
        }();

        // Calculate the mean magnetisation direction for each sublattice across this periodogram period.
        // We use the same windowing function as the FFT for consistency.
        for (auto m = 0; m < globals::lattice->num_basis_sites(); ++m)
        {
            for (auto n = 0; n < num_periodogram_samples(); ++n)
            {
                mean_directions(m) += fft_window_default(n, num_periodogram_samples()) *
                    mean_sublattice_directions_(m, n);
            }
            if (wsum > 0.0) mean_directions(m) *= (1.0 / wsum);
        }

        jams::MultiArray<Mat3, 1> rotations(globals::lattice->num_basis_sites());
        for (auto m = 0; m < globals::lattice->num_basis_sites(); ++m)
        {
            // Construct a local transverse basis (e1,e2,n) with a fixed gauge.
            // This avoids the ill-conditioning of "minimal rotation" when n ≈ z.
            Vec3 n_hat = mean_directions(m);
            const double n_norm = norm(n_hat);
            if (n_norm <= 0.0)
            {
                rotations(m) = kIdentityMat3;
                continue;
            }
            n_hat *= (1.0 / n_norm);

            // Choose the global Cartesian axis least aligned with n_hat to avoid discontinuous gauge flips.
            const Vec3 ex{1.0, 0.0, 0.0};
            const Vec3 ey{0.0, 1.0, 0.0};
            const Vec3 ez{0.0, 0.0, 1.0};

            const double ax = std::abs(dot(ex, n_hat));
            const double ay = std::abs(dot(ey, n_hat));
            const double az = std::abs(dot(ez, n_hat));

            Vec3 r = ex;
            double a_min = ax;
            if (ay < a_min)
            {
                r = ey;
                a_min = ay;
            }
            if (az < a_min)
            {
                r = ez;
                a_min = az;
            }

            // e1 = normalised projection of r into the plane normal to n
            Vec3 e1 = r - dot(r, n_hat) * n_hat;
            const double e1_norm = norm(e1);
            if (e1_norm <= 0.0)
            {
                // Extremely unlikely after the fallback, but be safe.
                rotations(m) = kIdentityMat3;
                continue;
            }
            e1 *= (1.0 / e1_norm);

            // e2 completes a right-handed orthonormal triad
            Vec3 e2 = cross(n_hat, e1);

            // Build rotation matrix that maps global -> local components:
            // v_local = [e1^T; e2^T; n^T] v_global.
            Mat3 R = kIdentityMat3;
            R[0][0] = e1[0];    R[0][1] = e1[1];    R[0][2] = e1[2];
            R[1][0] = e2[0];    R[1][1] = e2[1];    R[1][2] = e2[2];
            R[2][0] = n_hat[0]; R[2][1] = n_hat[1]; R[2][2] = n_hat[2];

            rotations(m) = R;

        }
        auto spectrum = compute_periodogram_rotated_spectrum(kspace_data_timeseries_, rotations);

        element_sum(cumulative_magnon_spectrum_, calculate_magnon_spectrum(spectrum));

        if (do_site_resolved_output_)
        {
            output_site_resolved_magnon_spectrum();
        }

        output_total_magnon_spectrum();

        if (do_magnon_density_)
        {
            output_magnon_density();
        }


        shift_and_zero_mean_directions();
    }
}

void MagnonSpectrumMonitor::output_total_magnon_spectrum()
{
    for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n)
    {
        std::ofstream ofs(jams::output::full_path_filename_series("magnon_spectrum_path.tsv", n, 1));
        ofs << jams::fmt::integer << "index";
        ofs << jams::fmt::decimal << "q_total";
        ofs << jams::fmt::decimal << "h" << jams::fmt::decimal << "k" << jams::fmt::decimal << "l";
        ofs << jams::fmt::decimal << "qx" << jams::fmt::decimal << "qy" << jams::fmt::decimal << "qz";
        ofs << jams::fmt::decimal << "f_THz";
        ofs << jams::fmt::decimal << "E_meV";
        for (const std::string& k : {"+", "-", "z"})
        {
            for (const std::string& l : {"+", "-", "z"})
            {
                ofs << jams::fmt::sci << "Re_sqw_" + k + l;
                ofs << jams::fmt::sci << "Im_sqw_" + k + l;
            }
        }
        ofs << std::endl;

        // sample time is here because the fourier transform in time is not an integral
        // but a discrete sum
        auto prefactor = (sample_time_interval() / num_periodogram_periods());
        auto time_points = cumulative_magnon_spectrum_.size(1);

        jams::MultiArray<MagnonSpectrumMonitor::Mat3cx, 2> total_magnon_spectrum(
            cumulative_magnon_spectrum_.size(1), cumulative_magnon_spectrum_.size(2));
        zero(total_magnon_spectrum);
        for (auto a = 0; a < cumulative_magnon_spectrum_.size(0); ++a)
        {
            for (auto f = 0; f < cumulative_magnon_spectrum_.size(1); ++f)
            {
                for (auto k = 0; k < cumulative_magnon_spectrum_.size(2); ++k)
                {
                    total_magnon_spectrum(f, k) += cumulative_magnon_spectrum_(a, f, k);
                }
            }
        }

        auto path_begin = kspace_continuous_path_ranges_[n];
        auto path_end = kspace_continuous_path_ranges_[n + 1];
        const auto freq_end = do_output_negative_frequencies_ ? time_points : (time_points / 2) + 1;
        const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
        for (auto i = 0; i < freq_end; ++i)
        {
            const auto f = do_output_negative_frequencies_ ? (freq_start + i) % time_points : i;
            const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                           : static_cast<int>(f) - static_cast<int>(time_points);
            const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
            double total_distance = 0.0;
            for (auto j = path_begin; j < path_end; ++j)
            {
                ofs << jams::fmt::integer << j;
                ofs << jams::fmt::decimal << total_distance;
                ofs << jams::fmt::decimal << kspace_paths_[j].hkl;
                ofs << jams::fmt::decimal << kspace_paths_[j].xyz;
                ofs << jams::fmt::decimal << freq_thz; // THz
                ofs << jams::fmt::decimal << freq_thz * 4.135668; // meV
                // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
                for (auto k : {0, 1, 2})
                {
                    for (auto l : {0, 1, 2})
                    {
                        ofs << jams::fmt::sci << prefactor * total_magnon_spectrum(f, j)[k][l].real();
                        ofs << jams::fmt::sci << prefactor * total_magnon_spectrum(f, j)[k][l].imag();
                    }
                }
                if (j + 1 < path_end) {
                    total_distance += norm(kspace_paths_[j].xyz - kspace_paths_[j + 1].xyz);
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
    for (auto site = 0; site < num_motif_atoms(); ++site)
    {
        for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n)
        {
            std::ofstream ofs(jams::output::full_path_filename_series(
                "magnon_spectrum_site_" + std::to_string(site) + "_path.tsv", n, 1));

            ofs << "# site: " << site << " ";
            ofs << "material: " << globals::lattice->material_name(
                globals::lattice->basis_site_atom(site).material_index) << "\n";
            ofs << jams::fmt::integer << "index";
            ofs << jams::fmt::decimal << "q_total";
            ofs << jams::fmt::decimal << "h" << jams::fmt::decimal << "k" << jams::fmt::decimal << "l";
            ofs << jams::fmt::decimal << "qx" << jams::fmt::decimal << "qy" << jams::fmt::decimal << "qz";
            ofs << jams::fmt::decimal << "f_THz";
            ofs << jams::fmt::decimal << "E_meV";
            for (const std::string& k : {"+", "-", "z"})
            {
                for (const std::string& l : {"+", "-", "z"})
                {
                    ofs << jams::fmt::sci << "Re_sqw_" + k + l;
                    ofs << jams::fmt::sci << "Im_sqw_" + k + l;
                }
            }
            ofs << std::endl;

            // sample time is here because the fourier transform in time is not an integral
            // but a discrete sum
            auto prefactor = (sample_time_interval() / num_periodogram_periods());
            auto time_points = cumulative_magnon_spectrum_.size(1);

            auto path_begin = kspace_continuous_path_ranges_[n];
            auto path_end = kspace_continuous_path_ranges_[n + 1];
            const auto freq_end = do_output_negative_frequencies_ ? time_points : (time_points / 2) + 1;
            const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
            for (auto i = 0; i < freq_end; ++i)
            {
                const auto f = do_output_negative_frequencies_ ? (freq_start + i) % time_points : i;
                const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                               : static_cast<int>(f) - static_cast<int>(time_points);
                const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
                double total_distance = 0.0;
                for (auto j = path_begin; j < path_end; ++j)
                {
                    ofs << jams::fmt::integer << j;
                    ofs << jams::fmt::decimal << total_distance;
                    ofs << jams::fmt::decimal << kspace_paths_[j].hkl;
                    ofs << jams::fmt::decimal << kspace_paths_[j].xyz;
                    ofs << jams::fmt::decimal << freq_thz; // THz
                    ofs << jams::fmt::decimal << freq_thz * 4.135668; // meV
                    for (auto k : {0, 1, 2})
                    {
                        for (auto l : {0, 1, 2})
                        {
                            ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[k][l].real();
                            ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, f, j)[k][l].imag();
                        }
                    }
                    if (j + 1 < path_end) {
                        total_distance += norm(kspace_paths_[j].xyz - kspace_paths_[j + 1].xyz);
                    }
                    ofs << "\n";
                }
                ofs << "\n";
            }
            ofs.close();
        }
    }
}

void MagnonSpectrumMonitor::output_magnon_density()
{
    for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n)
    {
        std::ofstream ofs(jams::output::full_path_filename_series("magnon_density_path.tsv", n, 1));
        ofs << jams::fmt::decimal << "f_THz";
        ofs << jams::fmt::decimal << "E_meV";
        ofs << jams::fmt::decimal << "density";
        ofs << std::endl;

        // sample time is here because the fourier transform in time is not an integral
        // but a discrete sum
        auto prefactor = (sample_time_interval() / (num_periodogram_periods() * product(globals::lattice->kspace_size())));
        auto time_points = cumulative_magnon_spectrum_.size(1);

        jams::MultiArray<double, 1> total_magnon_density(cumulative_magnon_spectrum_.size(1));
        zero(total_magnon_density);
        for (auto a = 0; a < cumulative_magnon_spectrum_.size(0); ++a)
        {
            for (auto f = 0; f < cumulative_magnon_spectrum_.size(1); ++f)
            {
                for (auto k = 0; k < cumulative_magnon_spectrum_.size(2); ++k)
                {
                    // [0][1] => S+-
                    total_magnon_density(f) += std::abs(cumulative_magnon_spectrum_(a, f, k)[0][1]);
                }
            }
        }

        const auto freq_end = do_output_negative_frequencies_ ? time_points : (time_points / 2) + 1;
        const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
        for (auto i = 0; i < freq_end; ++i)
        {
            const auto f = do_output_negative_frequencies_ ? (freq_start + i) % time_points : i;
            const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                           : static_cast<int>(f) - static_cast<int>(time_points);
            const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();

            ofs << jams::fmt::decimal << freq_thz; // THz
            ofs << jams::fmt::decimal << freq_thz * 4.135668; // meV

            ofs << jams::fmt::sci << prefactor * total_magnon_density(f);
            ofs << "\n";
        }

        ofs.close();
    }
}

void MagnonSpectrumMonitor::shift_and_zero_mean_directions()
{
    const std::size_t M  = globals::lattice->num_basis_sites();
    const std::size_t Ns = static_cast<std::size_t>(num_periodogram_samples());
    const std::size_t ov = static_cast<std::size_t>(periodogram_overlap());

    assert(ov < Ns);

    const std::size_t src0 = Ns - ov;

    for (std::size_t m = 0; m < M; ++m)
    {
        // Copy overlap block to the start: [src0, Ns) -> [0, ov)
        auto*       dst = &mean_sublattice_directions_(m, 0);
        const auto* src = &mean_sublattice_directions_(m, src0);
        std::copy_n(src, ov, dst);

        // Zero the tail: [ov, Ns)
        std::fill_n(&mean_sublattice_directions_(m, ov), Ns - ov, Vec3{0, 0, 0});
    }
}

jams::MultiArray<MagnonSpectrumMonitor::Mat3cx, 3>
MagnonSpectrumMonitor::calculate_magnon_spectrum(const jams::MultiArray<Vec3cx, 3>& spectrum)
{
    const auto num_sites = spectrum.size(0);
    const auto num_freqencies = spectrum.size(1);
    const auto num_reciprocal_points = spectrum.size(2);

    jams::MultiArray<Mat3cx, 3> magnon_spectrum(num_sites, num_freqencies, num_reciprocal_points);
    magnon_spectrum.zero();

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
    for (auto a = 0; a < num_sites; ++a)
    {
        const Vec3 r = globals::lattice->basis_site_atom(a).position_frac;
        for (auto k = 0; k < num_reciprocal_points; ++k)
        {
            auto kpoint = kspace_paths_[k];
            auto q = kpoint.hkl;
            for (auto f = 0; f < num_freqencies; ++f)
            {
                auto sqw = spectrum(a, f, k) * exp(-kImagTwoPi * dot(q, r));

                // S+(q,w) S+(-q,-w) => S+(q,w) conj(S-(q,w))
                magnon_spectrum(a, f, k)[0][0] = sqw[0] * conj(sqw[1]);
                // S+(q,w) S-(-q,-w) => S+(q,w) conj(S+(q,w))
                magnon_spectrum(a, f, k)[0][1] = sqw[0] * conj(sqw[0]);
                // S+(q,w) Sz(-q,-w) => S+(q,w) conj(Sz(q,w))
                magnon_spectrum(a, f, k)[0][2] = sqw[0] * conj(sqw[2]);

                // S-(q,w) S+(-q,-w) => S-(q,w) conj(S-(q,w))
                magnon_spectrum(a, f, k)[1][0] = sqw[1] * conj(sqw[1]);
                // S-(q,w) S-(-q,-w) => S-(q,w) conj(S+(q,w))
                magnon_spectrum(a, f, k)[1][1] = sqw[1] * conj(sqw[0]);
                // S-(q,w) Sz(-q,-w) => S-(q,w) conj(Sz(q,w))
                magnon_spectrum(a, f, k)[1][2] = sqw[1] * conj(sqw[2]);


                // Sz(q,w) S+(-q,-w) => Sz(q,w) conj(S-(q,w))
                magnon_spectrum(a, f, k)[2][0] = sqw[2] * conj(sqw[1]);
                // Sz(q,w) S-(-q,-w) => Sz(q,w) conj(S+(q,w))
                magnon_spectrum(a, f, k)[2][1] = sqw[2] * conj(sqw[0]);
                // Sz(q,w) Sz(-q,-w) => Sz(q,w) conj(Sz(q,w))
                magnon_spectrum(a, f, k)[2][2] = sqw[2] * conj(sqw[2]);
            }
        }
    }
    return magnon_spectrum;
}
