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

    do_site_resolved_output_ = jams::config_optional<bool>(settings, "site_resolved", do_site_resolved_output_);

    // Set channel mapping to spin raising and lowering operators
    const double inv_sqrt_2 = 1.0 / sqrt(2.0);
    Mat3cx mapping{
        inv_sqrt_2,  kImagOne*inv_sqrt_2, 0.0,
        inv_sqrt_2, -kImagOne*inv_sqrt_2, 0.0,
               0.0,               0.0, 1.0};

    set_channel_mapping(mapping);

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

    store_periodogram_data(globals::s);

    if (do_periodogram_update())
    {
        if (do_auto_basis_transform_)
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

            // Shift the sublattice direction data by the periodogram overlap and zero the end of the array
            for (auto m = 0; m < globals::lattice->num_basis_sites(); ++m)
            {
                for (auto n = 0; n < periodogram_overlap(); ++n)
                {
                    // time index
                    const auto shift_index = num_periodogram_samples() - periodogram_overlap() + n;
                    mean_sublattice_directions_(m, n) = mean_sublattice_directions_(m, shift_index);
                }
                for (auto n = periodogram_overlap(); n < num_periodogram_samples(); ++n)
                {
                    mean_sublattice_directions_(m, n) = Vec3{0, 0, 0};
                }
            }
        }
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
        for (auto i = 0; i < (time_points / 2) + 1; ++i)
        {
            double total_distance = 0.0;
            for (auto j = path_begin; j < path_end; ++j)
            {
                ofs << jams::fmt::integer << j;
                ofs << jams::fmt::decimal << total_distance;
                ofs << jams::fmt::decimal << kspace_paths_[j].hkl;
                ofs << jams::fmt::decimal << kspace_paths_[j].xyz;
                ofs << jams::fmt::decimal << i * frequency_resolution_thz(); // THz
                ofs << jams::fmt::decimal << i * frequency_resolution_thz() * 4.135668; // meV
                // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
                for (auto k : {0, 1, 2})
                {
                    for (auto l : {0, 1, 2})
                    {
                        ofs << jams::fmt::sci << prefactor * total_magnon_spectrum(i, j)[k][l].real();
                        ofs << jams::fmt::sci << prefactor * total_magnon_spectrum(i, j)[k][l].imag();
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
            for (auto i = 0; i < (time_points / 2) + 1; ++i)
            {
                double total_distance = 0.0;
                for (auto j = path_begin; j < path_end; ++j)
                {
                    ofs << jams::fmt::integer << j;
                    ofs << jams::fmt::decimal << total_distance;
                    ofs << jams::fmt::decimal << kspace_paths_[j].hkl;
                    ofs << jams::fmt::decimal << kspace_paths_[j].xyz;
                    ofs << jams::fmt::decimal << i * frequency_resolution_thz(); // THz
                    ofs << jams::fmt::decimal << i * frequency_resolution_thz() * 4.135668; // meV
                    for (auto k : {0, 1, 2})
                    {
                        for (auto l : {0, 1, 2})
                        {
                            ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, i, j)[k][l].real();
                            ofs << jams::fmt::sci << prefactor * cumulative_magnon_spectrum_(site, i, j)[k][l].imag();
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

jams::MultiArray<MagnonSpectrumMonitor::Mat3cx, 3>
MagnonSpectrumMonitor::calculate_magnon_spectrum(const jams::MultiArray<Vec3cx, 3>& spectrum)
{
    const auto num_sites = spectrum.size(0);
    const auto num_freqencies = spectrum.size(1);
    const auto num_reciprocal_points = spectrum.size(2);

    jams::MultiArray<Mat3cx, 3> magnon_spectrum(num_sites, num_freqencies, num_reciprocal_points);
    magnon_spectrum.zero();

    for (auto a = 0; a < num_sites; ++a)
    {
        // structure factor: note that q and r are in fractional coordinates (hkl, abc)
        const Vec3 r = globals::lattice->basis_site_atom(a).position_frac;
        for (auto k = 0; k < num_reciprocal_points; ++k)
        {
            auto kpoint = kspace_paths_[k];
            auto q = kpoint.hkl;
            for (auto f = 0; f < num_freqencies; ++f)
            {
                auto sqw = spectrum(a, f, k) * exp(-kImagTwoPi * dot(q, r));
                for (auto i : {0, 1, 2})
                {
                    for (auto j : {0, 1, 2})
                    {
                        magnon_spectrum(a, f, k)[i][j] = conj(sqw[i]) * sqw[j];
                    }
                }
            }
        }
    }
    return magnon_spectrum;
}
