//
// Created by Joseph Barker on 2018-11-22.
//

#include <cuda.h>
#include <cuComplex.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_common.h"
#include "jams/helpers/duration.h"
#include "jams/helpers/random.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

#include "jams/monitors/spectrum_general.h"
#include "jams/monitors/cuda_neutron_scattering_nolattice.h"
#include "jams/monitors/cuda_neutron_scattering_nolattice_kernel.cuh"
#include "jams/helpers/consts.h"
#include "jams/cuda/cuda_common.h"
#include "jams/helpers/neutrons.h"

using namespace std;
using namespace jams;
using namespace libconfig;

namespace {
    std::vector<cuFloatComplex> generate_expQR_float(const std::vector<std::vector<Vec3>> &qvecs, const Vec3& R) {

      const auto num_qvectors = qvecs.size();
      const auto num_qpoints = qvecs[0].size();

      std::vector<cuFloatComplex> result(num_qvectors * num_qpoints);

      std::complex<float> ImagTwoPi_f = {0.0f, static_cast<float>(2.0*kTwoPi)};
      for (auto q = 0; q < num_qpoints; ++q) {
        for (auto n = 0; n < num_qvectors; ++n) {
          const std::complex<float> val = exp(ImagTwoPi_f * static_cast<float>(dot(qvecs[n][q], R)));
          result[num_qvectors * q + n] = {val.real(), val.imag()};
        }
      }
      return result;
    }
}

CudaSpectrumGeneralMonitor::CudaSpectrumGeneralMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  configure_kspace_vectors(settings);
//  configure_form_factors(settings["form_factor"]);

  if (settings.exists("polarizations")) {
    configure_polarizations(settings["polarizations"]);
  }

  if (settings.exists("periodogram")) {
    configure_periodogram(settings["periodogram"]);
  }

  periodogram_props_.sample_time = output_step_freq_ * solver->time_step();

  periodogram_props_.length = 500;
  periodogram_props_.overlap = 250;

  kspace_path_.resize(num_k_);
  for (auto i = 0; i < kspace_path_.size(); ++i) {
    kspace_path_(i) = kvector_ * i * (kmax_ / num_k_);
  }

  rspace_displacement_.resize(globals::s.size(0));
  for (auto i = 0; i < globals::s.size(0); ++i) {
    rspace_displacement_(i) = lattice->displacement({0,0,0}, lattice->atom_position(i));
  }

  zero(kspace_spins_timeseries_.resize(periodogram_props_.length, kspace_path_.size()));
  zero(total_unpolarized_neutron_cross_section_.resize(
      periodogram_props_.length, kspace_path_.size()));
  zero(total_polarized_neutron_cross_sections_.resize(
      neutron_polarizations_.size(),periodogram_props_.length, kspace_path_.size()));
}

void CudaSpectrumGeneralMonitor::update(Solver *solver) {
  store_kspace_data_on_path();
  periodogram_index_++;

  if (is_multiple_of(periodogram_index_, periodogram_props_.length)) {
    auto spectrum = periodogram();
    shift_periodogram_overlap();
    total_periods_++;

    element_sum(total_unpolarized_neutron_cross_section_, calculate_unpolarized_cross_section(spectrum));

    if (!neutron_polarizations_.empty()) {
      element_sum(total_polarized_neutron_cross_sections_,
                  calculate_polarized_cross_sections(spectrum, neutron_polarizations_));
    }


    output_neutron_cross_section();
  }
}

//CudaSpectrumGeneralMonitor::~CudaSpectrumGeneralMonitor() {
//  using namespace std;
//  using namespace std::chrono;
//  using namespace std::placeholders;
//  using namespace globals;
//
//  cout << "calculating correlation function" << std::endl;
//  auto start_time = time_point_cast<milliseconds>(system_clock::now());
//  cout << "start   " << get_date_string(start_time) << "\n\n";
//  cout.flush();
//
//  std::cout << duration_string(start_time, system_clock::now()) << " calculating fft time => frequency" << std::endl;
//
//  this->apply_time_fourier_transform();
//
//  std::cout << duration_string(start_time, system_clock::now()) << " done" << std::endl;
//
//  jams::MultiArray<cuFloatComplex, 1> spin_data_float_(spin_data_.elements());
//
//  auto count = 0;
//  for (auto i = 0; i < spin_data_.size(0); ++i) {
//    for (auto j = 0; j < spin_data_.size(1); ++j) {
//      spin_data_float_(count).x = static_cast<float>(spin_data_(i,j).real());
//      spin_data_float_(count).y = static_cast<float>(spin_data_(i,j).imag());
//      count++;
//    }
//  }
//
//  std::vector<std::vector<Vec3>> qvecs(num_qvectors_);
//  for (auto n = 0; n < num_qvectors_; ++n) {
//    auto qvec_rand = qmax_ * uniform_random_sphere(jams::random_generator());
//    std::cout << "qvec " << n << ": " << qvec_rand << std::endl;
//    std::vector<Vec3> qpoints(num_qpoints_);
//    for (auto i = 0; i < num_qpoints_; ++i){
//      qpoints[i] = qvec_rand * (i / double(num_qpoints_-1));
//    }
//    qvecs[n] = qpoints;
//  }
//
//
//  vector<Vec3> r(num_spins);
//  for (auto i = 0; i < num_spins; ++i) {
//    r[i] = lattice->atom_position(i);
//  }
//
//  // support for lattice vacancies (we will skip these in the spectrum loop)
//  vector<bool> is_vacancy(num_spins, false);
//  for (auto i = 0; i < num_spins; ++i) {
//    if (s(i, 0) == 0.0 && s(i, 1) == 0.0 && s(i, 2) == 0.0) {
//      is_vacancy[i] = true;
//    }
//  }
//
//
//
//  jams::MultiArray<cuFloatComplex, 2> SQw(num_qpoints_, padded_size_/2+1);
//  SQw.zero();
//
//  const auto num_w_points = padded_size_/2+1;
//
//  cuFloatComplex *dev_qfactors = nullptr;
//  CHECK_CUDA_STATUS(cudaMalloc((void**)&dev_qfactors, (num_qpoints_ * num_qvectors_)*sizeof(cuFloatComplex)));
//
//  const dim3 block_size = {64, 8, 1};
//  auto grid_size = cuda_grid_size(block_size, {num_w_points, num_qpoints_, 1});
//
//  // generate spectrum looping over all i,j
//  for (unsigned i = 0; i < globals::num_spins; ++i) {
//    if (is_vacancy[i]) continue;
//    std::cout << duration_string(start_time, system_clock::now()) << " " << i << std::endl;
//    for (unsigned j = 0; j < globals::num_spins; ++j) {
//      if (is_vacancy[j]) continue;
//
////      for (unsigned n = 0; n < qvecs.size(); ++n) {
////       precalculate the exponential factors for the spatial fourier transform
//        const auto qfactors = generate_expQR_float(qvecs, lattice->displacement(j, i));
//
//
//      CHECK_CUDA_STATUS(cudaMemcpy(dev_qfactors, qfactors.data(),
//                                        num_qpoints_ * num_qvectors_ * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
//
//
//        CudaSpectrumGeneralKernel <<< grid_size, block_size >> >
//                                                  (i, j, num_w_points, num_qpoints_, num_qvectors_, padded_size_, dev_qfactors, spin_data_float_.device_data(), SQw.device_data());
//        DEBUG_CHECK_CUDA_ASYNC_STATUS;
////      }
//    }
//
//    if (i%10 == 0) {
//      std::ofstream cfile(seedname + "_corr.tsv");
//      cfile << "q\tfrequency\tRe_SQw\tIm_SQw\n";
//      for (unsigned q = 0; q < num_qpoints_; ++q) {
//        for (unsigned w = 0; w < padded_size_/2+1; ++w) {
//          cfile << qmax_ * (q / double(num_qpoints_-1)) << "\t";
//          cfile << 0.5*w * freq_delta_ << "\t";
//          cfile << SQw(q, w).x / static_cast<double>(padded_size_*(i + 1)*num_qvectors_) << "\t";
//          cfile << SQw(q, w).y / static_cast<double>(padded_size_*(i + 1)*num_qvectors_) << "\n";
//        }
//      }
//      cfile.flush();
//      cfile.close();
//    }
//
//  }
//
//  auto end_time = time_point_cast<milliseconds>(system_clock::now());
//  cout << "finish  " << get_date_string(end_time) << "\n\n";
//  cout << "runtime " << duration_string(start_time, end_time) << "\n";
//  cout.flush();
//
//}

void CudaSpectrumGeneralMonitor::configure_kspace_vectors(const libconfig::Setting &settings) {
  kvector_ = jams::config_optional<Vec3>(settings, "kvector", kvector_);
}

jams::MultiArray<Complex, 2>
CudaSpectrumGeneralMonitor::calculate_unpolarized_cross_section(const jams::MultiArray<Vec3cx,2> &spectrum) {
  const auto num_freqencies = spectrum.size(0);
  const auto num_reciprocal_points = kspace_path_.size();

  jams::MultiArray<Complex, 2> cross_section(num_freqencies, num_reciprocal_points);
  cross_section.zero();

  for (auto f = 0; f < num_freqencies; ++f) {
    for (auto k = 0; k < num_reciprocal_points; ++k) {
        auto Q = unit_vector(kspace_path_(k));
          auto s_a = conj(spectrum(f, k));
          auto s_b = spectrum(f, k);

          for (auto i : {0, 1, 2}) {
            for (auto j : {0, 1, 2}) {
              cross_section(f, k) += (kronecker_delta(i, j) - Q[i] * Q[j]) * s_a[i] * s_b[j];
            }
          }
        }
      }
  return cross_section;
}

jams::MultiArray<Complex, 3>
CudaSpectrumGeneralMonitor::calculate_polarized_cross_sections(const MultiArray<Vec3cx, 2> &spectrum,
                                                               const vector<Vec3> &polarizations) {
  const auto num_freqencies = spectrum.size(0);
  const auto num_reciprocal_points = kspace_path_.size();

  MultiArray<Complex, 3> convolved(polarizations.size(), num_freqencies, num_reciprocal_points);
  convolved.zero();

  for (auto f = 0; f < num_freqencies; ++f) {
    for (auto k = 0; k < num_reciprocal_points; ++k) {
      auto Q = unit_vector(kspace_path_(k));
      auto s_a = conj(spectrum(f, k));
      auto s_b = spectrum(f, k);
      for (auto p = 0; p < polarizations.size(); ++p) {
        auto P = polarizations[p];
        auto PxQ = cross(P, Q);

        convolved(p, f, k) += kImagOne * dot(P, cross(s_a, s_b));

        for (auto i : {0, 1, 2}) {
          for (auto j : {0, 1, 2}) {
            convolved(p, f, k) += kImagOne * PxQ[i] * Q[j] * (s_a[i] * s_b[j] - s_a[j] * s_b[i]);
          }
        }
      }
    }
  }
  return convolved;
}

jams::MultiArray<Vec3cx,2> CudaSpectrumGeneralMonitor::periodogram() {
  jams::MultiArray<Vec3cx,2> spectrum(kspace_spins_timeseries_);

  const int num_time_samples  = spectrum.size(0);
  const int num_kspace_samples = spectrum.size(1);

  int rank = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms = num_kspace_samples * 3;
  int nembed[1] = {num_time_samples};
  int stride = num_kspace_samples * 3;
  int dist = 1;

  fftw_plan fft_plan = fftw_plan_many_dft(rank, transform_size, num_transforms,
                                          FFTW_COMPLEX_CAST(spectrum.begin()), nembed, stride, dist,
                                          FFTW_COMPLEX_CAST(spectrum.begin()), nembed, stride, dist,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);

  assert(fft_plan);

  for (auto i = 0; i < num_time_samples; ++i) {
    for (auto j = 0; j < num_kspace_samples; ++j) {
      spectrum(i, j) *= fft_window_default(i, num_time_samples);
    }
  }

  fftw_execute(fft_plan);
  fftw_destroy_plan(fft_plan);

  element_scale(spectrum, 1.0 / double(num_time_samples));

  return spectrum;
}

void CudaSpectrumGeneralMonitor::shift_periodogram_overlap() {
  // shift overlap data to the start of the range
  for (auto i = 0; i < periodogram_props_.overlap; ++i) {
    for (auto j = 0; j < kspace_spins_timeseries_.size(1); ++j) {
      kspace_spins_timeseries_(i, j) = kspace_spins_timeseries_(kspace_spins_timeseries_.size(0) - periodogram_props_.overlap + i, j);
    }
  }

  // put the pointer to the overlap position
  periodogram_index_ = periodogram_props_.overlap;
}

void CudaSpectrumGeneralMonitor::output_neutron_cross_section() {
    ofstream ofs(seedname + "_neutron_scattering_path_" + to_string(0) + ".tsv");

    ofs << "index\t" << "qx\t" << "qy\t" << "qz\t" << "q\t";
    ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
    for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
      ofs << "sigma_pol" << to_string(k) << "_re\t" << "sigma_pol" << to_string(k) << "_im\t";
    }
    ofs << "\n";

    // sample time is here because the fourier transform in time is not an integral
    // but a discrete sum
    auto prefactor = (periodogram_props_.sample_time / double(total_periods_)) * (1.0 / (kTwoPi * kHBar))
                     * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));
    auto barns_unitcell = prefactor / (1e-28);
    auto time_points = total_unpolarized_neutron_cross_section_.size(0);
    auto freq_delta = 1.0 / (periodogram_props_.length * periodogram_props_.sample_time);

    for (auto i = 0; i < (time_points / 2) + 1; ++i) {
      for (auto j = 0; j < kspace_path_.size(); ++j) {
        ofs << fmt::integer << j << "\t";
        ofs << fmt::decimal << kspace_path_(j) << "\t";
        ofs << fmt::decimal << norm(kspace_path_(j)) << "\t";
        ofs << fmt::decimal << (i * freq_delta / 1e12) << "\t"; // THz
        ofs << fmt::decimal << (i * freq_delta / 1e12) * 4.135668 << "\t"; // meV
        // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
        ofs << fmt::sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).real() << "\t";
        ofs << fmt::sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).imag() << "\t";
        for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
          ofs << fmt::sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).real() << "\t";
          ofs << fmt::sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).imag() << "\t";
        }
        ofs << "\n";
      }
      ofs << endl;
    }

    ofs.close();
}

void CudaSpectrumGeneralMonitor::store_kspace_data_on_path() {
  auto i = periodogram_index_;

  for (auto k = 0; k < kspace_path_.size(); ++k) {
    Vec3cx sum = {0.0, 0.0};
    for (auto n = 0; n < globals::num_spins; ++n) {
      Vec3 spin = {globals::s(n,0), globals::s(n,1), globals::s(n,2)};
      Vec3 r = rspace_displacement_(n);
      auto q = kspace_path_(k);

      // apply 3D window in space
      auto window = pow2(cos(kPi*norm(r)));
      auto f = exp(-kImagTwoPi * dot(q, r));
      sum += f * spin * window;
    }
    kspace_spins_timeseries_(i, k) = sum / double(globals::num_spins);
  }
}

void CudaSpectrumGeneralMonitor::configure_polarizations(libconfig::Setting &settings) {
  for (auto i = 0; i < settings.getLength(); ++i) {
    neutron_polarizations_.push_back({
                                         double{settings[i][0]}, double{settings[i][1]}, double{settings[i][2]}});
  }
}

void CudaSpectrumGeneralMonitor::configure_periodogram(libconfig::Setting &settings) {
  periodogram_props_.length = settings["length"];
  periodogram_props_.overlap = settings["overlap"];
}

//void CudaSpectrumGeneralMonitor::configure_form_factors(Setting &settings) {
//  auto gj = read_form_factor_settings(settings);
//
//  auto num_sites     = lattice->num_motif_atoms();
//  auto num_materials = lattice->num_materials();
//
//  if (settings.getLength() != num_materials) {
//    throw runtime_error("NeutronScatteringMonitor:: there must be one form factor per material\"");
//  }
//
//  vector<FormFactorG> g_params(num_materials);
//  vector<FormFactorJ> j_params(num_materials);
//
//  for (auto i = 0; i < settings.getLength(); ++i) {
//    for (auto l : {0,2,4,6}) {
//      j_params[i][l] = config_optional<FormFactorCoeff>(settings[i], "j" + to_string(l), j_params[i][l]);
//    }
//    g_params[i] = config_required<FormFactorG>(settings[i], "g");
//  }
//
//  neutron_form_factors_.resize(num_sites, kspace_paths_.size());
//  for (auto a = 0; a < num_sites; ++a) {
//    for (auto i = 0; i < kspace_paths_.size(); ++i) {
//      auto m = lattice->motif_atom(a).material;
//      auto q = kspace_paths_[i].xyz;
//      neutron_form_factors_(a, i) = form_factor(q, kMeterToAngstroms * lattice->parameter(), g_params[m], j_params[m]);
//    }
//  }
//}