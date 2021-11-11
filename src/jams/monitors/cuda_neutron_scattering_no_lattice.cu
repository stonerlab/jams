// cuda_neutron_scattering_no_lattice.cc                               -*-C++-*-
#include "jams/monitors/cuda_neutron_scattering_no_lattice.h"
#include "jams/monitors/cuda_neutron_scattering_no_lattice_kernels.cuh"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

#include <cuda.h>

CudaNeutronScatteringNoLatticeMonitor::CudaNeutronScatteringNoLatticeMonitor(const libconfig::Setting &settings)
    : Monitor(settings){

  configure_kspace_vectors(settings);

  do_rspace_windowing_ = jams::config_optional(settings, "rspace_windowing", do_rspace_windowing_);
  std::cout << "rspace windowing: " << do_rspace_windowing_ << std::endl;

//  // default to 1.0 in case no form factor is given in the settings
//  fill(neutron_form_factors_.resize(lattice->num_materials(), num_k_), 1.0);
//  if (settings.exists("form_factor")) {
//    configure_form_factors(settings["form_factor"]);
//  }
//
//  if (settings.exists("polarizations")) {
//    configure_polarizations(settings["polarizations"]);
//  }

  if (settings.exists("periodogram")) {
    configure_periodogram(settings["periodogram"]);
  }

  periodogram_props_.sample_time = output_step_freq_ * solver->time_step();


  // NOTE: the memory layout here is DIFFERENT for the CPU version
  zero(spin_timeseries_.resize(periodogram_props_.length, globals::num_spins, 3));
  zero(spin_frequencies_.resize(periodogram_props_.length / 2 + 1, globals::num_spins, 3));

  zero(total_unpolarized_neutron_cross_section_.resize(
      periodogram_props_.length, kspace_path_.size()));
  zero(total_polarized_neutron_cross_sections_.resize(
      neutron_polarizations_.size(),periodogram_props_.length, kspace_path_.size()));
}

void CudaNeutronScatteringNoLatticeMonitor::configure_periodogram(libconfig::Setting &settings) {
  periodogram_props_.length = settings["length"];
  periodogram_props_.overlap = settings["overlap"];
}


void CudaNeutronScatteringNoLatticeMonitor::configure_kspace_vectors(const libconfig::Setting &settings) {
  kmax_ = jams::config_required<double>(settings, "kmax");
  kvector_ = jams::config_required<Vec3>(settings, "kvector");
  num_k_ = jams::config_required<int>(settings, "num_k");

  kspace_path_.resize(num_k_ + 1);
  for (auto i = 0; i < kspace_path_.size(); ++i) {
    kspace_path_(i) = kvector_ * i * (kmax_ / num_k_);
  }

}

void CudaNeutronScatteringNoLatticeMonitor::store_spin_data() {
  auto t = periodogram_index_;

  auto ptr_offset = t * globals::num_spins3;

  cudaMemcpy(spin_timeseries_.device_data() + ptr_offset,
             globals::s.device_data(),
             globals::num_spins3*sizeof(double), cudaMemcpyDeviceToDevice);
}

void CudaNeutronScatteringNoLatticeMonitor::output_fixed_spectrum() {
  // Do temporal fourier transform of spin data

  const int num_time_samples = periodogram_props_.length;



  int rank = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms = globals::num_spins3;
  int nembed[1] = {num_time_samples};
  int stride = globals::num_spins3;
  int dist = 1;

  cufftHandle fft_plan;

  CHECK_CUFFT_STATUS(
      cufftCreate(&fft_plan));

  CHECK_CUFFT_STATUS(
      cufftPlanMany(&fft_plan, rank, transform_size, nembed,
                        stride, dist, nembed, stride,
                    dist, CUFFT_D2Z, num_transforms));

//  jams::MultiArray<double, 2> spin_averages(globals::num_spins, 3);
//  zero(spin_averages);
//  for (auto i = 0; i < globals::num_spins; ++i) {
//    for (auto j = 0; j < 3; ++j) {
//      for (auto t = 0; t < num_time_samples; ++t) {
//        spin_averages(i, j) += spin_timeseries_(i, j, t);
//      }
//    }
//  }
//  element_scale(spin_averages, 1.0/double(num_time_samples));
//
//
//  for (auto i = 0; i < globals::num_spins; ++i) {
//    for (auto j = 0; j < 3; ++j) {
//      for (auto t = 0; t < num_time_samples; ++t) {
//        spin_frequencies_(i, j, t) = fft_window_default(t, num_time_samples) * (spin_timeseries_(i, j, t) - spin_averages(i,j));
//      }
//    }
//  }

  CHECK_CUFFT_STATUS(
    cufftExecD2Z(fft_plan, reinterpret_cast<cufftDoubleReal*>(spin_timeseries_.device_data()),  reinterpret_cast<cufftDoubleComplex*>(spin_frequencies_.device_data())));

  CHECK_CUFFT_STATUS(
      cufftDestroy(fft_plan));

  std::ofstream debug(jams::output::full_path_filename("debug.tsv"));
  for (auto t = 0; t < num_time_samples / 2 + 1; ++t) {
    debug << t << " " << spin_frequencies_(t, 0, 0).real() << " " << spin_frequencies_(t, 0, 0).imag() << " " << spin_frequencies_(t, 0, 1).real() << " " << spin_frequencies_(t, 0, 1).imag() << std::endl;
  }
  debug.close();


  // Calculate conj(S_i^a(w)) S_j^b(w) for every i and store in a structure like Sw(i, w) (i.e. a frequency spectrum for every spin)
  jams::MultiArray<std::complex<double>,2> s_conv(globals::num_spins, num_time_samples / 2 + 1);

  zero(s_conv);
  // this assumes out kspace_path is a single straight line
  const auto delta_q = kspace_path_(1) - kspace_path_(0);
  auto unit_q = unit_vector(delta_q);


  const int num_freq = num_time_samples / 2 + 1;
  const int num_k = kspace_path_.size();

  jams::MultiArray<std::complex<double>, 2> sqw(kspace_path_.size(),
                                                num_time_samples / 2 + 1);

  dim3 block_size = {32, 32, 1};
  dim3 grid_size = {(num_k + block_size.x - 1) / block_size.x,
                    (num_freq + block_size.y - 1) / block_size.y,
                    1};

  for (auto i = 0; i < globals::num_spins; ++i) {
    std::cout << i << std::endl;

    Vec3 r_i = lattice->atom_position(i);

    spectrum_i_equal_j<<<grid_size, block_size>>>(
        i, i, globals::num_spins, num_k, num_freq, unit_q[0], unit_q[1], unit_q[2],
        reinterpret_cast<const cufftDoubleComplex*>(spin_frequencies_.device_data()),
        reinterpret_cast<cufftDoubleComplex*>(sqw.device_data())
    );
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

    for (auto j = i+1; j < globals::num_spins; ++j) {
      const Vec3 r_ij = lattice->displacement(r_i, lattice->atom_position(j));

      spectrum_i_not_equal_j<<<grid_size, block_size>>>(
          i, j, globals::num_spins, num_k, num_freq, unit_q[0], unit_q[1], unit_q[2],
          r_ij[0], r_ij[1], r_ij[2],
          reinterpret_cast<double*>(kspace_path_.device_data()),
          reinterpret_cast<const cufftDoubleComplex*>(spin_frequencies_.device_data()),
          reinterpret_cast<cufftDoubleComplex*>(sqw.device_data())
          );
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
    }

//    if (i%100 == 0) {
//      std::ofstream ofs(jams::output::full_path_filename("neutron_scattering_fixed.tsv"));
//
//      ofs << "index\t" << "qx\t" << "qy\t" << "qz\t" << "q_A-1\t";
//      ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
//      ofs << "\n";
//
//      // sample time is here because the fourier transform in time is not an integral
//      // but a discrete sum
//      auto prefactor = (periodogram_props_.sample_time / double(total_periods_)) * (1.0 / (kTwoPi * kHBarIU))
//                       * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));
//      auto barns_unitcell = prefactor / (1e-28);
//      auto freq_delta = 1.0 / (periodogram_props_.length * periodogram_props_.sample_time);
//
//      for (auto w = 0; w <  num_time_samples / 2 + 1; ++w) {
//        for (auto k = 0; k < kspace_path_.size(); ++k) {
//          ofs << jams::fmt::integer << k << "\t";
//          ofs << jams::fmt::decimal << kspace_path_(k) << "\t";
//          ofs << jams::fmt::decimal << kTwoPi * norm(kspace_path_(k)) / (lattice->parameter() * 1e10) << "\t";
//          ofs << jams::fmt::decimal << (w * freq_delta) << "\t"; // THz
//          ofs << jams::fmt::decimal << (w * freq_delta) * 4.135668 << "\t"; // meV
//          // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
//          ofs << jams::fmt::sci << barns_unitcell * sqw(k, w).real() << "\t";
//          ofs << jams::fmt::sci << barns_unitcell * sqw(k, w).imag() << "\t";
//          ofs << "\n";
//        }
//        ofs << std::endl;
//      }
//
//      ofs.close();
//    }
  }

  std::ofstream ofs(jams::output::full_path_filename("neutron_scattering_fixed.tsv"));

  ofs << "index\t" << "qx\t" << "qy\t" << "qz\t" << "q_A-1\t";
  ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
  ofs << "\n";

  // sample time is here because the fourier transform in time is not an integral
  // but a discrete sum
  auto prefactor = (periodogram_props_.sample_time / double(total_periods_)) * (1.0 / (kTwoPi * kHBarIU))
                   * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));
  auto barns_unitcell = prefactor / (1e-28);
  auto freq_delta = 1.0 / (periodogram_props_.length * periodogram_props_.sample_time);

  for (auto w = 0; w <  num_time_samples / 2 + 1; ++w) {
    for (auto k = 0; k < kspace_path_.size(); ++k) {
      ofs << jams::fmt::integer << k << "\t";
      ofs << jams::fmt::decimal << kspace_path_(k) << "\t";
      ofs << jams::fmt::decimal << kTwoPi * norm(kspace_path_(k)) / (lattice->parameter() * 1e10) << "\t";
      ofs << jams::fmt::decimal << (w * freq_delta) << "\t"; // THz
      ofs << jams::fmt::decimal << (w * freq_delta) * 4.135668 << "\t"; // meV
      // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
      ofs << jams::fmt::sci << barns_unitcell * sqw(k, w).real() << "\t";
      ofs << jams::fmt::sci << barns_unitcell * sqw(k, w).imag() << "\t";
      ofs << "\n";
    }
    ofs << std::endl;
  }

  ofs.close();


}


void CudaNeutronScatteringNoLatticeMonitor::update(Solver *solver) {
  store_spin_data();
  periodogram_index_++;

  if (is_multiple_of(periodogram_index_, periodogram_props_.length)) {


//    shift_periodogram_overlap();
    total_periods_++;


    output_fixed_spectrum();
  }
}
