//
// Created by Joseph Barker on 2018-11-22.
//

#include <cuda.h>
#include <cuComplex.h>

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/helpers/duration.h"

#include "jams/monitors/spectrum_general.h"
#include "jams/monitors/cuda_spectrum_general.h"
#include "jams/monitors/cuda_spectrum_general_kernel.cuh"

namespace {
    std::vector<cuFloatComplex> generate_expQR_float(const std::vector<Vec3> &qvecs, const Vec3& R) {
      std::vector<cuFloatComplex> result(qvecs.size());

      for (auto q = 0; q < result.size(); ++q) {
        const auto val = exp(kImagTwoPi * dot(qvecs[q], R));
        result[q].x = val.real();
        result[q].y = val.imag();
      }
      return result;
    }
}

CudaSpectrumGeneralMonitor::CudaSpectrumGeneralMonitor(const libconfig::Setting &settings) : SpectrumGeneralMonitor(
        settings) {

}

CudaSpectrumGeneralMonitor::~CudaSpectrumGeneralMonitor() {
  using namespace std;
  using namespace std::chrono;
  using namespace std::placeholders;
  using namespace globals;

  cout << "calculating correlation function" << std::endl;
  auto start_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "start   " << get_date_string(start_time) << "\n\n";
  cout.flush();

  std::cout << duration_string(start_time, system_clock::now()) << " calculating fft time => frequency" << std::endl;

  this->apply_time_fourier_transform();

  std::cout << duration_string(start_time, system_clock::now()) << " done" << std::endl;

  jblib::Array<cuFloatComplex, 1> hst_spin_data(spin_data_.elements());
  for (auto i = 0; i < spin_data_.elements(); ++i) {
    hst_spin_data[i].x = static_cast<float>(spin_data_[i].real());
    hst_spin_data[i].y = static_cast<float>(spin_data_[i].imag());
  }
  jblib::CudaArray<cuFloatComplex, 1> dev_spin_data(hst_spin_data);


// precalculate a few quantities we need
  vector<Vec3> qvecs(num_q_, {0.0, 0.0, 0.0});
  for (auto i = 0; i < num_q_; ++i){
    qvecs[i] = qmax_ * (i / double(num_q_-1));
  }

  vector<Vec3> r(num_spins);
  for (auto i = 0; i < num_spins; ++i) {
    r[i] = lattice->atom_position(i);
  }

  // support for lattice vacancies (we will skip these in the spectrum loop)
  vector<bool> is_vacancy(num_spins, false);
  for (auto i = 0; i < num_spins; ++i) {
    if (s(i, 0) == 0.0 && s(i, 1) == 0.0 && s(i, 2) == 0.0) {
      is_vacancy[i] = true;
    }
  }



  jblib::Array<cuFloatComplex, 2> SQw(qvecs.size(), padded_size_/2+1);
  for (auto i = 0; i < SQw.elements(); ++i) {
    SQw[i].x = 0.0;
    SQw[i].y = 0.0;
  }

  jblib::CudaArray<cuFloatComplex, 1> dev_SQw(SQw);

  const auto num_w_points = padded_size_/2+1;

  cuFloatComplex *dev_qfactors = nullptr;
  cuda_api_error_check(cudaMalloc((void**)&dev_qfactors, (qvecs.size())*sizeof(cuFloatComplex)));

  const dim3 block_size = {64, 8, 1};
  auto grid_size = cuda_grid_size(block_size, {num_w_points, qvecs.size(), 1});

  // generate spectrum looping over all i,j
  for (unsigned i = 0; i < globals::num_spins; ++i) {
    if (is_vacancy[i]) continue;
    std::cout << duration_string(start_time, system_clock::now()) << " " << i << std::endl;
    for (unsigned j = 0; j < globals::num_spins; ++j) {
      if (is_vacancy[j]) continue;

//       precalculate the exponential factors for the spatial fourier transform
      const auto qfactors = generate_expQR_float(qvecs, lattice->displacement(r[j], r[i]));


      cuda_api_error_check(cudaMemcpy(dev_qfactors, qfactors.data(),
                                      qfactors.size() * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));


      CudaSpectrumGeneralKernel<<<grid_size, block_size>>>(i, j, num_w_points, qfactors.size(), padded_size_, dev_qfactors, dev_spin_data.data(), dev_SQw.data());
    }

    if (i%100 == 0) {
      dev_SQw.copy_to_host_array(SQw);
      std::ofstream cfile(seedname + "_corr.tsv");
      for (unsigned q = 0; q < qvecs.size(); ++q) {
        for (unsigned w = 0; w < padded_size_/2+1; ++w) {
          cfile << qvecs[q][0] << " " << qvecs[q][1] << " " << qvecs[q][2] << " " << 0.5*w * freq_delta_ << " " << -SQw(q, w).y / (i + 1)/ static_cast<double>(padded_size_) << "\n";
        }
      }
      cfile.flush();
      cfile.close();
    }

  }

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "finish  " << get_date_string(end_time) << "\n\n";
  cout << "runtime " << duration_string(start_time, end_time) << "\n";
  cout.flush();

}
