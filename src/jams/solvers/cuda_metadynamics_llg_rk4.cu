// cuda_metadynamics_llg_rk4.cc                                        -*-C++-*-
#include <jams/solvers/cuda_metadynamics_llg_rk4.h>
#include "jams/helpers/output.h"
#include "jams/core/thermostat.h"

#include "cuda_metadynamics_llg_rk4_kernel.cuh"


void CUDAMetadynamicsLLGRK4Solver::initialize(const libconfig::Setting &settings) {
  using namespace globals;

  kernel_choice_ = jams::config_required<int>(settings, "kernel_choice");

  if (kernel_choice_ < 1 || kernel_choice_ > 3) {
    throw std::runtime_error("llg-metadynamics-rk4-gpu kernel choice must be 1, 2 or 3");
  }

  // convert input in seconds to picoseconds for internal units
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;


  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  std::cout << "timestep " << step_size_ << "\n";
  std::cout << "t_max " << t_max << " steps (" <<  max_steps_ << ")\n";
  std::cout << "t_min " << t_min << " steps (" << min_steps_ << ")\n";

  std::string thermostat_name = jams::config_optional<std::string>(config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  std::cout << "done\n";

  s_old_.resize(num_spins, 3);
  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_old_(i, j) = s(i, j);
    }
  }

  k1_.resize(num_spins, 3);
  k2_.resize(num_spins, 3);
  k3_.resize(num_spins, 3);
  k4_.resize(num_spins, 3);
  // Set the pointer to the collective variables attached to the solver
  metad_potential_.reset(new jams::MetadynamicsPotential(settings));

  // ---------------------------------------------------------------------------
  // Read settings
  // ---------------------------------------------------------------------------

  // Read the number of monte carlo steps between gaussian depositions in metadynamics
  gaussian_deposition_stride_ = jams::config_required<int>(settings,"gaussian_deposition_stride");
  gaussian_deposition_delay_ = jams::config_optional<int>(settings,"gaussian_deposition_delay", 0);

  output_steps_ = jams::config_optional<int>(settings, "output_steps", gaussian_deposition_stride_);

  metadynamics_potential_file_.open(jams::output::full_path_filename("metad_potential.tsv"));

  // Toggle tempered metadynamics on or off
  do_tempering_ = jams::config_optional<bool>(settings,"tempering", false);

  if (do_tempering_) {
    // Read the bias temperature for tempered metadynamics
    tempering_bias_temperature_ = jams::config_required<double>(settings,"tempering_bias_temperature");
  }

  // ---------------------------------------------------------------------------

  std::cout << "  gaussian deposition stride: " << gaussian_deposition_stride_ << "\n";
  std::cout << "  gaussian deposition delay : " << gaussian_deposition_delay_ << "\n";

  std::cout << "  tempered metadynamics: " << std::boolalpha << do_tempering_ << "\n";
  if (do_tempering_) {
    std::cout << "  bias temperature (K): " << tempering_bias_temperature_ << "\n";
  }
}


void CUDAMetadynamicsLLGRK4Solver::run() {
  using namespace globals;

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cudaMemcpyAsync(s_old_.device_data(),           // void *               dst
                  s.device_data(),               // const void *         src
                  num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();

  compute_fields();

  // k1
  switch (kernel_choice_) {
    case 1:
      cuda_metadynamics_llg_rk4_kernel1<<<grid_size, block_size>>>
          (s.device_data(), k1_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 2:
      cuda_metadynamics_llg_rk4_kernel2<<<grid_size, block_size>>>
          (s.device_data(), k1_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 3:
      cuda_metadynamics_llg_rk4_kernel3<<<grid_size, block_size>>>
          (s.device_data(), k1_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  double mid_time_step = 0.5 * step_size_;
  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k1_.device_data(), 1, s.device_data(), 1));

  compute_fields();

  // k2
  switch (kernel_choice_) {
    case 1:
      cuda_metadynamics_llg_rk4_kernel1<<<grid_size, block_size>>>
          (s.device_data(), k2_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 2:
      cuda_metadynamics_llg_rk4_kernel2<<<grid_size, block_size>>>
          (s.device_data(), k2_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 3:
      cuda_metadynamics_llg_rk4_kernel3<<<grid_size, block_size>>>
          (s.device_data(), k2_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  mid_time_step = 0.5 * step_size_;
  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k2_.device_data(), 1, s.device_data(), 1));

  compute_fields();

  // k3
  switch (kernel_choice_) {
    case 1:
      cuda_metadynamics_llg_rk4_kernel1<<<grid_size, block_size>>>
          (s.device_data(), k3_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 2:
      cuda_metadynamics_llg_rk4_kernel2<<<grid_size, block_size>>>
          (s.device_data(), k3_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 3:
      cuda_metadynamics_llg_rk4_kernel3<<<grid_size, block_size>>>
          (s.device_data(), k3_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  mid_time_step = step_size_;
  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k3_.device_data(), 1, s.device_data(), 1));

  compute_fields();

  // k4
  switch (kernel_choice_) {
    case 1:
      cuda_metadynamics_llg_rk4_kernel1<<<grid_size, block_size>>>
          (s.device_data(), k4_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 2:
      cuda_metadynamics_llg_rk4_kernel2<<<grid_size, block_size>>>
          (s.device_data(), k4_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
    case 3:
      cuda_metadynamics_llg_rk4_kernel3<<<grid_size, block_size>>>
          (s.device_data(), k4_.device_data(),
           h.device_data(), metad_potential_->current_fields().device_data(), thermostat_->device_data(),
           gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
      break;
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_metadynamics_llg_rk4_combination_kernel<<<grid_size, block_size>>>
      (s.device_data(), s_old_.device_data(),
       k1_.device_data(), k2_.device_data(), k3_.device_data(), k4_.device_data(),
       step_size_, num_spins);

  iteration_++;


  // Don't do any of the metadynamics until we have passed the
  // gaussian_deposition_delay_
  if (iteration_ < gaussian_deposition_delay_) {
    return;
  }

  // Deposit a gaussian at the required interval
  if (iteration_ % gaussian_deposition_stride_ == 0) {
    double relative_amplitude = 1.0;

    // Set the relative amplitude of the gaussian if we are using tempering and
    // record the value in the stats file
    if (do_tempering_) {
      relative_amplitude = std::exp(-(metad_potential_->current_potential())
                               / (tempering_bias_temperature_ * kBoltzmannIU));

      jams::output::open_output_file_just_in_time(metadynamics_stats_file_, "metad_stats.tsv");

      metadynamics_stats_file_ << jams::fmt::sci << iteration() << " " << relative_amplitude << "\n";
    }

    // Insert the gaussian into the potential
    metad_potential_->insert_gaussian(relative_amplitude);
  }

  if (iteration_ % output_steps_ == 0) {
    metadynamics_stats_file_ << std::flush;

    double scale = 1.0;
    if (do_tempering_) {
      scale = (physics()->temperature() + tempering_bias_temperature_) / tempering_bias_temperature_;
    }
    metad_potential_->output(metadynamics_potential_file_, scale);
    metadynamics_potential_file_ << std::flush;
  }
}


