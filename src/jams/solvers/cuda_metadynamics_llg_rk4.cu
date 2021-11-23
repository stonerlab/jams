// cuda_metadynamics_llg_rk4.cc                                        -*-C++-*-
#include <jams/solvers/cuda_metadynamics_llg_rk4.h>
#include "jams/helpers/output.h"

void CUDAMetadynamicsLLGRK4Solver::initialize(const libconfig::Setting &settings) {
  CUDALLGRK4Solver::initialize(settings);

  // Set the pointer to the collective variables attached to the solver
  metad_potential_.reset(new jams::MetadynamicsPotential(settings));

  // ---------------------------------------------------------------------------
  // Read settings
  // ---------------------------------------------------------------------------

  // Read the number of monte carlo steps between gaussian depositions in metadynamics
  gaussian_deposition_stride_ = jams::config_required<int>(settings,"gaussian_deposition_stride");
  gaussian_deposition_delay_ = jams::config_optional<int>(settings,"gaussian_deposition_delay", 0);

  output_steps_ = jams::config_optional<int>(settings, "output_steps", gaussian_deposition_stride_);

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
  // run the dynamical solver
  CUDALLGRK4Solver::run();


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
      relative_amplitude = exp(-(metad_potential_->current_potential())
                               / (tempering_bias_temperature_ * kBoltzmannIU));

      jams::output::open_output_file_just_in_time(metadynamics_stats_file_, "metad_stats.tsv");

      metadynamics_stats_file_ << jams::fmt::sci << iteration() << " " << relative_amplitude << std::endl;
    }

    // Insert the gaussian into the potential
    metad_potential_->insert_gaussian(relative_amplitude);
  }

  if (iteration_ % output_steps_ == 0) {
    metad_potential_->output();
  }
}


void CUDAMetadynamicsLLGRK4Solver::compute_fields() {
  CudaSolver::compute_fields();

  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(),globals::h.elements(), &kOne, metad_potential_->current_fields().device_data(), 1, globals::h.device_data(), 1));
}

