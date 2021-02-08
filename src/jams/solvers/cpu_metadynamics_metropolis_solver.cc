//
// Created by ioannis charalampidis on 07/11/2020.
//

#include "cpu_metadynamics_metropolis_solver.h"
#include "jams/core/solver.h"

#include <fstream>
#include <jams/core/types.h>
#include <pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"
#include <vector>
#include <jams/core/globals.h>
#include <jams/helpers/montecarlo.h>
#include <iostream>
#include "jams/helpers/output.h"
#include <jams/maths/interpolation.h>
#include <jams/metadynamics/collective_variable_factory.h>

void MetadynamicsMetropolisSolver::initialize(const libconfig::Setting &settings) {
  // pass the settings through to the base class which will do the monte carlo
  // solving
  MetropolisMCSolver::initialize(settings);

  // create a collective variable object and keep the reference inside this class
  cv_potential_.reset(jams::CollectiveVariableFactory::create(settings));

  // ---------------------------------------------------------------------------
  // config settings
  // ---------------------------------------------------------------------------

  // number of monte carlo steps between gaussian depositions in metadynamics
  gaussian_deposition_stride_ = jams::config_optional<int>(settings,"gaussian_deposition_stride",200);

  // toggle tempered metadynamics on or off
  do_tempering_ = jams::config_optional<bool>(settings,"tempering", false);

  // the bias temperature for tempered metadynamics
  tempering_bias_temperature_ = jams::config_optional<double>(settings,"bias_temperature",0.0);

  std::cout << "  gaussian deposition stride: " << gaussian_deposition_stride_ << "\n";

  if (do_tempering_) {
    std::cout << "  tempered metadynamics: \n";
    std::cout << "  bias temperature (K): " << tempering_bias_temperature_ << "\n";
  }
}

void MetadynamicsMetropolisSolver::run() {
  // run the monte carlo solver
  MetropolisMCSolver::run();

  // deposit a gaussian if needed
  if (iteration_ % gaussian_deposition_stride_ == 0) {
    // Only change the relative amplitude of the gaussian if we are using
    // tempered metadynamics.
    double relative_amplitude = do_tempering_ ? tempering_amplitude() : 1.0;

    if (!metadynamics_stats.is_open()) {
      metadynamics_stats.open(jams::output::full_path_filename("metad_stats.tsv"));
    }

    metadynamics_stats << iteration() << " " << tempering_amplitude() << "\n";

    cv_potential_->insert_gaussian(relative_amplitude);
    cv_potential_->output();
  }
}

  double MetadynamicsMetropolisSolver::energy_difference(const int spin_index,
														 const Vec3 &initial_Spin,
														 const Vec3 &final_Spin) {

  // re-define the energy difference for the monte carlo solver
  // so that it uses the metadynamics potential
	return MetropolisMCSolver::energy_difference(spin_index, initial_Spin, final_Spin)
		+ cv_potential_->potential_difference(spin_index, initial_Spin, final_Spin);
  }

  void MetadynamicsMetropolisSolver::accept_move(const int spin_index,
												 const Vec3 &initial_spin,
												 const Vec3 &final_spin) {
	MetropolisMCSolver::accept_move(spin_index, initial_spin, final_spin);

	// As well as updating the monte carlo solver we update the collective variable
	// which can often avoid some expensive recalculations if we tell it which
	// spin has changed and what the new value is
	cv_potential_->spin_update(spin_index, initial_spin, final_spin);
  }

double MetadynamicsMetropolisSolver::tempering_amplitude() {
  return exp(-cv_potential_->current_potential()*kBohrMagneton / (tempering_bias_temperature_ * kBoltzmann));
}

