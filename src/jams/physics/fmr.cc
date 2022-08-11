// Copyright 2014 Joseph Barker. All rights reserved.

#include "fmr.h"

#include <libconfig.h++>

#include <cmath>
#include <string>

#include "jams/core/hamiltonian.h"
#include "jams/core/solver.h"

#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/output.h"
#include "jams/helpers/error.h"
#include "jams/core/lattice.h"

FMRPhysics::FMRPhysics(const libconfig::Setting &settings)
: Physics(settings),
  ACFieldFrequency(0),
  ACFieldStrength(3, 0),
  DCFieldStrength(3, 0),
  PSDFile(jams::output::full_path_filename("psd.tsv")),
  PSDIntegral(0) {
  using namespace globals;
    if(settings.exists("dc_local_field")) {
        if (settings["dc_local_field"].getLength() != lattice->num_materials()) {
            jams_die("FMR: dc_local_field must be specified for every material");
        }

        for (int i = 0; i < 3; ++i) {
            DCFieldStrength[i] = settings["dc_local_field"][0][i];
            ACFieldStrength[i] = settings["ac_local_field"][0][i];
        }
    }

    ACFieldFrequency = settings["ac_local_frequency"][0];
    ACFieldFrequency = kTwoPi*ACFieldFrequency;

    for (int i = 0; i < 3; ++i) {
        ACFieldStrength[i] = settings["ac_local_field"][0][i];
    }

    // Now all the fields have been read in, need to initialise a ZeemanHamiltonian class

    solver->register_hamiltonian(
            Hamiltonian::create(settings, num_spins, solver->is_cuda_solver()));

    PSDIntegral.resize(num_spins);

  for (int i = 0; i < num_spins; ++i) {
    PSDIntegral(i) = 0;
  }

  initialized = true;

  PSDFile.setf(std::ios::right);
  PSDFile << "time (ps)\t" << "PSD_Av\n";
}

FMRPhysics::~FMRPhysics() {
  PSDFile.close();
}

void FMRPhysics::update(const int &iteration, const double &time, const double &dt) {
  using namespace globals;

  const double cosValue = cos(ACFieldFrequency*time);
  const double sinValue = sin(ACFieldFrequency*time);

  for (int i = 0; i < 3; ++i) {
    applied_field_[i] = DCFieldStrength[i]
    + ACFieldStrength[i] * cosValue;
  }

  for (int i = 0; i < num_spins; ++i) {
    const double sProjection =
    s(i, 0)*ACFieldStrength[0] + s(i, 1)*ACFieldStrength[1]
    + s(i, 2)*ACFieldStrength[2];

    PSDIntegral(i) += sProjection * sinValue * dt;
  }


  if (iteration % output_step_freq_ == 0) {


    double pAverage = 0.0;

    for (int i = 0; i < num_spins; ++i) {
      pAverage += (PSDIntegral(i)*(ACFieldFrequency*mus(i))/time);
    }

    PSDFile.width(12);

    PSDFile << std::scientific << std::setprecision(15) << time << "\t" << pAverage/num_spins << std::endl;
  }
}
