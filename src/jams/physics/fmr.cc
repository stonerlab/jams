// Copyright 2014 Joseph Barker. All rights reserved.

#include "fmr.h"


#include <libconfig.h++>

#include <cmath>
#include <string>

#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/output.h"

FMRPhysics::FMRPhysics(const libconfig::Setting &settings)
: Physics(settings),
  ACFieldFrequency(0),
  ACFieldStrength(3, 0),
  DCFieldStrength(3, 0),
  PSDFile(jams::output::open_file(simulation_name + "_psd.tsv")),
  PSDIntegral(0) {
  using namespace globals;

  ACFieldFrequency = settings["ACFieldFrequency"];
  ACFieldFrequency = 2.0*M_PI*ACFieldFrequency;

  for (int i = 0; i < 3; ++i) {
    ACFieldStrength[i] = settings["ACFieldStrength"][i];
  }

  for (int i = 0; i < 3; ++i) {
    DCFieldStrength[i] = settings["DCFieldStrength"][i];
  }

  PSDIntegral.resize(num_spins);

  for (int i = 0; i < num_spins; ++i) {
    PSDIntegral(i) = 0;
  }

  initialized = true;
}

FMRPhysics::~FMRPhysics() {
  PSDFile.close();
}

void FMRPhysics::update(const int &iterations, const double &time, const double &dt) {
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

  if (iterations%output_step_freq_ == 0) {
    double pAverage = 0.0;

    for (int i = 0; i < num_spins; ++i) {
      pAverage += (PSDIntegral(i)*(ACFieldFrequency*mus(i)*kBohrMagneton)/time);
    }

    PSDFile << time << "\t" << pAverage/num_spins << "\n";
  }
}
