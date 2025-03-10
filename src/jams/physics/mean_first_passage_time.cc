// Copyright 2014 Joseph Barker. All rights reserved.

#include "mean_first_passage_time.h"

#include <libconfig.h++>

#include <string>

#include "jams/core/globals.h"
#include "jams/core/types.h"
#include "jams/helpers/output.h"

MFPTPhysics::MFPTPhysics(const libconfig::Setting &settings)
  : Physics(settings),
  maskArray(),
  MFPTFile(jams::output::full_path_filename("mfpt.tsv")) {
  maskArray.resize(globals::num_spins);

  for (int i = 0; i < globals::num_spins; ++i) {
    maskArray[i] = true;
  }
}

MFPTPhysics::~MFPTPhysics() {
  MFPTFile.close();
}

void MFPTPhysics::update(const int &iterations, const double &time, const double &dt) {
  if (iterations%output_step_freq_ == 0) {
    for (int i = 0; i < globals::num_spins; ++i) {
      if (globals::s(i, 2) < 0.0 && maskArray[i] == true) {
        MFPTFile << time << "\n";
        maskArray[i] = false;
      }
    }
  }
}
