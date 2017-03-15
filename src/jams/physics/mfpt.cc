// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/physics/mfpt.h"

#include <libconfig.h++>

#include <string>

#include "jams/core/globals.h"
#include "jams/core/output.h"
#include "jams/core/types.h"

MFPTPhysics::MFPTPhysics(const libconfig::Setting &settings)
  : Physics(settings),
  maskArray(),
  MFPTFile() {
  using namespace globals;

  output->write("  * MFPT physics module\n");

  std::string fileName = "_mfpt.dat";
  fileName = seedname+fileName;
  MFPTFile.open(fileName.c_str());

  maskArray.resize(num_spins);

  for (int i = 0; i < num_spins; ++i) {
    maskArray[i] = true;
  }
}

MFPTPhysics::~MFPTPhysics() {
  MFPTFile.close();
}

void MFPTPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
  if (iterations%output_step_freq_ == 0) {
    for (int i = 0; i < num_spins; ++i) {
      if (s(i, 2) < 0.0 && maskArray[i] == true) {
        MFPTFile << time << "\n";
        maskArray[i] = false;
      }
    }
  }
}
