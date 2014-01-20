// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/mfpt.h"

#include <libconfig.h++>

#include <cmath>
#include <string>

#include "core/globals.h"

void MFPTPhysics::initialize(libconfig::Setting &phys) {
  using namespace globals;

  output.write("  * MFPT physics module\n");

  std::string fileName = "_mfpt.dat";
  fileName = seedname+fileName;
  MFPTFile.open(fileName.c_str());

  maskArray.resize(num_spins);

  for (int i = 0; i < num_spins; ++i) {
    maskArray[i] = true;
  }

  initialized = true;
}

MFPTPhysics::~MFPTPhysics() {
  MFPTFile.close();
}

void MFPTPhysics::run(const double realtime, const double dt) {
}

void MFPTPhysics::monitor(const double realtime, const double dt) {
  using namespace globals;

  for (int i = 0; i < num_spins; ++i) {
    if (s(i, 2) < 0.0 && maskArray[i] == true) {
      MFPTFile << realtime << "\n";
      maskArray[i] = false;
    }
  }
}
