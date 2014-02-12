// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

void MagnetisationMonitor::initialize() {
  using namespace globals;
  output.write("\nInitialising Magnetisation monitor...\n");

  std::string name = "_mag.dat";
  name = seedname+name;
  outfile.open(name.c_str());

  // mx my mz |m|
  mag.resize(lattice.numTypes(), 4);


  old_avg = 0.0;

  initialized = true;
}

void MagnetisationMonitor::run() {
}

void MagnetisationMonitor::initialize_convergence(ConvergenceType type,
  const double meanTolerance, const double devTolerance) {
  convType = type;
  meanTol = meanTolerance;
  devTol = devTolerance;
}

bool MagnetisationMonitor::has_converged() {
  if (convType == convNone) {
    return false;
  } else {
    if (runningMean.numDataValues() > 0) {
      const double meanRelErr = fabs((runningMean.mean()-blockStats.mean())
        /runningMean.mean());
      const double devRelErr = fabs((runningMean.stdDev()-blockStats.stdDev())
        /runningMean.stdDev());
      output.write("Convergence: r_mean %1.5f b_mean %1.5f meanRelErr %1.5f [%1.5f] :: r_stddev %1.5f b_stddev %1.5f devRelErr %1.5f [%1.5f] \n",
        runningMean.mean(), blockStats.mean(), meanRelErr, meanTol,
        runningMean.stdDev(), blockStats.stdDev(), devRelErr, devTol);

      if ((meanRelErr < meanTol) && (devRelErr < devTol)) {
        output.write("Converged: mean %1.5f meanRelErr %1.5f stdDev %1.5f stdDevRelErr %1.5f\n", blockStats.mean(), meanRelErr,
          blockStats.stdDev(), devRelErr);
        return true;
      }
    }
    runningMean = blockStats;
  }
  return false;
}

void MagnetisationMonitor::write(Solver *solver) {
  using namespace globals;
  assert(initialized);
  int i, j, type;

  for (i = 0; i < lattice.numTypes(); ++i) {
    for (j = 0; j < 4; ++j) {
      mag(i, j) = 0.0;
    }
  }

  for (i = 0; i < num_spins; ++i) {
    type = lattice.getType(i);
    for (j = 0; j < 3; ++j) {
      mag(type, j) += s(i, j);
    }
  }


  for (i = 0; i < lattice.numTypes(); ++i) {
    for (j = 0; j < 3; ++j) {
      mag(i, j) = mag(i, j)/static_cast<double>(lattice.getTypeCount(i));
    }
  }

  for (i = 0; i < lattice.numTypes(); ++i) {
    mag(i, 3) = sqrt(mag(i, 0)*mag(i, 0) + mag(i, 1)*mag(i, 1)
      + mag(i, 2)*mag(i, 2));
  }


  outfile << solver->time();

  outfile << "\t" << globalTemperature;

  for (i = 0; i < 3; ++i) {
    outfile << "\t" << h_app[i];
  }

  for (i = 0; i < lattice.numTypes(); ++i) {
    outfile <<"\t"<< mag(i, 0) <<"\t"<< mag(i, 1) <<"\t"<< mag(i, 2)
    <<"\t"<< mag(i, 3);
  }
#ifdef NDEBUG
  outfile << "\n";
#else
  outfile << std::endl;
#endif

  switch (convType) {
  case convNone:
    break;
  case convMag:
    blockStats.push(mag(0, 3));
    break;
  case convPhi:
    blockStats.push(acos(mag(0, 2)/mag(0, 3)));
    break;
  case convSinPhi:
    blockStats.push(sin(acos(mag(0, 2)/mag(0, 3))));
    break;
  default:
    break;
  }
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
