#include "globals.h"
#include "magnetisation.h"
#include "lattice.h"

#include <cmath>

void MagnetisationMonitor::initialise() {
  using namespace globals;
  output.write("Initialising Magnetisation monitor\n");

  std::string name = "_mag.dat";
  name = seedname+name;
  outfile.open(name.c_str());

  mag.resize(lattice.numTypes(),4);

  initialised = true;
}

void MagnetisationMonitor::run() {

}

void MagnetisationMonitor::write(const double &time) {
  using namespace globals;
  assert(initialised);
  int i,j,type;
  
  for(i=0; i<lattice.numTypes(); ++i) {
    for(j=0; j<4; ++j) {
      mag(i,j) = 0.0;
    }
  }

  for(i=0; i<nspins; ++i) {
    type = lattice.getType(i);
    for(j=0;j<3;++j) {
      mag(type,j) += s(i,j);
    }
  }

  for(i=0; i<lattice.numTypes(); ++i) {
    for(j=0; j<3; ++j) {
      mag(i,j) = mag(i,j)/static_cast<double>(lattice.getTypeCount(i));
    }
  }

  for(i=0; i<lattice.numTypes(); ++i) {
    mag(i,3) = sqrt(mag(i,0)*mag(i,0) + mag(i,1)*mag(i,1) + mag(i,2)*mag(i,2));
  }

  outfile << time;
  for(i=0; i<lattice.numTypes(); ++i) {
    outfile <<"\t"<< mag(i,0) <<"\t"<< mag(i,1) <<"\t"<< mag(i,2) <<"\t"<< mag(i,3);
  }
  outfile << "\n";

}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
