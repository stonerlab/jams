#include "globals.h"
#include "boltzmann.h"
#include "maths.h"

void BoltzmannMonitor::initialise() {
  output.write("Initialising Boltzmann monitor\n");

  outfile.open("boltzmann.dat");

  bins.resize(36);
  for(int i=0;i<36;++i){
    bins(i) = 0.0;
  }
  initialised = true;
}

void BoltzmannMonitor::run() {
  using namespace globals;

  double angle;
  unsigned int round;
  for(int i=0; i<nspins; ++i) {
    angle = rad_to_deg(acos(s(i,2)));
    round = static_cast<unsigned int>(angle*0.2);
    bins(round)++;
  }
}

void BoltzmannMonitor::write() {
  for(int i=0;i<36;++i) {
    outfile << i*5+2.5 << "\t" << bins(i) << "\n";
  }
  outfile << "\n" << std::endl;
}

BoltzmannMonitor::~BoltzmannMonitor() {
  outfile.close();
}
