#include "monitors/boltzmann.h"

#include <string>

#include "core/globals.h"
#include "core/maths.h"

void BoltzmannMonitor::initialise() {
  output.write("Initialising Boltzmann monitor\n");

  std::string name = "_blt.dat";
  name = seedname+name;
  outfile.open(name.c_str());
  
  bins.resize(36);
  for(int i=0;i<36;++i){
    bins(i) = 0.0;
  }
  initialised = true;
}

void BoltzmannMonitor::run() {



}

void BoltzmannMonitor::write(Solver *solver) {
    using namespace globals;
    double angle;
    unsigned int round;
    for(int i=0; i<nspins; ++i) {
        angle = rad_to_deg(acos(s(i,2)));
        round = static_cast<unsigned int>(angle*0.2);
        bins(round)++;
        total++;
    }

    if(total > 0.0){
        for(int i=0;i<36;++i) {
            outfile << i*5+2.5 << "\t" << bins(i)/total << "\n";
        }
        outfile << "\n" << std::endl;  
    }
}

BoltzmannMonitor::~BoltzmannMonitor() {
  outfile.close();
}
