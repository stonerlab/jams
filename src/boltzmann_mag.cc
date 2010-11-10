#include "globals.h"
#include "boltzmann_mag.h"
#include "maths.h"

void BoltzmannMagMonitor::initialise() {
  output.write("Initialising Boltzmann Mag monitor\n");

  outfile.open("boltzmann_mag.dat");

  bins.resize(101);
  for(int i=0;i<101;++i){
    bins(i) = 0.0;
  }
  initialised = true;
}

void BoltzmannMagMonitor::run() {
  using namespace globals;

  double mag[3]={0.0,0.0,0.0};
  unsigned int round;
  for(int i=0; i<nspins; ++i) {
    for(int j=0;j<3;++j) {
      mag[j] += s(i,j);
    }
  }

  for(int j=0;j<3;++j) {
    mag[j] = mag[j]/static_cast<double>(nspins); 
  }
  double modmag = sqrt(mag[0]*mag[0]+mag[1]*mag[1]+mag[2]*mag[2]);
        
  round = nint(modmag*100);
  bins(round)++;
  total++;
}

void BoltzmannMagMonitor::write() {
  for(int i=0;i<101;++i) {
    outfile << i*0.01+0.005 << "\t" << bins(i)/total << "\n";
  }
  outfile << "\n" << std::endl;
}

BoltzmannMagMonitor::~BoltzmannMagMonitor() {
  outfile.close();
}
