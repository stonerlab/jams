#include "globals.h"
#include "energy.h"
#include "lattice.h"

#include <cmath>

void EnergyMonitor::initialise() {
  using namespace globals;
  output.write("\nInitialising Energy monitor...\n");

  std::string name = "_eng.dat";
  name = seedname+name;
  outfile.open(name.c_str());

  outfile << "# time (s) | e_tot | e1_s | e1_t | e2_s | e2_t | e4_s "<<std::endl;

  initialised = true;
}

void EnergyMonitor::run() {

}

void EnergyMonitor::initConvergence(ConvergenceType type, const double meanTolerance, const double devTolerance){
}

bool EnergyMonitor::checkConvergence(){
	return true;
}
void EnergyMonitor::write(Solver *solver) {
  using namespace globals;
  assert(initialised);

    double e1_s=0.0, e1_t=0.0, e2_s=0.0, e2_t=0.0, e4_s=0.0;

    solver->calcEnergy(e1_s,e1_t,e2_s,e2_t,e4_s);
  
    outfile << solver->getTime();

      outfile << "\t" << e1_s+e1_t+e2_s+e2_t+e4_s;

      outfile << "\t" << e1_s;
      outfile << "\t" << e1_t;
      outfile << "\t" << e2_s;
      outfile << "\t" << e2_t;
      outfile << "\t" << e4_s;

#ifdef NDEBUG
  outfile << "\n";
#else
  outfile << std::endl;
#endif
}

EnergyMonitor::~EnergyMonitor() {
  outfile.close();
}
