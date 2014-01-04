#include "core/monitor.h"

#include "core/globals.h"
#include "core/solver.h"

void Monitor::initialise() {
  if(initialised == true) {
    jams_error("Monitor is already initialised");
  }

  output.write("Initialising monitor\n");
}

void Monitor::run() {

}

void Monitor::write(Solver *solver) {

}

void Monitor::initConvergence(ConvergenceType type, const double meanTol, const double devTol){
}

bool Monitor::checkConvergence(){
	return false;
}
