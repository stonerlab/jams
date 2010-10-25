#include "solver.h"
#include "heunllg.h"
#include "globals.h"
#include "consts.h"

void Solver::initialise(int argc, char **argv, double idt) {

  if(initialised == true) {
    jams_error("Solver is already initialised");
  }

  output.write("Initialising solver\n");

  // initialise time and iterations to 0
  time = 0.0;
  iteration = 0;
  
  dt = idt*gamma_electron_si;

  initialised = true;
}

void Solver::run()
{

}

Solver* Solver::Create()
{
  // default solver type
  return Solver::Create(HEUNLLG);
}

Solver* Solver::Create(SolverType type)
{
  if( type == HEUNLLG ) {
    return new HeunLLGSolver;
  }
}
