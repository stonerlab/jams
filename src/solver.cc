#include "solver.h"
#include "heunllg.h"
#include "heunllms.h"
#include "semillg.h"
#include "cuda_semillg.h"
#include "cuda_heunllg.h"
#include "fftnoise.h"
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
  
  t_step = idt;
  dt = idt*gamma_electron_si;

  initialised = true;
}

void Solver::run()
{

}

void Solver::syncOutput()
{

}

Solver* Solver::Create()
{
  // default solver type
  return Solver::Create(HEUNLLG);
}

Solver* Solver::Create(SolverType type)
{
  switch (type) {
    case HEUNLLG:
      return new HeunLLGSolver;
      break;
    case HEUNLLMS:
      return new HeunLLMSSolver;
      break;
    case SEMILLG:
      return new SemiLLGSolver;
      break;
#ifdef CUDA
    case CUDASEMILLG:
      return new CUDASemiLLGSolver;
      break;
    case CUDAHEUNLLG:
      return new CUDAHeunLLGSolver;
      break;
#endif
    case FFTNOISE:
      return new FFTNoise;
      break;
    default:
      jams_error("Unknown solver selected.");
  }
  return NULL;
}
