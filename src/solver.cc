#include "solver.h"
#include "heunllg.h"
#include "cuda_heunllg.h"
#include "cuda_heunllms.h"
#include "cuda_heunllbp.h"
#include "cuda_srk4llg.h"
#include "metropolismc.h"
#include "globals.h"
#include "consts.h"

void Solver::initialise(int argc, char **argv, double idt) {

  if(initialised == true) {
    jams_error("Solver is already initialised");
  }

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
#ifdef CUDA
    case CUDAHEUNLLG:
      return new CUDAHeunLLGSolver;
      break;
    case CUDASRK4LLG:
      return new CUDALLGSolverSRK4;
      break;
    case CUDAHEUNLLMS:
      return new CUDAHeunLLMSSolver;
      break;
    case CUDAHEUNLLBP:
      return new CUDAHeunLLBPSolver;
      break;
#endif
    case METROPOLISMC:
      return new MetropolisMCSolver;
      break;
    default:
      jams_error("Unknown solver selected.");
  }
  return NULL;
}
