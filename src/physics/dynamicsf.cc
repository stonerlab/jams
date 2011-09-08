#include "globals.h"
#include "dynamicsf.h"

#include <fftw3.h>

void DynamicSFPhysics::init(libconfig::Setting &phys)
{
  using namespace globals;

  qSpace = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*nspins));

  initialised = true;
}

DynamicSFPhysics::~DynamicSFPhysics()
{
  if(qSpace != NULL) {
    fftw_free(qSpace);
  }

}

void  DynamicSFPhysics::run(double realtime, const double dt)
{
  using namespace globals;
  assert(initialised);
}

void DynamicSFPhysics::monitor(double realtime, const double dt)
{
  using namespace globals;
  assert(initialised);

}
