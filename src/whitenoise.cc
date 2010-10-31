#include "whitenoise.h"
#include "globals.h"
#include "consts.h"
#include <cmath>

void WhiteNoise::initialise(double dt) {
  using namespace globals;

  Noise::initialise(dt);

  output.write("Initialising white noise\n");

  sigma.resize(nspins,3);

  temperature = 300.0;

  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sigma(i,j) = sqrt( (2.0*boltzmann_si*alpha(i)*mus(i)) / (dt) );
    }
  }
}

void WhiteNoise::run() {
  using namespace globals;

  assert(initialised);


  // toggle for half step
  if(half == false) {
    if(temperature > 0.0) {

      int i,j;

      const double stmp = sqrt(temperature);
      for(i=0; i<nspins; ++i) {
        for(j=0; j<3; ++j) {
          w(i,j) = (rng.normal())*sigma(i,j)*stmp;
        }
      }
    } 
    half = true;
  } else {
    half = false;
  }
}

