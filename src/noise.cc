#include "noise.h"
#include "globals.h"
#include "whitenoise.h"

void Noise::initialise(double dt) {
  if(initialised == true) {
    jams_error("Noise is already initialised");
  }
}

void Noise::run() {

}

Noise* Noise::Create() {
  // default noise type
  return Noise::Create(WHITE);
}

Noise* Noise::Create(NoiseType type) {
  switch (type) {
    case WHITE:
      return new WhiteNoise;
      break;
    default:
      jams_error("Unknown noise selected.");
  }
}
