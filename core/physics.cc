// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/physics.h"

#include <libconfig.h++>

#include "core/globals.h"

#include "physics/dynamicsf.h"
#include "physics/empty.h"
#include "physics/fieldcool.h"
#include "physics/fmr.h"
#include "physics/mfpt.h"
#include "physics/spinwaves.h"
#include "physics/square.h"
#include "physics/ttm.h"


void Physics::initialize(libconfig::Setting &phys) {
  if (initialized == true) {
    jams_error("Physics module is already initialized");
  }

  initialized = true;
}

void Physics::run(double realtime, const double dt) {
}

void Physics::monitor(const double realtime, const double dt) {
}

Physics* Physics::Create(PhysicsType type) {
  switch (type) {
    case FMR:
      return new FMRPhysics;
      break;
    case MFPT:
      return new MFPTPhysics;
      break;
    case TTM:
      return new TTMPhysics;
      break;
    case SPINWAVES:
      return new SpinwavesPhysics;
      break;
    case SQUARE:
      return new SquarePhysics;
      break;
    case DYNAMICSF:
      return new DynamicSFPhysics;
      break;
    case FIELDCOOL:
      return new FieldCoolPhysics;
      break;
    case EMPTY:
      return new EmptyPhysics;
      break;
    default:
      jams_error("Unknown physics selected.");
  }
  return NULL;
}
