// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/empty.h"

#include <libconfig.h++>

void EmptyPhysics::initialize(libconfig::Setting &phys) {
}

EmptyPhysics::~EmptyPhysics() {
}

void EmptyPhysics::run(const double realtime, const double dt) {
}

void EmptyPhysics::monitor(const double realtime, const double dt) {
}
