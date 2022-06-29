// collective_variable_factory.cc                                      -*-C++-*-

#include "jams/metadynamics/collective_variable_factory.h"

#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/metadynamics/cvars/cvar_topological_charge.h>
#include <jams/metadynamics/cvars/cvar_topological_charge_finite_diff.h>
#include <jams/metadynamics/cvars/cvar_skyrmion_center_coordinate.h>

#include <stdexcept>

#define DEFINED_METADYNAMICS_CVAR(name, type, settings) \
{ \
  if (lowercase(settings["name"]) == name) { \
    return new type(settings); \
  } \
}

jams::CollectiveVariable *
jams::CollectiveVariableFactory::create(const libconfig::Setting &settings) {

  // New CollectiveVariablePotential derived classes should be added here
  // and the header included above.
  DEFINED_METADYNAMICS_CVAR("magnetisation", CVarMagnetisation, settings);
  DEFINED_METADYNAMICS_CVAR("topological_charge", CVarTopologicalCharge, settings);
  DEFINED_METADYNAMICS_CVAR("topological_charge_finite_diff", CVarTopologicalChargeFiniteDiff, settings);
  DEFINED_METADYNAMICS_CVAR("skyrmion_core_center_coordinate",CVarSkyrmionCoreCoordinate,settings);

  throw std::runtime_error("unknown metadynamics collective variable: " + std::string(settings["collective_variable"].c_str()));
}

#undef DEFINED_METADYNAMICS_CVAR