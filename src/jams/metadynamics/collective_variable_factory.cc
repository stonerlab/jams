// collective_variable_factory.cc                                      -*-C++-*-

#include "jams/metadynamics/collective_variable_factory.h"

#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#include "jams/metadynamics/magnetisation_cv.h"
#include "jams/metadynamics/mz_orthogonal_mz_cv.h"
#include "jams/metadynamics/skyrmion_center_cv.h"

#include <stdexcept>

#define DEFINED_METADYNAMICS_CV(name, type, settings) \
{ \
  if (lowercase(settings["collective_variable"]) == name) { \
    return new type(settings); \
  } \
}

jams::CollectiveVariablePotential *
jams::CollectiveVariableFactory::create(const libconfig::Setting &settings) {

  // New CollectiveVariablePotential derived classes should be added here
  // and the header included above.
  DEFINED_METADYNAMICS_CV("magnetisation", MagnetisationCollectiveVariable, settings);
  DEFINED_METADYNAMICS_CV("mz_orthogonal_mz", MzOrthogonalMzCV,settings);
  DEFINED_METADYNAMICS_CV("skyrmion_center_of_mass", SkyrmionCenterCV, settings)

  throw std::runtime_error("unknown metadynamics collective variable: " + std::string(settings["collective_variable"].c_str()));
}

#undef DEFINED_METADYNAMICS_CV