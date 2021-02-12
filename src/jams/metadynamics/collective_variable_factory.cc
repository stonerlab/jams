#include <jams/metadynamics/collective_variable_factory.h>
#include <jams/metadynamics/magnetisation_cv.h>
#include <jams/metadynamics/mz_orthogonal_mz_cv.h>

#define DEFINED_METADYNAMICS_CV(name, type, settings) \
{ \
  if (lowercase(settings["collective_variable"]) == name) { \
    return new type(settings); \
  } \
}

jams::CollectiveVariablePotential *
jams::CollectiveVariableFactory::create(const libconfig::Setting &settings) {
  DEFINED_METADYNAMICS_CV("magnetisation", MagnetisationCollectiveVariable, settings);
  DEFINED_METADYNAMICS_CV("mz_orthogonal_mz",MzOrthogonalMzCV,settings);

  throw std::runtime_error("unknown metadynamics collective variable: " + std::string(settings["collective_variable"].c_str()));
}