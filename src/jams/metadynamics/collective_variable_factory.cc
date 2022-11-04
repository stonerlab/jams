// collective_variable_factory.cc                                      -*-C++-*-

#include "jams/metadynamics/collective_variable_factory.h"

#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#include <jams/metadynamics/cvars/cvar_magnetisation.h>
#include <jams/metadynamics/cvars/cvar_topological_charge.h>
#include <jams/metadynamics/cvars/cvar_reduced_mz.h>
#include <jams/metadynamics/cvars/cvar_topological_charge_finite_diff.h>
#include <jams/metadynamics/cvars/cvar_skyrmion_center_coordinate.h>

#ifdef HAS_CUDA
#include <jams/metadynamics/cvars/cvar_magnetisation_cuda.h>
#include <jams/metadynamics/cvars/cvar_reduced_mz_cuda.h>
#endif


#include <stdexcept>

#define DEFINED_METADYNAMICS_CVAR(name, type, settings) \
{ \
  if (lowercase(settings["name"]) == name) { \
    return new type(settings); \
  } \
}

#ifdef HAS_CUDA
#define CUDA_CVAR_VARIANT_NAME(type) type##Cuda
#define DEFINED_METADYNAMICS_CVAR_CUDA_VARIANT(name, type, is_cuda_solver, settings) \
  { \
    if (lowercase(settings["name"]) == name) { \
      if(is_cuda_solver) { \
        return new CUDA_CVAR_VARIANT_NAME(type)(settings); \
      } \
      return new type(settings); \
    } \
  }
#else
#define DEFINED_METADYNAMICS_CVAR_CUDA_VARIANT(name, type, is_cuda_solver, settings) \
  DEFINED_METADYNAMICS_CVAR(name, type, settings)
#endif

jams::CollectiveVariable *
jams::CollectiveVariableFactory::create(const libconfig::Setting &settings,
                                        bool is_cuda_solver) {

  // New CollectiveVariablePotential derived classes should be added here
  // and the header included above.

  DEFINED_METADYNAMICS_CVAR("topological_charge", CVarTopologicalCharge, settings);

  DEFINED_METADYNAMICS_CVAR("topological_charge_finite_diff",
                            CVarTopologicalChargeFiniteDiff, settings);

  DEFINED_METADYNAMICS_CVAR("skyrmion_core_center_coordinate",
                            CVarSkyrmionCoreCoordinate,settings);

  DEFINED_METADYNAMICS_CVAR_CUDA_VARIANT("reduced_mz", CVarReducedMz, is_cuda_solver, settings);
  DEFINED_METADYNAMICS_CVAR_CUDA_VARIANT("magnetisation", CVarMagnetisation, is_cuda_solver, settings);


  throw std::runtime_error("unknown metadynamics collective variable: "
  + std::string(settings["collective_variable"].c_str()));

}

#undef DEFINED_METADYNAMICS_CVAR