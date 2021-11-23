// collective_variable_factory.h                                       -*-C++-*-

#ifndef INCLUDED_JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_FACTORY
#define INCLUDED_JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_FACTORY

#include "jams/metadynamics/metadynamics_potential.h"
#include "jams/interface/config.h"

namespace jams {

    // ===============================
    // class CollectiveVariableFactory
    // ===============================

    /// Factory class for producing CollectiveVariablePotential objects
    class CollectiveVariableFactory {
    public:
        static jams::CollectiveVariable *
        create(const libconfig::Setting &settings,
               bool is_cuda_solver);
    };
}
#endif //INCLUDED_JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_FACTORY
