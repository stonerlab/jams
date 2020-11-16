#ifndef JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_FACTORY_H
#define JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_FACTORY_H

#include <jams/metadynamics/collective_variable_potential.h>
#include <jams/interface/config.h>

namespace jams {
    class CollectiveVariableFactory {
    public:
        static CollectiveVariablePotential* create(const libconfig::Setting& settings);
    };
}
#endif //JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_FACTORY_H
