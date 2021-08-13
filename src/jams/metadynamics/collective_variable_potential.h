#ifndef JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H
#define JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H

#include <iosfwd>
#include <jams/containers/vec3.h>

namespace jams {
    class CollectiveVariablePotential {
    public:
        virtual ~CollectiveVariablePotential() = default;

        /// Inserts a Gaussian energy peak into the potential energy landscape.
        /// This may be multi-dimensional. The widths and absolute amplitude are
        /// constant and should be specified in the constructor. The relative
        /// amplitude is for scaling inserted Gaussians, for example when doing
        /// tempered metadynamics.
        virtual void insert_gaussian(const double& relative_amplitude = 1.0) = 0;

        /// Output the potential landscape to a file stream.
        /// TODO: Define what the format of this file will be
        virtual void output() = 0;

        /// Returns the value of the potential at the current coordinates of the
        /// collective variable
        virtual double current_potential() = 0;

        /// Calculate the difference in potential energy for the system when a
        /// single spin is changed from spin_initial to spin_final
        virtual double potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) = 0;

        virtual void spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) = 0;
    };
}

#endif //JAMS_METADYNAMICS_COLLECTIVE_VARIABLE_POTENTIAL_H
