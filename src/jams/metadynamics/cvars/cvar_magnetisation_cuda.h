// cvar_cuda_magnetisation.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_METADYNAMICS_CVAR_MAGNETISATION_CUDA
#define INCLUDED_JAMS_METADYNAMICS_CVAR_MAGNETISATION_CUDA

#include <jams/metadynamics/cvars/cvar_magnetisation.h>

#include <jams/containers/multiarray.h>

namespace jams {
class CVarMagnetisationCuda : public CVarMagnetisation {
public:
    CVarMagnetisationCuda() = default;
    explicit CVarMagnetisationCuda(const libconfig::Setting &settings);

    double value() override;

    const jams::MultiArray<double, 2>& derivatives();
private:
    jams::MultiArray<double, 2> derivatives_;

};
}


#endif //INCLUDED_JAMS_METADYNAMICS_CVAR_MAGNETISATION_CUDA
