// cvar_cuda_magnetisation.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_METADYNAMICS_CVAR_REDUCED_MZ_CUDA
#define INCLUDED_JAMS_METADYNAMICS_CVAR_REDUCED_MZ_CUDA

#include <jams/metadynamics/cvars/cvar_reduced_mz.h>

#include <jams/containers/multiarray.h>

namespace jams {
class CVarReducedMzCuda : public CVarReducedMz {
public:
    CVarReducedMzCuda() = default;
    explicit CVarReducedMzCuda(const libconfig::Setting &settings);

    double value() override;

    const jams::MultiArray<double, 2>& derivatives();
private:
    jams::MultiArray<double, 2> derivatives_;

};
}


#endif //INCLUDED_JAMS_METADYNAMICS_CVAR_REDUCED_MZ_CUDA
