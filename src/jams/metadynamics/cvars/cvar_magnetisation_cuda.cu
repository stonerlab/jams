// cvar_cuda_magnetisation.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_magnetisation_cuda.h>
#include <jams/cuda/cuda_stride_reduce.h>
#include <jams/core/globals.h>


jams::CVarMagnetisationCuda::CVarMagnetisationCuda(const libconfig::Setting &settings)
: CVarMagnetisation(settings)
{
  zero(derivatives_.resize(globals::num_spins, 3));
  for (auto i = 0; i < globals::num_spins; ++i) {
    derivatives_(i,magnetisation_component_) = (1.0/globals::num_spins);
  }
}

double jams::CVarMagnetisationCuda::value() {
  return cuda_stride_reduce_array(globals::s.device_data(), globals::num_spins3, 3, magnetisation_component_)  / globals::num_spins;
}


const jams::MultiArray<double, 2>&
jams::CVarMagnetisationCuda::derivatives() {
  return derivatives_;
}
