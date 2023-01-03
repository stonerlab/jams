#ifndef JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H
#define JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H

#include <jams/cuda/cuda_stream.h>
#include <jams/hamiltonian/applied_field.h>

class CudaAppliedFieldHamiltonian : public AppliedFieldHamiltonian {
public:
    CudaAppliedFieldHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    void calculate_fields(double time) override;
    void calculate_energies(double time) override;

private:
    CudaStream cuda_stream_;
};
#endif //JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H