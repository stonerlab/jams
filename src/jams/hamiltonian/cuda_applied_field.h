#ifndef JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H
#define JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H

#include <jams/cuda/cuda_stream.h>
#include <jams/hamiltonian/applied_field.h>

class CudaAppliedFieldHamiltonian : public AppliedFieldHamiltonian {
public:
    CudaAppliedFieldHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    void calculate_fields(jams::Real time) override;
    void calculate_energies(jams::Real time) override;
    jams::Real calculate_total_energy(jams::Real time) override;
};
#endif //JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H