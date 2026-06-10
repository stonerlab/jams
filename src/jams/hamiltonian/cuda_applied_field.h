#ifndef JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H
#define JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H

#include <jams/cuda/cuda_stream.h>
#include <jams/hamiltonian/applied_field.h>

class CudaAppliedFieldHamiltonian : public AppliedFieldHamiltonian {
public:
    CudaAppliedFieldHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    void calculate_fields(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;
    void calculate_energies(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;
    jams::Real calculate_total_energy(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;
};
#endif //JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_H