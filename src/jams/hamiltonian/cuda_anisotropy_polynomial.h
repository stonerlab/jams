#ifndef JAMS_HAMILTONIAN_CUDA_ANISOTROPY_POLYNOMIAL_H
#define JAMS_HAMILTONIAN_CUDA_ANISOTROPY_POLYNOMIAL_H

#include <jams/hamiltonian/anisotropy_polynomial.h>

class CudaAnisotropyPolynomialHamiltonian : public AnisotropyPolynomialHamiltonian {
public:
    CudaAnisotropyPolynomialHamiltonian(const libconfig::Setting &settings, unsigned int size);

    void calculate_fields(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;
    void calculate_energies(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;
    jams::Real calculate_total_energy(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;

private:
    unsigned int dev_blocksize_ = 256;
};

#endif // JAMS_HAMILTONIAN_CUDA_ANISOTROPY_POLYNOMIAL_H
