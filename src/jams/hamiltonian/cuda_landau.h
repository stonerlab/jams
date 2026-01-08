// cuda_landau.h                                                       -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_LANDAU_H
#define INCLUDED_JAMS_CUDA_LANDAU_H


#include <jams/cuda/cuda_stream.h>
#include <jams/core/hamiltonian.h>
#include <jams/helpers/exception.h>

class CudaLandauHamiltonian : public Hamiltonian {
public:
    CudaLandauHamiltonian(const libconfig::Setting &settings, const unsigned int size);


    inline Vec3R calculate_field(int i, jams::Real time)  override
    { throw jams::unimplemented_error("CudaLandauHamiltonian::calculate_field"); }


    inline jams::Real calculate_energy(int i, jams::Real time)  override
    { throw jams::unimplemented_error("CudaLandauHamiltonian::calculate_energy"); }

    inline jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time)  override
    { throw jams::unimplemented_error("CudaLandauHamiltonian::calculate_energy_difference"); }

    jams::Real calculate_total_energy(jams::Real time) override;
    void calculate_fields(jams::Real time) override;
    void calculate_energies(jams::Real time) override;

private:
    unsigned int dev_blocksize_ = 64;

    jams::MultiArray<jams::Real,1> landau_A_;
    jams::MultiArray<jams::Real,1> landau_B_;
    jams::MultiArray<jams::Real,1> landau_C_;
};



#endif
// ----------------------------- END-OF-FILE ----------------------------------