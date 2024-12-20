// cuda_landau.h                                                       -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_LANDAU_H
#define INCLUDED_JAMS_CUDA_LANDAU_H


#include <jams/cuda/cuda_stream.h>
#include <jams/core/hamiltonian.h>
#include <jams/helpers/exception.h>

class CudaLandauHamiltonian : public Hamiltonian {
public:
    CudaLandauHamiltonian(const libconfig::Setting &settings, const unsigned int size);


    inline Vec3 calculate_field(int i, double time)  override
    { throw jams::unimplemented_error("CudaLandauHamiltonian::calculate_field"); }


    inline double calculate_energy(int i, double time)  override
    { throw jams::unimplemented_error("CudaLandauHamiltonian::calculate_energy"); }

    inline double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time)  override
    { throw jams::unimplemented_error("CudaLandauHamiltonian::calculate_energy_difference"); }

    double calculate_total_energy(double time) override;
    void calculate_fields(double time) override;
    void calculate_energies(double time) override;

private:
    CudaStream cuda_stream_;
    unsigned int dev_blocksize_ = 64;

    jams::MultiArray<double,1> landau_A_;
    jams::MultiArray<double,1> landau_B_;
    jams::MultiArray<double,1> landau_C_;
};



#endif
// ----------------------------- END-OF-FILE ----------------------------------