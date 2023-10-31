// cuda_fft_interaction.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_FFT_INTERACTION
#define INCLUDED_JAMS_CUDA_FFT_INTERACTION
#include <jams/core/hamiltonian.h>

class CudaFFTInteractionHamiltonian : public Hamiltonian {

    CudaFFTInteractionHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

private:
    CudaStream cusparse_stream_; // cuda stream to run in

};

#endif
// ----------------------------- END-OF-FILE ----------------------------------