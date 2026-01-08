#ifndef JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_H
#define JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_H

#include <jams/core/hamiltonian.h>
#include <jams/cuda/cuda_stream.h>

#include "jams/helpers/exception.h"

///
/// Hamiltonian for a time and space variable magnetic field pulse
///
/// \f[
///     \mathcal{H} = -\sum_{i} \mu_{s,i} B(t, \vec{r}_i) \cdot \vec{S}_{i}
/// \f]
///
/// @details 2020-11-17 In principle this can be used extend in the future for
/// different types of pulses. For now the pulse is gaussian in time and has a
/// hard cutoff in the z-direction allowing to be be allied to just a few
/// surface layers of atoms. This is for a specific project with Tobias
/// Kampfrath.
///
/// @warning Most of the members are not implemented
///
class CudaFieldPulseHamiltonian : public Hamiltonian {

public:

    CudaFieldPulseHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Real calculate_total_energy(jams::Real time) override {
      throw jams::unimplemented_error("CudaFieldPulseHamiltonian::calculate_total_energy is unimplemented");
    }

    void calculate_energies(jams::Real time) override {
      throw jams::unimplemented_error("CudaFieldPulseHamiltonian::calculate_energies is unimplemented");
    }

    void calculate_fields(jams::Real time) override;

    Vec3R calculate_field(int i, jams::Real time) override {
      throw jams::unimplemented_error("CudaFieldPulseHamiltonian::calculate_field is unimplemented");
    }

    jams::Real calculate_energy(int i, jams::Real time) override {
      throw jams::unimplemented_error("CudaFieldPulseHamiltonian::calculate_energy is unimplemented");
    }

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override {
      throw jams::unimplemented_error("CudaFieldPulseHamiltonian::calculate_energy_difference is unimplemented");
    }

private:
    jams::MultiArray<jams::Real, 2> positions_;

    jams::Real surface_cutoff_;
    jams::Real temporal_width_;
    jams::Real temporal_center_;
    Vec3R max_field_;

    void output_pulse(std::ofstream& pulse_file);
};

#endif //JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_H
