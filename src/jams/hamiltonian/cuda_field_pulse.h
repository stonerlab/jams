#ifndef JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_H
#define JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_H

#include <jams/core/hamiltonian.h>
#include <jams/cuda/cuda_stream.h>

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

    double calculate_total_energy() override {
      assert(false); // unimplemented
      return 0.0;
    }

    void calculate_energies() override {
      assert(false); // unimplemented
    }

    void calculate_fields() override;

    Vec3 calculate_field(int i) override {
      assert(false); // unimplemented
      return {0.0, 0.0, 0.0};
    }

    double calculate_energy(int i) override {
      assert(false); // unimplemented
      return 0.0;
    }

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override {
      assert(false); // unimplemented
      return 0.0;
    }

private:
    CudaStream cuda_stream_;

    jams::MultiArray<double, 2> positions_;

    double surface_cutoff_;
    double temporal_width_;
    double temporal_center_;
    Vec3 max_field_;

    void output_pulse(std::ofstream& pulse_file);
};

#endif //JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_H
