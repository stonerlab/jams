#ifndef JAMS_HAMILTONIAN_FIELD_PULSE_H
#define JAMS_HAMILTONIAN_FIELD_PULSE_H


#include <jams/core/hamiltonian.h>
#include <jams/containers/vec3.h>
#include <jams/interface/config.h>

#include <iosfwd>

///
/// Generic concept for a field pulse which can vary in both time and space
///
class TemporalFieldPulse {
public:
    virtual ~TemporalFieldPulse() = default;
    virtual Vec3 local_field(const double& time, const Vec3& r) = 0;
    virtual Vec3 max_field(const double& time) = 0;
};

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
/// Kampfrath
///
class FieldPulseHamiltonian : public Hamiltonian {

public:

    FieldPulseHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    void output_pulse(std::ofstream& pulse_file);

    std::unique_ptr<TemporalFieldPulse> temporal_field_pulse_;
};

#endif //JAMS_HAMILTONIAN_FIELD_PULSE_H
