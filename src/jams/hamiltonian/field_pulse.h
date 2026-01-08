#ifndef JAMS_HAMILTONIAN_FIELD_PULSE_H
#define JAMS_HAMILTONIAN_FIELD_PULSE_H

#include <jams/core/hamiltonian.h>

#include <iosfwd>
#include <memory>

///
/// Generic concept for a field pulse which can vary in both time and space
///
class TemporalFieldPulse {
public:
    virtual ~TemporalFieldPulse() = default;
    virtual Vec3R local_field(const jams::Real& time, const Vec3R& r) = 0;
    virtual Vec3R max_field(const jams::Real& time) = 0;
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

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

private:
    void output_pulse(std::ofstream& pulse_file);

    std::unique_ptr<TemporalFieldPulse> temporal_field_pulse_;
};

#endif //JAMS_HAMILTONIAN_FIELD_PULSE_H
