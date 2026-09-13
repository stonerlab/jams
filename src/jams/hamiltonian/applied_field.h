#ifndef JAMS_HAMILTONIAN_APPLIED_FIELD_H
#define JAMS_HAMILTONIAN_APPLIED_FIELD_H

#include <jams/core/hamiltonian.h>

///
/// Hamiltonian for a spatially homogeneous, applied magnetic field
///
/// \f[
///     \mathcal{H} = -\sum_{i} \mu_{s,i} \vec{B} \cdot \vec{S}_{i}
/// \f]
///
/// @details Provides virtual method @p set_b_field to allow derived class
/// to change the b_field for example if doing time dependent fields. This
/// class makes no assumption that the field will be static in time.
///
class AppliedFieldHamiltonian : public Hamiltonian {

public:
    struct TimeDependentField {
    public:
        virtual ~TimeDependentField() = default;
        virtual jams::Vec<jams::Real, 3> field(const double time) = 0;
    };

    AppliedFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Vec<jams::Real, 3> calculate_field(int i, double time) override;

    jams::Real calculate_energy(int i, double time) override;

    EnergyCurrentInteractionSupport energy_current_interaction_support() const override {
      return EnergyCurrentInteractionSupport::None;
    }

protected:
    jams::Real calculate_energy_from_spins(int i, double time, const SpinHostView& spins) override;

    jams::Real calculate_energy_for_spin(int i, const jams::Vec<double, 3> &spin, double time) override;

    std::unique_ptr<TimeDependentField> time_dependent_field_;
};

#endif //JAMS_HAMILTONIAN_APPLIED_FIELD_H
