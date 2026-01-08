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
        virtual Vec3R field(const jams::Real time) = 0;
    };

    AppliedFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

protected:
    std::unique_ptr<TimeDependentField> time_dependent_field_;
};

#endif //JAMS_HAMILTONIAN_APPLIED_FIELD_H
