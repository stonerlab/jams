#ifndef JAMS_HAMILTONIAN_APPLIED_FIELD_H
#define JAMS_HAMILTONIAN_APPLIED_FIELD_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/vec3.h>
#include <jams/interface/config.h>

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
    AppliedFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

    /// Returns the B field used in the Hamiltonian
    const Vec3& b_field() const;

    /// Changes the B field used in the Hamiltonian
    virtual void set_b_field(const Vec3& field);

private:
    Vec3 applied_b_field_ = {0.0, 0.0, 0.0};
};

#endif //JAMS_HAMILTONIAN_APPLIED_FIELD_H
