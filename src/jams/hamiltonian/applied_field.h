#ifndef JAMS_HAMILTONIAN_APPLIED_FIELD_H
#define JAMS_HAMILTONIAN_APPLIED_FIELD_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/vec3.h>
#include <jams/interface/config.h>

class AppliedFieldHamiltonian : public Hamiltonian {

public:
    AppliedFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    Vec3 applied_field_ = {0.0, 0.0, 0.0};
};

#endif //JAMS_HAMILTONIAN_APPLIED_FIELD_H
