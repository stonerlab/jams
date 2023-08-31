// landau.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_HAMILTONIAN_LANDAU
#define INCLUDED_JAMS_HAMILTONIAN_LANDAU

#include <jams/core/hamiltonian.h>


class LandauHamiltonian : public Hamiltonian {

public:
    LandauHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

protected:
    jams::MultiArray<double,1> landau_A_;
    jams::MultiArray<double,1> landau_B_;
    jams::MultiArray<double,1> landau_C_;
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------