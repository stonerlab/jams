#ifndef JAMS_HAMILTONIAN_STRATEGY
#define JAMS_HAMILTONIAN_STRATEGY

// forward declarations
namespace libconfig {
    class Setting;
}

class HamiltonianStrategy {
    public:
        inline HamiltonianStrategy(const libconfig::Setting &settings, const unsigned int size) {};
        inline virtual ~HamiltonianStrategy() {};

        virtual double calculate_total_energy() = 0;
        virtual double calculate_one_spin_energy(const int i) = 0;
        virtual double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) = 0;
        virtual void   calculate_energies(jams::MultiArray<double, 1>& energies) = 0;

        virtual void   calculate_one_spin_field(const int i, double h[3]) = 0;
        virtual void   calculate_fields(jams::MultiArray<double, 2>& fields) = 0;
};

#endif  // JAMS_HAMILTONIAN_STRATEGY