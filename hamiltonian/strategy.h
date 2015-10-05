#ifndef JAMS_HAMILTONIAN_STRATEGY
#define JAMS_HAMILTONIAN_STRATEGY

// forward declarations
namespace libconfig {
    class Setting;
}

namespace jblib {
    template <typename T>
    class Vec3;

    template <typename T, int n, typename Idx>
    class Array;
}

class HamiltonianStrategy {
    public:
        inline HamiltonianStrategy(const libconfig::Setting &settings) {};
        inline virtual ~HamiltonianStrategy() {};

        virtual double calculate_total_energy() = 0;
        virtual double calculate_one_spin_energy(const int i) = 0;
        virtual double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) = 0;
        virtual void   calculate_energies(jblib::Array<double, 1>& energies) = 0;

        virtual void   calculate_one_spin_field(const int i, double h[3]) = 0;
        virtual void   calculate_fields(jblib::Array<double, 2>& fields) = 0;
};

#endif  // JAMS_HAMILTONIAN_STRATEGY