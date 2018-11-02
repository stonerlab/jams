#ifndef JAMS_HAMILTONIAN_STRATEGY
#define JAMS_HAMILTONIAN_STRATEGY

#if HAS_CUDA
#include "jblib/containers/cuda_array.h"
#endif

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
        inline HamiltonianStrategy(const libconfig::Setting &settings, const unsigned int size) {};
        inline virtual ~HamiltonianStrategy() {};

        virtual double calculate_total_energy() = 0;
        virtual double calculate_one_spin_energy(const int i) = 0;
        virtual double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) = 0;
        virtual void   calculate_energies(jblib::Array<double, 1>& energies) = 0;

        virtual void   calculate_one_spin_field(const int i, double h[3]) = 0;
        virtual void   calculate_fields(jblib::Array<double, 2>& fields) = 0;

#if HAS_CUDA
        virtual void   calculate_fields(jblib::CudaArray<double, 1>& fields) {};
#endif
};

#endif  // JAMS_HAMILTONIAN_STRATEGY