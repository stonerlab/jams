// cuda_pisd_field_exchange.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_PISD_FIELD_EXCHANGE
#define INCLUDED_JAMS_CUDA_PISD_FIELD_EXCHANGE

#include <jams/hamiltonian/general_sparse_two_site_interaction.h>
#include <jams/containers/interaction_list.h>
#include <jams/containers/sparse_matrix_builder.h>

class CudaPISDFieldExchangeHamiltonian : public GeneralSparseTwoSiteInteractionHamiltonian<double> {
public:
    CudaPISDFieldExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;


private:
    std::array<double, 3> applied_field_ = {0.0, 0.0, 0.0};
    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour list
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------