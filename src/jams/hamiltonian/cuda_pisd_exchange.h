//
// Created by Joseph Barker on 04/02/2025.
//

#ifndef INCLUDED_JAMS_CUDA_PISD_EXCHANGE_H
#define INCLUDED_JAMS_CUDA_PISD_EXCHANGE_H

#include <jams/containers/interaction_list.h>
#include <jams/containers/sparse_matrix_builder.h>

#include <libconfig.h++>

#include <jams/hamiltonian/neighbour_list_interaction.h>

class CudaPisdExchangeHamiltonian : public NeighbourListInteractionHamiltonian {
  public:
    CudaPisdExchangeHamiltonian(const libconfig::Setting &setting, unsigned int size);

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

  private:
    double bz_field_;
    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour list
};

#endif //INCLUDED_JAMS_CUDA_PISD_EXCHANGE_H
