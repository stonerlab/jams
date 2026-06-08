#ifndef JAMS_TEST_CORE_INTERACTIONS_H
#define JAMS_TEST_CORE_INTERACTIONS_H

#include <vector>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/helpers/exception.h"

namespace {
InteractionData make_test_interaction(const int i, const int j, const double rx) {
  InteractionData interaction;
  interaction.basis_site_i = i;
  interaction.basis_site_j = j;
  interaction.type_i = "A";
  interaction.type_j = "A";
  interaction.interaction_vector_cart = {rx, 0.0, 0.0};
  interaction.interaction_value_tensor[0][0] = 1.0;
  interaction.interaction_value_tensor[1][1] = 2.0;
  interaction.interaction_value_tensor[2][2] = 3.0;
  return interaction;
}
}

class InteractionsPostProcessTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::config = std::make_unique<libconfig::Config>();
    globals::lattice = new Lattice();

    globals::config->readString(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        check_closeness = false;
        parameter = 1.0e-9;
        basis = (
          [1000000.0, 0.0, 0.0],
          [0.0, 1000000.0, 0.0],
          [0.0, 0.0, 1000000.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0])
        );
      };

      lattice : {
        size = [1, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };
    )");
    globals::lattice->init_from_config(*globals::config);
  }

  void TearDown() override {
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
  }
};

TEST(InteractionsTest, SymmetryCheckAcceptsReversedInteraction) {
  auto forward = make_test_interaction(0, 1, 1.0);
  auto reverse = make_test_interaction(1, 0, -1.0);

  EXPECT_NO_THROW(check_interaction_list_symmetry({forward, reverse}));
}

TEST(InteractionsTest, SymmetryCheckRejectsMissingReversedInteraction) {
  auto forward = make_test_interaction(0, 1, 1.0);

  EXPECT_THROW(check_interaction_list_symmetry({forward}), jams::SanityException);
}

TEST_F(InteractionsPostProcessTest, RadiusCutoffUsesAbsoluteDistanceTolerance) {
  InteractionFileDescription desc;
  desc.type = InteractionFileFormat::UNDEFINED;
  desc.dimension = InteractionType::TENSOR;

  std::vector<InteractionData> interactions;
  interactions.push_back(make_test_interaction(0, 0, 1000000.05));

  post_process_interactions(
      interactions,
      desc,
      CoordinateFormat::CARTESIAN,
      false,
      0.0,
      1000000.0,
      1.0e-4);

  EXPECT_TRUE(interactions.empty());
}

TEST_F(InteractionsPostProcessTest, NonIntegerLatticeTranslationThrows) {
  InteractionFileDescription desc;
  desc.type = InteractionFileFormat::UNDEFINED;
  desc.dimension = InteractionType::TENSOR;

  std::vector<InteractionData> interactions;
  interactions.push_back(make_test_interaction(0, 0, 1250000.0));

  EXPECT_THROW(
      post_process_interactions(
          interactions,
          desc,
          CoordinateFormat::CARTESIAN,
          false,
          0.0,
          0.0,
          1.0e-4),
      jams::SanityException);
}

TEST_F(InteractionsPostProcessTest, SymmetryCheckRejectsLargeVectorFractionalMismatch) {
  auto forward = make_test_interaction(0, 0, 10000000000.0);
  auto reverse = make_test_interaction(0, 0, -10000000000.0 + 200.0);

  EXPECT_THROW(check_interaction_list_symmetry({forward, reverse}), jams::SanityException);
}

#endif // JAMS_TEST_CORE_INTERACTIONS_H
