#ifndef JAMS_TEST_CORE_INTERACTIONS_H
#define JAMS_TEST_CORE_INTERACTIONS_H

#include <vector>

#include <gtest/gtest.h>

#include "jams/core/interactions.h"
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

TEST(InteractionsTest, SymmetryCheckAcceptsReversedInteraction) {
  auto forward = make_test_interaction(0, 1, 1.0);
  auto reverse = make_test_interaction(1, 0, -1.0);

  EXPECT_NO_THROW(check_interaction_list_symmetry({forward, reverse}));
}

TEST(InteractionsTest, SymmetryCheckRejectsMissingReversedInteraction) {
  auto forward = make_test_interaction(0, 1, 1.0);

  EXPECT_THROW(check_interaction_list_symmetry({forward}), jams::SanityException);
}

#endif // JAMS_TEST_CORE_INTERACTIONS_H
