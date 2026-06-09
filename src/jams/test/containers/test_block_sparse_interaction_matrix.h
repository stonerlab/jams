#ifndef JAMS_TEST_CONTAINERS_TEST_BLOCK_SPARSE_INTERACTION_MATRIX_H
#define JAMS_TEST_CONTAINERS_TEST_BLOCK_SPARSE_INTERACTION_MATRIX_H

#include <jams/containers/block_sparse_interaction_matrix.h>

TEST(BlockSparseInteractionMatrixTest, StoresIsotropicScalarBlocks) {
  jams::BlockSparseInteractionMatrix<double>::Builder builder(
      2, jams::InteractionTensorStorage::Auto, 0.0);

  builder.insert(0, 1, 2.0);
  builder.insert(1, 0, 2.0);

  EXPECT_TRUE(builder.is_symmetric());
  auto matrix = builder.build();

  EXPECT_EQ(matrix.storage(), jams::InteractionTensorStorage::Isotropic);
  EXPECT_EQ(matrix.components_per_block(), 1);
  EXPECT_EQ(matrix.num_blocks(), 2);

  jams::MultiArray<double, 2> spins(2, 3);
  spins(0, 0) = 1.0;
  spins(0, 1) = 2.0;
  spins(0, 2) = 3.0;
  spins(1, 0) = 4.0;
  spins(1, 1) = 5.0;
  spins(1, 2) = 6.0;

  jams::MultiArray<double, 2> field(2, 3);
  matrix.multiply(spins, field);

  EXPECT_DOUBLE_EQ(field(0, 0), 8.0);
  EXPECT_DOUBLE_EQ(field(0, 1), 10.0);
  EXPECT_DOUBLE_EQ(field(0, 2), 12.0);
  EXPECT_DOUBLE_EQ(field(1, 0), 2.0);
  EXPECT_DOUBLE_EQ(field(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(field(1, 2), 6.0);
}

TEST(BlockSparseInteractionMatrixTest, StoresSymmetricTensorBlocks) {
  jams::BlockSparseInteractionMatrix<double>::Builder builder(
      2, jams::InteractionTensorStorage::Auto, 0.0);

  const jams::Mat<double, 3, 3> tensor = {
      1.0, 2.0, 3.0,
      2.0, 4.0, 5.0,
      3.0, 5.0, 6.0};
  builder.insert(0, 1, tensor);
  builder.insert(1, 0, tensor);

  EXPECT_TRUE(builder.is_symmetric());
  auto matrix = builder.build();
  EXPECT_EQ(matrix.storage(), jams::InteractionTensorStorage::Symmetric);
  EXPECT_EQ(matrix.components_per_block(), 6);

  jams::MultiArray<double, 2> spins(2, 3);
  spins(0, 0) = 1.0;
  spins(0, 1) = 0.0;
  spins(0, 2) = 0.0;
  spins(1, 0) = 0.0;
  spins(1, 1) = 1.0;
  spins(1, 2) = 0.0;

  jams::MultiArray<double, 2> field(2, 3);
  matrix.multiply(spins, field);

  EXPECT_DOUBLE_EQ(field(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(field(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(field(0, 2), 5.0);
  EXPECT_DOUBLE_EQ(field(1, 0), 1.0);
  EXPECT_DOUBLE_EQ(field(1, 1), 2.0);
  EXPECT_DOUBLE_EQ(field(1, 2), 3.0);
}

TEST(BlockSparseInteractionMatrixTest, StoresAntisymmetricTensorBlocks) {
  jams::BlockSparseInteractionMatrix<double>::Builder builder(
      2, jams::InteractionTensorStorage::Auto, 0.0);

  const jams::Mat<double, 3, 3> tensor = {
      0.0, 7.0, 11.0,
      -7.0, 0.0, 13.0,
      -11.0, -13.0, 0.0};
  builder.insert(0, 1, tensor);
  builder.insert(1, 0, transpose(tensor));

  EXPECT_TRUE(builder.is_symmetric());
  auto matrix = builder.build();
  EXPECT_EQ(matrix.storage(), jams::InteractionTensorStorage::Antisymmetric);
  EXPECT_EQ(matrix.components_per_block(), 3);

  jams::MultiArray<double, 2> spins(2, 3);
  spins(0, 0) = 1.0;
  spins(0, 1) = 0.0;
  spins(0, 2) = 0.0;
  spins(1, 0) = 0.0;
  spins(1, 1) = 1.0;
  spins(1, 2) = 0.0;

  jams::MultiArray<double, 2> field(2, 3);
  matrix.multiply(spins, field);

  EXPECT_DOUBLE_EQ(field(0, 0), 7.0);
  EXPECT_DOUBLE_EQ(field(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(field(0, 2), -13.0);
  EXPECT_DOUBLE_EQ(field(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(field(1, 1), 7.0);
  EXPECT_DOUBLE_EQ(field(1, 2), 11.0);
}

TEST(BlockSparseInteractionMatrixTest, EscalatesAntisymmetricAndIsotropicMixToGeneral) {
  jams::BlockSparseInteractionMatrix<double>::Builder builder(
      3, jams::InteractionTensorStorage::Auto, 0.0);

  const jams::Mat<double, 3, 3> antisymmetric = {
      0.0, 1.0, 2.0,
      -1.0, 0.0, 3.0,
      -2.0, -3.0, 0.0};
  builder.insert(0, 1, antisymmetric);
  builder.insert(1, 0, transpose(antisymmetric));
  builder.insert(1, 2, 4.0);
  builder.insert(2, 1, 4.0);

  EXPECT_TRUE(builder.is_symmetric());
  const auto matrix = builder.build();
  EXPECT_EQ(matrix.storage(), jams::InteractionTensorStorage::General);
  EXPECT_EQ(matrix.components_per_block(), 9);
}

#endif  // JAMS_TEST_CONTAINERS_TEST_BLOCK_SPARSE_INTERACTION_MATRIX_H
