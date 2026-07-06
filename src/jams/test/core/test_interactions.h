#ifndef JAMS_TEST_CORE_INTERACTIONS_H
#define JAMS_TEST_CORE_INTERACTIONS_H

#include <fstream>
#include <filesystem>
#include <vector>
#include <memory>
#include <string>
#include <system_error>

#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/helpers/exception.h"

namespace {
class InteractionTestFile {
public:
  InteractionTestFile(const std::string& filename, const std::string& contents)
      : path_(std::filesystem::temp_directory_path() / filename) {
    std::filesystem::remove(path_);

    std::ofstream output(path_);
    output << contents;
    output.close();

    file_.open(path_);
  }

  ~InteractionTestFile() {
    file_.close();

    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  operator std::ifstream&() {
    return file_;
  }

private:
  std::filesystem::path path_;
  std::ifstream file_;
};

InteractionTestFile interaction_test_file(const std::string& filename, const std::string& contents) {
  return InteractionTestFile(filename, contents);
}

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

void expect_neighbour_list_excludes_but_uses_co(
    const jams::InteractionList<jams::Mat<double, 3, 3>, 2>& neighbour_list) {
  ASSERT_GT(neighbour_list.num_interactions(), 0);

  bool saw_co_interaction = false;
  for (int i = 0; i < globals::num_spins; ++i) {
    for (const auto& [indices, value] : neighbour_list.interactions_of(i)) {
      (void)value;
      const auto j = indices[1];
      const auto local_material = globals::lattice->lattice_site_material_name(i);
      const auto neighbour_material = globals::lattice->lattice_site_material_name(j);
      EXPECT_NE(local_material, "B");
      EXPECT_NE(neighbour_material, "B");
      saw_co_interaction = saw_co_interaction
          || local_material == "Co"
          || neighbour_material == "Co";
    }
  }

  EXPECT_TRUE(saw_co_interaction);
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

class InteractionsPositionalMaterialTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::config = std::make_unique<libconfig::Config>();
    globals::lattice = new Lattice();

    globals::config->readString(R"(
      materials = (
        { name = "Fe"; moment = 1.0; spin = [1.0, 0.0, 0.0]; },
        { name = "Co"; moment = 1.0; spin = [0.0, 1.0, 0.0]; },
        { name = "B";  moment = 0.0; spin = [0.0, 0.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        check_closeness = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("Fe", [0.0, 0.0, 0.0])
        );
      };

      lattice : {
        size = [4, 1, 1];
        periodic = [true, false, false];
        impurities = (
          ("Fe", "Co", 0.25),
          ("Fe", "B",  0.25)
        );
        impurities_seed = 123456;
        normalise_spins = false;
      };

      hamiltonians = (
        {
          module = "exchange";
          symops = false;
          coordinate_format = "fractional";
          check_no_zero_motif_neighbour_count = false;
          check_identical_motif_neighbour_count = false;
          check_identical_motif_total_exchange = false;
          interactions = (
            (1, 1, "Fe", "Fe", [ 1.0, 0.0, 0.0], 1.0),
            (1, 1, "Fe", "Fe", [-1.0, 0.0, 0.0], 1.0),
            (1, 1, "Fe", "Co", [ 1.0, 0.0, 0.0], 2.0),
            (1, 1, "Co", "Fe", [-1.0, 0.0, 0.0], 2.0),
            (1, 1, "Co", "Fe", [ 1.0, 0.0, 0.0], 3.0),
            (1, 1, "Fe", "Co", [-1.0, 0.0, 0.0], 3.0),
            (1, 1, "Co", "Co", [ 1.0, 0.0, 0.0], 4.0),
            (1, 1, "Co", "Co", [-1.0, 0.0, 0.0], 4.0)
          );
        }
      );
    )");
    globals::lattice->init_from_config(*globals::config);
  }

  void TearDown() override {
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
  }
};

class InteractionsSafetyCheckTest : public ::testing::Test {
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
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0]),
          ("A", [0.9998, 0.0, 0.0])
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

TEST(InteractionsTest, InteractionFileFormatAcceptsPositionalMaterialName) {
  EXPECT_EQ(
      interaction_file_format_from_string("positional_material"),
      InteractionFileFormat::POSITIONAL_MATERIAL);
}

TEST(InteractionsTest, DiscoversPositionalMaterialFileFormats) {
  {
    auto file = interaction_test_file("exc_positional_material_scalar.in", "1 1 Fe Co 0.0 0.0 1.0 2.0\n");
    EXPECT_TRUE(discover_interaction_file_format(file) ==
      InteractionFileDescription(InteractionFileFormat::POSITIONAL_MATERIAL, InteractionType::SCALAR));
  }

  {
    auto file = interaction_test_file(
        "exc_positional_material_tensor.in",
        "1 1 Fe Co 0.0 0.0 1.0 1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0\n");
    EXPECT_TRUE(discover_interaction_file_format(file) ==
      InteractionFileDescription(InteractionFileFormat::POSITIONAL_MATERIAL, InteractionType::TENSOR));
  }
}

TEST(InteractionsTest, ParsesPositionalMaterialFileScalar) {
  auto file = interaction_test_file(
      "exc_positional_material_scalar.in",
      "1 2 Fe Co 0.0 0.0 1.0 2.0\n");
  const auto desc = discover_interaction_file_format(file);
  const auto interactions = interactions_from_file(file, desc);

  ASSERT_EQ(interactions.size(), 1);
  EXPECT_EQ(interactions[0].basis_site_i, 0);
  EXPECT_EQ(interactions[0].basis_site_j, 1);
  EXPECT_EQ(interactions[0].type_i, "Fe");
  EXPECT_EQ(interactions[0].type_j, "Co");
  EXPECT_EQ(interactions[0].interaction_vector_cart, (jams::Vec<double, 3>{0.0, 0.0, 1.0}));
  EXPECT_EQ(interactions[0].interaction_value_tensor, (jams::diagonal_matrix<double, 3>(2.0)));
}

TEST(InteractionsTest, ParsesPositionalMaterialFileTensor) {
  auto file = interaction_test_file(
      "exc_positional_material_tensor.in",
      "2 1 Co Fe 0.0 0.0 -1.0 1.0 0.1 0.2 0.3 2.0 0.4 0.5 0.6 3.0\n");
  const auto desc = discover_interaction_file_format(file);
  const auto interactions = interactions_from_file(file, desc);

  ASSERT_EQ(interactions.size(), 1);
  EXPECT_EQ(interactions[0].basis_site_i, 1);
  EXPECT_EQ(interactions[0].basis_site_j, 0);
  EXPECT_EQ(interactions[0].type_i, "Co");
  EXPECT_EQ(interactions[0].type_j, "Fe");
  EXPECT_EQ(interactions[0].interaction_vector_cart, (jams::Vec<double, 3>{0.0, 0.0, -1.0}));
  EXPECT_EQ(interactions[0].interaction_value_tensor,
            (jams::Mat<double, 3, 3>{1.0, 0.1, 0.2, 0.3, 2.0, 0.4, 0.5, 0.6, 3.0}));
}

TEST(InteractionsTest, ParsesPositionalMaterialConfigScalar) {
  libconfig::Config config;
  config.readString(R"(
    interactions = (
      (1, 2, "Fe", "Co", [0.0, 0.0, 1.0], 2.0)
    );
  )");
  auto& settings = config.lookup("interactions");
  const auto desc = discover_interaction_setting_format(settings);
  const auto interactions = interactions_from_settings(settings, desc);

  ASSERT_EQ(interactions.size(), 1);
  EXPECT_EQ(desc, (InteractionFileDescription(InteractionFileFormat::POSITIONAL_MATERIAL, InteractionType::SCALAR)));
  EXPECT_EQ(interactions[0].basis_site_i, 0);
  EXPECT_EQ(interactions[0].basis_site_j, 1);
  EXPECT_EQ(interactions[0].type_i, "Fe");
  EXPECT_EQ(interactions[0].type_j, "Co");
  EXPECT_EQ(interactions[0].interaction_vector_cart, (jams::Vec<double, 3>{0.0, 0.0, 1.0}));
  EXPECT_EQ(interactions[0].interaction_value_tensor, (jams::diagonal_matrix<double, 3>(2.0)));
}

TEST(InteractionsTest, ParsesPositionalMaterialConfigTensor) {
  libconfig::Config config;
  config.readString(R"(
    interactions = (
      (2, 1, "Co", "Fe", [0.0, 0.0, -1.0], [1.0, 0.1, 0.2, 0.3, 2.0, 0.4, 0.5, 0.6, 3.0])
    );
  )");
  auto& settings = config.lookup("interactions");
  const auto desc = discover_interaction_setting_format(settings);
  const auto interactions = interactions_from_settings(settings, desc);

  ASSERT_EQ(interactions.size(), 1);
  EXPECT_EQ(desc, (InteractionFileDescription(InteractionFileFormat::POSITIONAL_MATERIAL, InteractionType::TENSOR)));
  EXPECT_EQ(interactions[0].basis_site_i, 1);
  EXPECT_EQ(interactions[0].basis_site_j, 0);
  EXPECT_EQ(interactions[0].type_i, "Co");
  EXPECT_EQ(interactions[0].type_j, "Fe");
  EXPECT_EQ(interactions[0].interaction_vector_cart, (jams::Vec<double, 3>{0.0, 0.0, -1.0}));
  EXPECT_EQ(interactions[0].interaction_value_tensor,
            (jams::Mat<double, 3, 3>{1.0, 0.1, 0.2, 0.3, 2.0, 0.4, 0.5, 0.6, 3.0}));
}

TEST(InteractionsTest, RejectsMalformedPositionalMaterialConfig) {
  libconfig::Config config;
  config.readString(R"(
    interactions = (
      (1, 1, "Fe", [0.0, 0.0, 1.0], 2.0)
    );
  )");
  auto& settings = config.lookup("interactions");

  EXPECT_THROW(discover_interaction_setting_format(settings), std::runtime_error);
}

TEST_F(InteractionsPositionalMaterialTest, PositionalMaterialInteractionsFilterImpurityMaterials) {
  auto& settings = globals::config->lookup("hamiltonians")[0]["interactions"];
  const auto neighbour_list = generate_neighbour_list(
      settings,
      CoordinateFormat::FRACTIONAL,
      false,
      0.0,
      0.0,
      1.0e-6,
      {});

  expect_neighbour_list_excludes_but_uses_co(neighbour_list);
}

TEST_F(InteractionsPositionalMaterialTest, PositionalMaterialExchangeFileFiltersImpurityMaterials) {
  auto file = interaction_test_file("positional_material_exchange.in", R"(
1 1 Fe Fe  1.0 0.0 0.0 1.0
1 1 Fe Fe -1.0 0.0 0.0 1.0
1 1 Fe Co  1.0 0.0 0.0 2.0
1 1 Co Fe -1.0 0.0 0.0 2.0
1 1 Co Fe  1.0 0.0 0.0 3.0
1 1 Fe Co -1.0 0.0 0.0 3.0
1 1 Co Co  1.0 0.0 0.0 4.0
1 1 Co Co -1.0 0.0 0.0 4.0
)");
  const auto neighbour_list = generate_neighbour_list(
      file,
      CoordinateFormat::FRACTIONAL,
      false,
      0.0,
      0.0,
      1.0e-6,
      {});

  expect_neighbour_list_excludes_but_uses_co(neighbour_list);
}

TEST_F(InteractionsPostProcessTest, RadiusCutoffUsesOwnCartesianTolerance) {
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
      1.0,
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

TEST_F(InteractionsSafetyCheckTest, DistanceToleranceChecksPeriodicFractionalImages) {
  EXPECT_THROW(safety_check_distance_tolerance(5.0e-4), jams::SanityException);
}

#endif // JAMS_TEST_CORE_INTERACTIONS_H
