#ifndef JAMS_TEST_LATTICE_SIZE_H
#define JAMS_TEST_LATTICE_SIZE_H

#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <string>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/exception.h"
#include "jams/interface/fft.h"

namespace jams::testing {

class LatticeSizeTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::config = std::make_unique<libconfig::Config>();
    globals::lattice = new Lattice();
  }

  void TearDown() override {
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
  }

  void initialise_lattice(const std::string& lattice_config) {
    globals::config->readString(base_config() + lattice_config);
    globals::lattice->init_from_config(*globals::config);
  }

  void initialise_lattice_with_z_motif(const std::string& lattice_config) {
    globals::config->readString(base_z_motif_config() + lattice_config);
    globals::lattice->init_from_config(*globals::config);
  }

  void initialise_lattice_with_high_z_motif(const std::string& lattice_config) {
    globals::config->readString(base_high_z_motif_config() + lattice_config);
    globals::lattice->init_from_config(*globals::config);
  }

  void initialise_lattice_with_large_unitcell(const std::string& lattice_config) {
    globals::config->readString(base_large_unitcell_config() + lattice_config);
    globals::lattice->init_from_config(*globals::config);
  }

  static std::string base_config() {
    return R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; },
        { name = "B"; moment = 2.0; spin = [2.0, 0.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0]),
          ("B", [0.5, 0.0, 0.0])
        );
      };
    )";
  }

  static std::string base_z_motif_config() {
    return R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; },
        { name = "B"; moment = 2.0; spin = [2.0, 0.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0]),
          ("B", [0.0, 0.0, 0.5])
        );
      };
    )";
  }

  static std::string base_high_z_motif_config() {
    return R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; },
        { name = "B"; moment = 2.0; spin = [2.0, 0.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0]),
          ("B", [0.0, 0.0, 0.916667])
        );
      };
    )";
  }

  static std::string base_large_unitcell_config() {
    return R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; },
        { name = "B"; moment = 2.0; spin = [2.0, 0.0, 0.0]; }
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
          ("A", [0.0, 0.0, 0.0]),
          ("B", [0.0, 0.0, 0.916667])
        );
      };
    )";
  }
};

TEST_F(LatticeSizeTest, IntegerSizeBuildsDenseCellMotifMap) {
  initialise_lattice(R"(
    lattice : {
      size = [2, 1, 3];
      periodic = [true, true, true];
    };
  )");

  EXPECT_EQ(globals::lattice->size(), (jams::Vec<int, 3>{2, 1, 3}));
  EXPECT_EQ(globals::lattice->num_cells(), 6u);
  EXPECT_EQ(globals::num_spins, 12);
  EXPECT_EQ(globals::num_spins3, 36);

  int expected_site = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int m = 0; m < 2; ++m) {
          const int site = globals::lattice->site_index_by_unit_cell(i, j, k, m);
          EXPECT_EQ(site, expected_site);
          EXPECT_EQ(globals::lattice->cell_offset(site), (jams::Vec<int, 3>{i, j, k}));
          EXPECT_EQ(globals::lattice->lattice_site_basis_index(site), static_cast<unsigned>(m));
          ++expected_site;
        }
      }
    }
  }

  EXPECT_EQ(globals::lattice->get_supercell().a1(), (jams::Vec<double, 3>{2.0, 0.0, 0.0}));
  EXPECT_EQ(globals::lattice->get_supercell().a2(), (jams::Vec<double, 3>{0.0, 1.0, 0.0}));
  EXPECT_EQ(globals::lattice->get_supercell().a3(), (jams::Vec<double, 3>{0.0, 0.0, 3.0}));
}

TEST_F(LatticeSizeTest, IntegerSizeSpinArrayIsDenseSpatialFftInput) {
  initialise_lattice(R"(
    lattice : {
      size = [2, 1, 2];
      periodic = [true, true, true];
      normalise_spins = false;
    };
  )");

  jams::MultiArray<jams::Vec<std::complex<double>, 3>, 4> sk;
  fft_supercell_vector_field_to_kspace(
      globals::s,
      sk,
      globals::lattice->size(),
      globals::lattice->kspace_size(),
      globals::lattice->num_basis_sites());

  const double expected_scale = std::sqrt(static_cast<double>(jams::product(globals::lattice->size())));
  EXPECT_NEAR(sk(0, 0, 0, 0)[0].real(), expected_scale * 1.0, 1e-12);
  EXPECT_NEAR(sk(0, 0, 0, 0)[0].imag(), 0.0, 1e-12);
  EXPECT_NEAR(sk(0, 0, 0, 1)[0].real(), expected_scale * 2.0, 1e-12);
  EXPECT_NEAR(sk(0, 0, 0, 1)[0].imag(), 0.0, 1e-12);

  for (int i = 0; i < sk.extent(0); ++i) {
    for (int j = 0; j < sk.extent(1); ++j) {
      for (int k = 0; k < sk.extent(2); ++k) {
        if (i == 0 && j == 0 && k == 0) {
          continue;
        }
        for (int m = 0; m < sk.extent(3); ++m) {
          EXPECT_NEAR(std::abs(sk(i, j, k, m)[0]), 0.0, 1e-12);
          EXPECT_NEAR(std::abs(sk(i, j, k, m)[1]), 0.0, 1e-12);
          EXPECT_NEAR(std::abs(sk(i, j, k, m)[2]), 0.0, 1e-12);
        }
      }
    }
  }
}

TEST_F(LatticeSizeTest, LargePeriodicSupercellKeepsHighMotifInLastCell) {
  initialise_lattice_with_high_z_motif(R"(
    lattice : {
      size = [1, 1, 1024];
      periodic = [true, true, true];
      normalise_spins = false;
    };
  )");

  EXPECT_EQ(globals::lattice->size(), (jams::Vec<int, 3>{1, 1, 1024}));
  EXPECT_EQ(globals::num_spins, 2048);
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 1023, 0));
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 1023, 1));

  const int high_z_site = globals::lattice->site_index_by_unit_cell(0, 0, 1023, 1);
  EXPECT_EQ(globals::lattice->cell_offset(high_z_site), (jams::Vec<int, 3>{0, 0, 1023}));
  EXPECT_EQ(globals::lattice->lattice_site_basis_index(high_z_site), 1u);
}

TEST_F(LatticeSizeTest, LargeUnitCellVectorsKeepHighMotifInsideExtent) {
  initialise_lattice_with_large_unitcell(R"(
    lattice : {
      size = [1, 1, 1];
      periodic = [true, true, true];
      normalise_spins = false;
    };
  )");

  EXPECT_EQ(globals::lattice->size(), (jams::Vec<int, 3>{1, 1, 1}));
  EXPECT_EQ(globals::num_spins, 2);
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 0, 0));
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 0, 1));
  EXPECT_EQ(globals::lattice->get_supercell().a3(), (jams::Vec<double, 3>{0.0, 0.0, 1000000.0}));
}

TEST_F(LatticeSizeTest, PeriodicBoundaryConditionsWrapMultipleCells) {
  initialise_lattice(R"(
    lattice : {
      size = [3, 4, 5];
      periodic = [true, true, true];
    };
  )");

  jams::Vec<int, 3> pos{-7, 10, -11};
  EXPECT_TRUE(globals::lattice->apply_boundary_conditions(pos));
  EXPECT_EQ(pos, (jams::Vec<int, 3>{2, 2, 4}));

  int a = 7;
  int b = -10;
  int c = 16;
  EXPECT_TRUE(globals::lattice->apply_boundary_conditions(a, b, c));
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 1);
}

TEST_F(LatticeSizeTest, OpenBoundaryConditionsRejectOutOfRangeBeforeWrapping) {
  initialise_lattice(R"(
    lattice : {
      size = [3, 4, 5];
      periodic = [true, false, true];
    };
  )");

  jams::Vec<int, 3> pos{-4, -1, 11};
  EXPECT_FALSE(globals::lattice->apply_boundary_conditions(pos));
  EXPECT_EQ(pos, (jams::Vec<int, 3>{2, -1, 11}));

  int a = -4;
  int b = 4;
  int c = 11;
  EXPECT_FALSE(globals::lattice->apply_boundary_conditions(a, b, c));
  EXPECT_EQ(a, 2);
  EXPECT_EQ(b, 4);
  EXPECT_EQ(c, 11);
}

TEST_F(LatticeSizeTest, NonIntegerSizeCropsUpperLatticePlane) {
  initialise_lattice_with_z_motif(R"(
    lattice : {
      size = [1.0, 1.0, 2.5];
      periodic = [true, true, false];
      normalise_spins = false;
    };
  )");

  EXPECT_EQ(globals::lattice->size(), (jams::Vec<int, 3>{1, 1, 3}));
  EXPECT_EQ(globals::lattice->extent(), (jams::Vec<double, 3>{1.0, 1.0, 2.5}));
  EXPECT_TRUE(globals::lattice->has_cropping());
  EXPECT_TRUE(globals::lattice->is_cropped(2));
  EXPECT_EQ(globals::lattice->num_cells(), 3u);
  EXPECT_EQ(globals::num_spins, 5);
  EXPECT_EQ(globals::num_spins3, 15);

  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 0, 0));
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 0, 1));
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 1, 0));
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 1, 1));
  EXPECT_TRUE(globals::lattice->has_site_at_unit_cell(0, 0, 2, 0));
  EXPECT_FALSE(globals::lattice->has_site_at_unit_cell(0, 0, 2, 1));
  EXPECT_THROW(globals::lattice->site_index_by_unit_cell(0, 0, 2, 1), jams::SanityException);

  EXPECT_EQ(globals::lattice->get_supercell().a1(), (jams::Vec<double, 3>{1.0, 0.0, 0.0}));
  EXPECT_EQ(globals::lattice->get_supercell().a2(), (jams::Vec<double, 3>{0.0, 1.0, 0.0}));
  EXPECT_EQ(globals::lattice->get_supercell().a3(), (jams::Vec<double, 3>{0.0, 0.0, 2.5}));
  EXPECT_EQ(globals::lattice->rmax(), (jams::Vec<double, 3>{1.0, 1.0, 2.5}));
}

TEST_F(LatticeSizeTest, NonIntegerSizeRequiresOpenBoundaryInCroppedDirection) {
  globals::config->readString(base_z_motif_config() + R"(
    lattice : {
      size = [1.0, 1.0, 2.5];
      periodic = [true, true, true];
      normalise_spins = false;
    };
  )");

  EXPECT_THROW(globals::lattice->init_from_config(*globals::config), jams::ConfigException);
}

TEST_F(LatticeSizeTest, CroppedLatticeSpinArrayIsPackedForSpatialFft) {
  initialise_lattice_with_z_motif(R"(
    lattice : {
      size = [1.0, 1.0, 2.5];
      periodic = [true, true, false];
      normalise_spins = false;
    };
  )");

  jams::MultiArray<jams::Vec<std::complex<double>, 3>, 4> sk;
  fft_lattice_vector_field_to_kspace(globals::s, sk, *globals::lattice);

  ASSERT_EQ(sk.extent(0), 1);
  ASSERT_EQ(sk.extent(1), 1);
  ASSERT_EQ(sk.extent(2), 4);
  ASSERT_EQ(sk.extent(3), 2);

  const double expected_scale = std::sqrt(static_cast<double>(jams::product(globals::lattice->kspace_size())));
  EXPECT_NEAR(sk(0, 0, 0, 0)[0].real(), 3.0 / expected_scale, 1e-12);
  EXPECT_NEAR(sk(0, 0, 0, 0)[0].imag(), 0.0, 1e-12);
  EXPECT_NEAR(sk(0, 0, 0, 1)[0].real(), 4.0 / expected_scale, 1e-12);
  EXPECT_NEAR(sk(0, 0, 0, 1)[0].imag(), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(sk(0, 0, 0, 0)[1]), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(sk(0, 0, 0, 0)[2]), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(sk(0, 0, 0, 1)[1]), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(sk(0, 0, 0, 1)[2]), 0.0, 1e-12);
}

}  // namespace jams::testing

#endif  // JAMS_TEST_LATTICE_SIZE_H
