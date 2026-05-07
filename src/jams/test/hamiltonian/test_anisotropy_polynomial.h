#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/anisotropy_polynomial.h"
#include "jams/hamiltonian/cuda_anisotropy_polynomial.h"
#include "jams/helpers/utils.h"
#include "jams/test/output.h"

class CudaAnisotropyPolynomialHamiltonianTests : public ::testing::Test {
public:
    void SetUp() override
    {
        cudaDeviceReset();
        jams::testing::toggle_cout();

        globals::lattice = new Lattice();
        globals::config = std::make_unique<libconfig::Config>();
        globals::config->readString(config_string());
        globals::lattice->init_from_config(*globals::config);

        cpu_hamiltonian_ = std::make_unique<AnisotropyPolynomialHamiltonian>(
            globals::config->lookup("hamiltonians.[0]"),
            globals::num_spins);
        cuda_hamiltonian_ = std::make_unique<CudaAnisotropyPolynomialHamiltonian>(
            globals::config->lookup("hamiltonians.[0]"),
            globals::num_spins);

        jams::testing::toggle_cout();
    }

    void TearDown() override
    {
        cpu_hamiltonian_ = nullptr;
        cuda_hamiltonian_ = nullptr;

        globals::num_spins = 0;
        globals::num_spins3 = 0;

        jams::util::force_deallocation(globals::s);
        jams::util::force_deallocation(globals::h);
        jams::util::force_deallocation(globals::ds_dt);
        jams::util::force_deallocation(globals::positions);
        jams::util::force_deallocation(globals::alpha);
        jams::util::force_deallocation(globals::mus);
        jams::util::force_deallocation(globals::gyro);

        globals::config = nullptr;

        if (globals::lattice) {
            delete globals::lattice;
            globals::lattice = nullptr;
        }
    }

protected:
    static std::string config_string()
    {
        return R"(
            solver : {
              module = "llg-heun-gpu";
              t_step = 1.0e-16;
              t_min = 1.0e-16;
              t_max = 1.0e-16;
            };

            materials = (
              { name = "A"; moment = 2.0; spin = [1.0, 0.0, 0.0]; },
              { name = "B"; moment = 1.5; spin = [0.0, 1.0, 0.0]; }
            );

            unitcell : {
              parameter = 0.3e-9;
              basis = (
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]);
              positions = (
                ("A", [0.0, 0.0, 0.0]),
                ("B", [0.5, 0.0, 0.0])
              );
            };

            lattice : {
              size = [2, 1, 1];
              periodic = [false, false, false];
            };

            hamiltonians = ({
              module = "anisotropy-polynomial";
              energy_units = "meV";
              normalisation = "racah";
              anisotropies = (
                ("A", [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                  (2, -2, 0.31),
                  (2, -1, -0.17),
                  (2, 0, 0.23),
                  (2, 1, -0.29),
                  (2, 2, 0.11),
                  (4, -4, -0.07),
                  (4, -3, 0.13),
                  (4, -2, -0.19),
                  (4, -1, 0.05),
                  (4, 0, -0.03),
                  (4, 1, -0.2),
                  (4, 2, 0.09),
                  (4, 3, -0.15),
                  (4, 4, 0.21),
                  (6, -6, 0.04),
                  (6, -5, -0.06),
                  (6, -4, 0.08),
                  (6, -3, -0.1),
                  (6, -2, 0.12),
                  (6, -1, -0.14),
                  (6, 0, 0.16),
                  (6, 1, -0.18),
                  (6, 2, 0.2),
                  (6, 3, -0.22),
                  (6, 4, 0.24),
                  (6, 5, -0.26),
                  (6, 6, 0.05)),
                (2, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0],
                  (2, 2, -0.1),
                  (4, -3, 0.07),
                  (6, 0, 0.02))
              );
            });
        )";
    }

    void set_test_spins()
    {
        const double inv_sqrt_3 = 1.0 / std::sqrt(3.0);
        const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
        const double z0 = std::sqrt(1.0 - 0.2 * 0.2 - 0.3 * 0.3);

        const double spins[4][3] = {
            {1.0, 0.0, 0.0},
            {0.2, -0.3, z0},
            {-inv_sqrt_2, inv_sqrt_2, 0.0},
            {inv_sqrt_3, inv_sqrt_3, inv_sqrt_3}
        };

        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                globals::s(i, j) = spins[i][j];
            }
        }
    }

    std::unique_ptr<AnisotropyPolynomialHamiltonian> cpu_hamiltonian_;
    std::unique_ptr<CudaAnisotropyPolynomialHamiltonian> cuda_hamiltonian_;
};

TEST_F(CudaAnisotropyPolynomialHamiltonianTests, factory_selects_cuda_variant)
{
    std::unique_ptr<Hamiltonian> hamiltonian(
        Hamiltonian::create(globals::config->lookup("hamiltonians.[0]"), globals::num_spins, true));

    ASSERT_NE(dynamic_cast<CudaAnisotropyPolynomialHamiltonian*>(hamiltonian.get()), nullptr);
}

TEST_F(CudaAnisotropyPolynomialHamiltonianTests, energies_and_fields_match_cpu)
{
    set_test_spins();

    cpu_hamiltonian_->calculate_energies(0.0);
    cuda_hamiltonian_->calculate_energies(0.0);

    const double tolerance = 5e-6;
    for (int i = 0; i < globals::num_spins; ++i) {
        ASSERT_NEAR(cuda_hamiltonian_->energy(i), cpu_hamiltonian_->energy(i), tolerance);
    }

    const double cuda_total = cuda_hamiltonian_->calculate_total_energy(0.0);
    const double cpu_total = cpu_hamiltonian_->calculate_total_energy(0.0);
    ASSERT_NEAR(cuda_total, cpu_total, tolerance);

    cpu_hamiltonian_->calculate_fields(0.0);
    cuda_hamiltonian_->calculate_fields(0.0);

    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_NEAR(cuda_hamiltonian_->field(i, j), cpu_hamiltonian_->field(i, j), tolerance);
        }
    }
}
