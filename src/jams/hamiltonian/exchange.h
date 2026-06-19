// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <memory>
#include <optional>
#include <string>
#include <jams/containers/interaction_list.h>
#include <jams/core/hamiltonian.h>
#include <jams/core/interactions.h>

enum class ExchangeBackendPolicy {
    Auto,
    Stencil,
    SparseMatrix,
    Benchmark
};

enum class ExchangeBackend {
    Stencil,
    SparseMatrix
};

struct ExchangeBackendSupport {
    bool supported = false;
    std::string reason;
};

class ExchangeBackendImpl;

class ExchangeInteractionSetup {
public:
    ExchangeInteractionSetup(
        const libconfig::Setting &settings,
        double input_energy_unit_conversion,
        bool debug_enabled,
        const std::string& hamiltonian_name);

    const jams::InteractionList<jams::Mat<double, 3, 3>, 2>& neighbour_list() const;

    const std::vector<InteractionData>& interaction_templates() const {
      return interaction_templates_;
    }

    const std::vector<InteractionChecks>& interaction_checks() const {
      return interaction_checks_;
    }

    jams::SparseMatrixSymmetryCheck sparse_matrix_symmetry_check() const {
      return sparse_matrix_symmetry_check_;
    }

    jams::RealHi interaction_prefactor() const {
      return interaction_prefactor_;
    }

    jams::RealHi energy_cutoff() const {
      return energy_cutoff_;
    }

    double input_energy_unit_conversion() const {
      return input_energy_unit_conversion_;
    }

private:
    bool debug_enabled_ = false;
    std::string hamiltonian_name_;
    double input_energy_unit_conversion_ = 1.0;
    std::vector<InteractionData> interaction_templates_;
    std::vector<InteractionChecks> interaction_checks_;
    mutable std::optional<jams::InteractionList<jams::Mat<double, 3, 3>, 2>> neighbour_list_;
    jams::SparseMatrixSymmetryCheck sparse_matrix_symmetry_check_ = jams::SparseMatrixSymmetryCheck::Symmetric;
    jams::RealHi interaction_prefactor_ = 1.0;
    jams::RealHi energy_cutoff_ = 0.0;
    jams::RealHi radius_cutoff_ = 100.0;
    jams::RealHi radius_cutoff_tolerance_ = jams::defaults::lattice_tolerance;
    jams::RealHi distance_tolerance_ = jams::defaults::lattice_tolerance;
};

class ExchangeHamiltonian : public Hamiltonian {
public:
    ExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size, bool is_cuda_solver);
    ExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);
    ~ExchangeHamiltonian() override;

    ExchangeBackend active_backend() const {
      return active_backend_;
    }

    const jams::InteractionList<jams::Mat<double, 3, 3>, 2> &neighbour_list() const;

    jams::Real calculate_total_energy(jams::Real time, const SpinArray& spins) override;
    void calculate_energies(jams::Real time, const SpinArray& spins) override;
    void calculate_fields(jams::Real time, const SpinArray& spins) override;
    bool supports_calculate_fields_in_parallel() const override;
    void calculate_fields_in_parallel(jams::Real time, const SpinArray& spins) override;
    jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;
    jams::Real calculate_energy(int i, jams::Real time) override;
    jams::Real calculate_energy_difference(
        int i,
        const jams::Vec<double, 3>& spin_initial,
        const jams::Vec<double, 3>& spin_final,
        jams::Real time) override;

    EnergyCurrentInteractionSupport energy_current_interaction_support() const override {
      return EnergyCurrentInteractionSupport::Supported;
    }

    void add_energy_current_interactions(jams::EnergyCurrentInteractionSink& sink) const override;

private:
    void initialize_backend(const libconfig::Setting& settings, unsigned int size, bool is_cuda_solver);
    std::unique_ptr<ExchangeBackendImpl> make_sparse_backend(const libconfig::Setting& settings);
    std::unique_ptr<ExchangeBackendImpl> make_stencil_backend(
        const libconfig::Setting& settings,
        bool is_cuda_solver);
    std::unique_ptr<ExchangeBackendImpl> benchmark_and_make_backend(
        const libconfig::Setting& settings,
        bool is_cuda_solver);
    void synchronize_backend_if_device(const ExchangeBackendImpl& backend);
    static ExchangeBackendPolicy parse_backend_policy(const libconfig::Setting& settings);
    static const char* backend_name(ExchangeBackend backend);

    std::shared_ptr<const ExchangeInteractionSetup> setup_;
    std::unique_ptr<ExchangeBackendImpl> backend_;
    ExchangeBackend active_backend_ = ExchangeBackend::SparseMatrix;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H
