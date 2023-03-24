//
// Created by ioannis charalampidis on 02/08/2022.
//

#ifndef INCLUDED_JAMS_MONITORS_TOPOLOGICAL_CHARGE_FINITE_DIFF
#define INCLUDED_JAMS_MONITORS_TOPOLOGICAL_CHARGE_FINITE_DIFF

#include <jams/core/monitor.h>
#include <jams/containers/multiarray.h>

#include <fstream>
#include <string>
#include <vector>

/// @class TopologicalFiniteDiffChargeMonitor
///
/// Calculates the topological charge using finite differences on the lattice.
///
/// @details
/// This monitor calculates the topological charge from the equation
///
/// Q = (1/4π) ∫ d²x m⃗ ⋅ [(∂m⃗ / ∂x) × (∂m⃗ / ∂y)]
///
/// where the partial differentials are calculated using a finite difference
/// scheme. Currently we assume a specific hexagonal lattice with vectors
/// u₁ = x, u2 = x/2 + √3/2  and the finite difference scheme is
/// (see Rohart, Phys. Rev. B 93, 214412 (2016) Supplementary Info)
///
/// ∂ₓm ≈ [S(rᵢ + u₁ - u₂) + S(rᵢ + u₂) - S(rᵢ - u₁ + u₂) - S(rᵢ - u₂)] / 2
/// ∂ᵧm ≈ [S(rᵢ - u₁ + u₂) + S(rᵢ + u₂) - S(rᵢ + u₁ - u₂) - S(rᵢ - u₂)] / 2√3
///
/// The convergence setting for this monitor allows the simulation to be stopped
/// when the topological charge is outside of a given interval.
///
///     Q <= min_threshold  || Q >= max_threshold
///
/// NOTE: the convergence condition is only updated every 'output_steps'.
///
/// @setting `min_threshold` (optional) when Q is below this value the monitor
///                          is converged (default -1.0)
///
/// @setting `max_threshold` (optional) when Q is above this value the monitor
///                          is converged (default 1.0)
///
/// @example
/// @code
/// monitors = (
///   {
///     module = "topological-charge-finite-diff";
///     output_steps = 100;
///     min_threshold = -1.0;
///     max_threshold = 1.0;
///   }
/// );
/// @endcode

class TopologicalFiniteDiffChargeMonitor : public Monitor {
 public:
  TopologicalFiniteDiffChargeMonitor(const libconfig::Setting &settings);
  ~TopologicalFiniteDiffChargeMonitor() override = default;

  void update(Solver& solver) override;
  void post_process() override {};

  ConvergenceStatus convergence_status() override;

 private:
    enum class Grouping {
        MATERIALS,
        POSITIONS
    };

    std::string tsv_header();

  double local_topological_charge(const int i) const;
  double topological_charge_from_indices(const jams::MultiArray<int,1>& indices) const;

  Grouping grouping_ = Grouping::POSITIONS;
  std::vector<jams::MultiArray<int,1>> group_spin_indicies_;


    std::string name_ = "topological-charge-finite-diff";
  std::ofstream outfile;

  // basically a CSR matrix
  std::vector<std::vector<int>> dx_indices_;
  std::vector<std::vector<double>> dx_values_;

  // basically a CSR matrix
  std::vector<std::vector<int>> dy_indices_;
  std::vector<std::vector<double>> dy_values_;

  std::vector<double> monitor_top_charge_cache_;

  double max_tolerance_threshold_ = 1.0;
  double min_tolerance_threshold_ = -1.0;
};

#endif //INCLUDED_JAMS_MONITORS_TOPOLOGICAL_CHARGE_FINITE_DIFF



