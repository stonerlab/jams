//
// Created by ioannis charalampidis on 17/08/2022.
//

#ifndef JAMS_SRC_JAMS_MONITORS_TOPOLOGICAL_CHARGE_GEOMETRICAL_DEF_H_
#define JAMS_SRC_JAMS_MONITORS_TOPOLOGICAL_CHARGE_GEOMETRICAL_DEF_H_

#include <jams/core/monitor.h>
#include <fstream>
#include <vector>
#include <libconfig.h++>

/// @class TopologicalFiniteDiffChargeMonitor
///
/// Calculates the topological charge using geometrical definition.
///
/// @details
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
/// @setting 'radius_cutoff' (optional)
///
/// @example
/// @code
/// monitors = (
///   {
///     module = "topological-charge-geometrical-def";
///     output_steps = 10000;
///     max_threshold = 1.0;
///     min_threshold = -1.0;
///     convergence = -1.0;
//    }
///  );
/// @endcode


class TopologicalGeometricalDefMonitor : public Monitor {
 public:
  TopologicalGeometricalDefMonitor(const libconfig::Setting &settings);
  ~TopologicalGeometricalDefMonitor() override = default;

  void update(Solver *solver) override;
  void post_process() override {};

  bool is_converged() override;

 private:
  struct Triplet {
	int i;
	int j;
	int k;
  };
  struct TripletHasher {
	std::size_t operator()(Triplet a) const {
	  std::hash<int> hasher;
	  return hasher(a.i) + hasher(a.j) + hasher(a.k);
	}

  };
  struct HandedTripletComparator {
	bool operator()(const Triplet &a, const Triplet &b) const {
	  return
		// clockwise
		  ((a.i == b.i && a.j == b.j && a.k == b.k)
			  || (a.i == b.j && a.j == b.k && a.k == b.i)
			  || (a.i == b.k && a.j == b.i && a.k == b.j)
		  ) || ( // anti-clockwise
			  (a.i == b.i && a.j == b.j && a.k == b.k)
				  || (a.i == b.k && a.j == b.j && a.k == b.i)
				  || (a.i == b.j && a.j == b.i && a.k == b.k)
		  );
	}
  };

  void calculate_elementary_triangles();
  double local_topological_charge(const Vec3& s_i, const Vec3& s_j, const Vec3& s_k) const;
  double local_topological_charge(const Triplet &t) const;


  double total_topological_charge() const;


  double radius_cutoff_ = 1.0;
  double recip_num_layers_ = 1.0;

  // Vector is num_spins long, first index is spin index, the sub-vector
  // contains an integer pointer to triangles which include this spin index.
  std::vector<std::vector<int>> adjacent_triangles_;
  std::vector<Triplet> triangle_indices_;

//  static std::string tsv_header;
  std::string tsv_header();
  std::string name_ = "topological-charge-geometrical-def";
  std::ofstream outfile;

  double monitor_top_charge_cache_ = 0.0;

  double max_tolerance_threshold_ = 1.0;
  double min_tolerance_threshold_ = -1.0;

};

#endif //JAMS_SRC_JAMS_MONITORS_TOPOLOGICAL_CHARGE_GEOMETRICAL_DEF_H_
