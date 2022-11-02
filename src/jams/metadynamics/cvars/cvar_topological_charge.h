// cvar_topological_charge.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE
#define INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE

#include <jams/metadynamics/caching_collective_variable.h>

// ***************************** WARNING *************************************
// CVarTopologicalCharge is based on the "geometrical" definition of topological
// charge and CANNOT be used as a CV in metadynamics because it is strictly an
// integer. We have retained the class purely to be able to demonstrate of it
// that it does not work.
// ***************************** WARNING *************************************

// ---------------------------------------------------------------------------
// config settings
// ---------------------------------------------------------------------------
//
// Settings in collective_variables (standard settings are given in
// MetadynamicsPotential documentation).
//
// -------
//
//  solver : {
//    module = "monte-carlo-metadynamics-cpu";
//    max_steps  = 100000;
//    gaussian_amplitude = 10.0;
//    gaussian_deposition_stride = 100;
//    output_steps = 100;
//
//    collective_variables = (
//      {
//        name = "topological_charge";
//        gaussian_width = 0.1;
//        range_min = -1.05;
//        range_max = 0.05;
//        range_step = 0.01;
//      }
//    );
//  };
//

namespace jams {
class CVarTopologicalCharge : public CachingCollectiveVariable {
public:
    // Simple struct for storing ijk
    struct Triplet {
        int i;
        int j;
        int k;
    };

    // Hash function object which calculates the has of an ijk triplet.
    // Triplets which contains the same indices in any order must have
    // the same hash. This does NOT guarantee they are unique, just that the two
    // triplets MAY contain the same indices and should be compared with the
    // comparator.
    struct TripletHasher {
        std::size_t operator() (Triplet a) const {
          std::hash<int> hasher;
          return hasher(a.i) + hasher(a.j) + hasher(a.k);
        }

    };

    // This comparator returns true if the indices ijk of two triplets are both
    // from the same clockwise or anti-clockwise permutation.
    //
    // A triangle is defined by the vertices i,j,k which follow each other in the
    // order i->j->k->i... in the right handed circle. Therefore ijk, jki and kij
    // are identical in terms of indices and handedness.
    struct HandedTripletComparator {
        bool operator() (const Triplet& a, const Triplet& b) const {
          return
            // clockwise
              (    (a.i == b.i && a.j == b.j && a.k == b.k)
                   || (a.i == b.j && a.j == b.k && a.k == b.i)
                   || (a.i == b.k && a.j == b.i && a.k == b.j)
              ) || ( // anti-clockwise
                  (a.i == b.i && a.j == b.j && a.k == b.k)
                  || (a.i == b.k && a.j == b.j && a.k == b.i)
                  || (a.i == b.j && a.j == b.i && a.k == b.k)
              );
        }
    };


    CVarTopologicalCharge() = default;
    explicit CVarTopologicalCharge(const libconfig::Setting &settings);

    std::string name() override;

    double value() override;

    inline const jams::MultiArray<double, 2>& derivatives() override {
      throw std::runtime_error("unimplemented function");
    };

    /// Returns the value of the collective variable after a trial
    /// spin move from spin_initial to spin_final (to be used with Monte Carlo).
    double spin_move_trial_value(
        int i, const Vec3 &spin_initial, const Vec3 &spin_trial) override;

    double calculate_expensive_value() override;

private:
    std::string name_ = "topo_charge";

    void calculate_elementary_triangles();

    double local_topological_charge(const Vec3& s_i, const Vec3& s_j, const Vec3& s_k) const;
    double local_topological_charge(const Triplet &t) const;

    double topological_charge_difference(int index, const Vec3 &spin_initial, const Vec3 &spin_final) const;


    double total_topological_charge() const;


    double radius_cutoff_ = 1.0;
    double recip_num_layers_ = 1.0;

    // Vector is num_spins long, first index is spin index, the sub-vector
    // contains an integer pointer to triangles which include this spin index.
    std::vector<std::vector<int>> adjacent_triangles_;
    std::vector<Triplet> triangle_indices_;

};
}


#endif //INCLUDED_JAMS_CVAR_TOPOLOGICAL_CHARGE
