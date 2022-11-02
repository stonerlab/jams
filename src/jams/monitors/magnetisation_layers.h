// magnetisation_layers.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_MONITORS_MAGNETISATION_LAYERS
#define INCLUDED_JAMS_MONITORS_MAGNETISATION_LAYERS

/// @class MagnetisationLayersMonitor
///
/// Calculates the net magnetisation in each layer of the system with respect
/// to a given layer normal vector.
///
/// @details
/// This monitor is designed for calculating the layer-wise magnetisation even
/// in systems which consist of a single materials. This is useful for example
/// in finite systems where the surfaces may have different properties or for
/// modelling domain walls where we need the magnetisation profile across the
/// system. Output from this monitor will be substantially smaller than
/// outputting the whole spin system and calculating the magnetisation as a
/// post-process.
///
/// The magnetisation is the  total magnetisation (not normalised by
/// number of spins) in units of Bohr magnetons. No transformations are applied
/// to anti-parallel moments will cancel each other. If different layers have
/// different numbers of atoms (but the same magnetic moment) then the total
/// magnetisation of those layers will be different.
///
/// The data is written to a h5 file "<simulation_name>_monitor.h5" in the
/// group "/jams/monitors/magnetisation_layers". The group contains the
/// datasets:
///
/// - num_layers:       The number of layers calculated
///                     (shape = [1], type = int)
/// - layer_normal:     The normal vector along which the layers are defined
///                     (shape = [3], type = double)
/// - layer_positions:  The positions of the layers along the normal
///                     (shape = [num_layers], type = double)
/// - layer_spin_count: Number of spins in each layer
///                     (shape = [num_layers], type = int)
///
/// For each output a new group is made in /jams/monitors/magnetisation_layers
/// with the solver iteration as the group name (zero padded of length 9).
/// e.g. /jams/monitors/magnetisation_layers/000000000 will be the first output.
/// Each of these groups has the attribute "time" which is the solver time
/// at the output iteration. The group contains the datasets:
///
/// - layer_magnetisation: Total magnetisation of each layer
///                        (shape = [num_layers, 3], type = double)
///
/// The layers are calculated with respect to a configured normal vector. In
/// most cases this will be something simple like [1, 0, 0] for layers along
/// the x-axis. In principle it can be any general vector though such as
/// [1, 1, 1] to get the magnetisation along the [1, 1, 1] planes.
/// the layers are determined directly from the atomic positions without any
/// knowledge of the unit cell, i.e. the magnetisation in a layer is not per
/// unit cell but really just the atoms in a given layer. It also means that for
/// non-crystalline unit cells a large number of layers, possibly very close
/// together may be found. The monitor uses jams::defaults::lattice_tolerance
/// to decide if two atoms are in the same layer.
///
/// @setting `layer_normal` (required) normal vector along which to calculate
///           layers.
///
/// @example
/// @code
/// monitors = (
///   {
///   module = "magnetisation-layers";
///   output_steps = 1000;
///   layer_normal = [1, 0, 0];
///   }
/// );
/// @endcode


#include <jams/core/monitor.h>
#include <jams/interface/config.h>
#include <jams/containers/multiarray.h>

#include <vector>

class Solver;

class MagnetisationLayersMonitor : public Monitor {
public:
    explicit MagnetisationLayersMonitor(const libconfig::Setting &settings);

    ~MagnetisationLayersMonitor() override = default;

    void update(Solver *solver) override;

    inline void post_process() override {};

private:
    std::string h5_group = "/jams/monitors/magnetisation_layers/";

    int num_layers_;
    jams::MultiArray<double,2>           layer_magnetisation_;
    std::vector<jams::MultiArray<int,1>> layer_spin_indicies_;
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------