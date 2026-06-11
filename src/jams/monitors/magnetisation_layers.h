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
/// in systems which consist of a single material. This is useful for finite
/// systems where the surfaces may have different properties, or for modelling
/// domain walls where the magnetisation profile across the system is needed.
/// Output from this monitor will be substantially smaller than outputting the
/// whole spin system and calculating the magnetisation as a post-process.
/// CUDA builds automatically use a CUDA reduction path when the active solver
/// is a CUDA solver; the public module name and HDF5 layout are unchanged.
///
/// The magnetisation is the total magnetisation (not normalised by number of
/// spins) in units of Bohr magnetons. No transformations are applied, so
/// anti-parallel moments will cancel each other. If different layers have
/// different numbers of atoms, but the same magnetic moment, then the total
/// magnetisation of those layers will be different.
///
/// Static layer metadata is written to the monitor h5 file under
/// "/jams/monitors/<monitor-name>/groups/<group-name>". Each group contains the
/// datasets:
///
/// - num_layers:       The number of layers calculated
///                     (shape = [1], type = int)
/// - layer_normal:     The normal vector along which the layers are defined
///                     (shape = [3], type = double)
/// - layer_thickness:  The configured layer thickness in nm
///                     (shape = [1], type = double)
/// - layer_positions:  The positions of the layer centres along the normal
///                     in nm
///                     (shape = [num_layers], type = double)
/// - layer_saturation_moment:
///                     Sum of spin moments in each layer in Bohr magnetons
///                     (shape = [num_layers], type = double)
/// - layer_spin_count: Number of spins in each layer
///                     (shape = [num_layers], type = int)
///
/// For ParaView visualisation, a sidecar XDMF file is written alongside the H5
/// file. By default it contains exact layer-centre slice faces only. The
/// optional `xdmf_outputs` setting selects any combination of "slice", "glyph",
/// and "volume"; an empty list disables ParaView geometry while keeping the
/// core HDF5 layer data. Static geometry for the selected outputs is stored
/// under "/jams/monitors/<monitor-name>/groups/<group-name>/xdmf":
///
/// - volume_points:    XYZ vertices for clipped layer volumes, in nm
///                     (shape = [num_volume_points, 3], type = double)
/// - volume_tetrahedra:
///                     Tetrahedron connectivity for clipped layer volumes
///                     (shape = [num_volume_tetrahedra, 4], type = int)
/// - volume_tetra_layer_index:
///                     Layer index for each volume tetrahedron
///                     (shape = [num_volume_tetrahedra], type = int)
/// - slice_points:     XYZ vertices for exact layer-centre slice faces, in nm
///                     (shape = [num_slice_points, 3], type = double)
/// - slice_triangles:  Triangle connectivity for exact layer-centre slice faces
///                     (shape = [num_slice_triangles, 3], type = int)
/// - slice_triangle_layer_index:
///                     Layer index for each slice triangle
///                     (shape = [num_slice_triangles], type = int)
/// - glyph_points:     One XYZ point per layer for arrow glyph filters
///                     (shape = [num_layers, 3], type = double)
///
/// For each output a new group is made in
/// "/jams/monitors/<monitor-name>/timeseries/<iteration>", with the solver
/// iteration zero padded to length 9. For example,
/// "/jams/monitors/magnetisation-layers/timeseries/000000000" is the first
/// output for the default monitor name. Each of these groups has the attributes
/// "time", "time_step", and "units". Per-spin-group data is written below that
/// output group:
///
/// - <group-name>/magnetisation:
///                     Total magnetisation of each layer in Bohr magnetons
///                     (shape = [num_layers, 3], type = double)
///
/// The layers are calculated with respect to a configured normal vector. In
/// most cases this will be something simple like [1, 0, 0] for layers along
/// the x-axis. In principle it can be any general vector though such as
/// [1, 1, 1] to get the magnetisation along the [1, 1, 1] planes. The layers
/// are determined directly from the atomic positions without any
/// knowledge of the unit cell, i.e. the magnetisation in a layer is not per
/// unit cell but really just the atoms in a given layer. It also means that for
/// non-crystalline unit cells a large number of layers, possibly very close
/// together may be found.
///
/// @setting `layer_normal` (required) normal vector along which to calculate
///           layers.
/// @setting `layer_thickness` (optional) layer thickness in nm. The default is
///           0.0, which groups atoms into zero-thickness layers using
///           distance_tolerance.
/// @setting `distance_tolerance` (optional) tolerance in nm for zero-thickness
///           layer grouping and finite-thickness boundary snapping. The default
///           is jams::defaults::lattice_tolerance converted from lattice
///           parameter units to nm.
/// @setting `xdmf_outputs` (optional) array/list containing "slice", "glyph",
///           and/or "volume". The default is ["slice"]. An empty list disables
///           ParaView geometry and XDMF grids.
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
#include <jams/monitors/spin_grouping.h>

#include <memory>
#include <vector>
#include <string>

class Solver;

class MagnetisationLayersMonitor : public Monitor {
public:
    explicit MagnetisationLayersMonitor(const libconfig::Setting &settings);

    ~MagnetisationLayersMonitor() override;

    void update(Solver& solver) override;

    inline void post_process() override {};

private:
    struct XdmfTimeStep {
        int iteration = 0;
        double time = 0.0;
    };

    struct XdmfOutputSelection {
        bool volume = false;
        bool slice = true;
        bool glyph = false;

        bool any() const { return volume || slice || glyph; }
    };

    void accumulate_layer_magnetisation_cpu();
    void write_xdmf_file() const;
    void append_xdmf_time_step(const Solver& solver);

#if HAS_CUDA
    struct CudaBackend;

    void prepare_cuda_backend_from_cpu_indices();
    void accumulate_layer_magnetisation_cuda();
#endif

    jams::monitors::SpinGrouping grouping_ = jams::monitors::SpinGrouping::NONE;

    std::string h5_group_root_name_;
    std::string h5_file_name_;
    std::string xdmf_file_name_;
    XdmfOutputSelection xdmf_outputs_;

    std::vector<jams::monitors::SpinGroup> spin_groups_;
    std::vector<int> group_num_layers_;
    std::vector<XdmfTimeStep> xdmf_time_steps_;
    std::vector<std::vector<double>> group_layer_positions_;
    std::vector<std::vector<double>> group_layer_saturation_moment_;
    std::vector<std::vector<int>> group_layer_spin_count_;
    std::vector<std::vector<int>> group_volume_tetra_layer_indices_;
    std::vector<std::vector<int>> group_slice_triangle_layer_indices_;
    std::vector<int> group_volume_point_counts_;
    std::vector<int> group_slice_point_counts_;
    std::vector<jams::MultiArray<double,2>>           group_layer_magnetisation_;
    std::vector<jams::MultiArray<int,1>>              group_spin_layer_indices_;

#if HAS_CUDA
    std::unique_ptr<CudaBackend> cuda_backend_;
#endif
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------
