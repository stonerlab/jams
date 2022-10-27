include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)

set(JAMS_SOURCES_CXX
        common.cc
        containers/cell.cc
        core/args.cc
        core/hamiltonian.cc
        core/interactions.cc
        core/jams++.cc
        core/lattice.cc
        core/monitor.cc
        core/physics.cc
        core/solver.cc
        core/thermostat.cc
        hamiltonian/applied_field.cc
        hamiltonian/cubic_anisotropy.cc
        hamiltonian/dipole_bruteforce.cc
        hamiltonian/dipole_fft.cc
        hamiltonian/dipole_neartree.cc
        hamiltonian/dipole_neighbour_list.cc
        hamiltonian/dipole_tensor.cc
        hamiltonian/exchange.cc
        hamiltonian/exchange_functional.cc
        hamiltonian/exchange_neartree.cc
        hamiltonian/field_pulse.cc
        hamiltonian/random_anisotropy.cc
        hamiltonian/sparse_interaction.cc
        hamiltonian/uniaxial_anisotropy.cc
        hamiltonian/uniaxial_microscopic_anisotropy.cc
        hamiltonian/zeeman.cc
        helpers/error.cc
        helpers/interaction_calculator.cc
        helpers/maths.cc
        helpers/neutrons.cc
        helpers/output.cc
        helpers/slice.cc
        helpers/spinops.cc
        helpers/stats.cc
        helpers/utils.cc
        initializer/init_dispatcher.cc
        initializer/init_bloch_domain_wall.cc
        initializer/init_h5.cc
        initializer/init_skyrmion.cc
        interface/config.cc
        interface/fft.cc
        interface/system.cc
        metadynamics/collective_variable_factory.cc
        metadynamics/metadynamics_potential.cc
        metadynamics/collective_variable.cc
        metadynamics/caching_collective_variable.cc
        metadynamics/cvars/cvar_magnetisation.cc
        metadynamics/cvars/cvar_skyrmion_center_coordinate.cc
        metadynamics/cvars/cvar_topological_charge.cc
        metadynamics/cvars/cvar_topological_charge_finite_diff.cc
        lattice/minimum_image.cc
        lattice/interaction_neartree.cc
        maths/functions.cc
        maths/parallelepiped.cc
        maths/interpolation.cc
        monitors/binary.cc
        monitors/boltzmann.cc
        monitors/energy.cc
        monitors/field.cc
        monitors/topological_charge_finite_diff.cc
        monitors/topological_charge_geometrical_def.cc
        monitors/hdf5.cc
        monitors/magnetisation.cc
        monitors/magnetisation_layers.cc
        monitors/magnetisation_rate.cc
        monitors/magnon_spectrum.cc
        monitors/neutron_scattering.cc
        monitors/neutron_scattering_no_lattice.cc
        monitors/skyrmion.cc
        monitors/smr.cc
        monitors/spectrum_base.cc
        monitors/spectrum_fourier.cc
        monitors/spectrum_general.cc
        monitors/spin_correlation.cc
        monitors/spin_pumping.cc
        monitors/spin_temperature.cc
        monitors/torque.cc
        monitors/unitcell_average.cc
        monitors/vtu.cc
        monitors/xyz.cc
        physics/field_cool.cc
        physics/flips.cc
        physics/fmr.cc
        physics/mean_first_passage_time.cc
        physics/ping.cc
        physics/pinned_boundaries.cc
        physics/square_field_pulse.cc
        physics/two_temperature_model.cc
        solvers/cpu_llg_heun.cc
        solvers/cpu_metadynamics_metropolis_solver.cc
        solvers/cpu_monte_carlo_constrained.cc
        solvers/cpu_monte_carlo_metropolis.cc
        solvers/cpu_rotations.cc
        version.cc)

set(JAMS_SOURCES_CUDA
        cuda/cuda_array_kernels.cu
        cuda/cuda_array_reduction.cu
        cuda/cuda_common.cu
        cuda/cuda_solver.cc
        cuda/cuda_spin_ops.cu
        cuda/cuda_minimum_image.cu
        hamiltonian/cuda_applied_field.cu
        hamiltonian/cuda_cubic_anisotropy.cu
        hamiltonian/cuda_dipole_bruteforce.cu
        hamiltonian/cuda_dipole_fft.cu
        hamiltonian/cuda_field_pulse.cu
        hamiltonian/cuda_random_anisotropy.cu
        hamiltonian/cuda_uniaxial_anisotropy.cu
        hamiltonian/cuda_uniaxial_microscopic_anisotropy.cu
        hamiltonian/cuda_zeeman.cu
        monitors/cuda_spin_current.cc
        monitors/cuda_spin_current_kernel.cu
        monitors/cuda_thermal_current.cc
        monitors/cuda_thermal_current_kernel.cu
        monitors/cuda_neutron_scattering_no_lattice.cu
        solvers/cuda_llg_heun.cu
        solvers/cuda_llg_rk4.cu
        solvers/cuda_ll_lorentzian_rk4.cu
        thermostats/thm_bose_einstein_cuda_srk4.cu
        thermostats/thm_bose_einstein_cuda_srk4_kernel.cuh
        thermostats/cuda_langevin_bose.cu
        thermostats/cuda_lorentzian.cu
        thermostats/cuda_langevin_white.cc)