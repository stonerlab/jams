include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)

add_executable(jams "main.cc")
target_link_libraries(jams
        fftw_external
        cblas_external
        hdf5_external
        pcg_builtin
        spglib_builtin
        libconfig_builtin
        highfive_builtin)

target_compile_features(jams PRIVATE cxx_std_11)

target_include_directories(jams PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

if (MKL_FOUND)
    target_compile_definitions(jams PRIVATE HAS_MKL=1)
endif()

if (JAMS_BUILD_OMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(jams PRIVATE HAS_OMP=1)
    target_compile_options(jams PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(jams OpenMP::OpenMP_CXX)
endif()

if(JAMS_BUILD_CUDA)
    target_compile_definitions(jams PRIVATE HAS_CUDA=1)
    target_link_libraries(jams
            cuda
            cusparse
            curand
            cublas
            cufft)

    if (JAMS_BUILD_MIXED_PREC)
        target_compile_definitions(jams PRIVATE HAS_CUSPARSE_MIXED_PREC=1)
    endif()

    if(APPLE)
        # We need to add the path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_target_properties(jams PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endif()

if (JAMS_BUILD_FASTMATH)
    jams_set_fast_math(jams)
endif()

target_sources(jams
        PRIVATE
        containers/cell.cc
        core/hamiltonian.cc
        core/interactions.cc
        core/jams++.cc
        core/lattice.cc
        core/monitor.cc
        core/physics.cc
        core/random.cc
        core/solver.cc
        core/thermostat.cc
        hamiltonian/dipole.cc
        hamiltonian/dipole_bruteforce.cc
        hamiltonian/dipole_ewald.cc
        hamiltonian/dipole_fft.cc
        hamiltonian/dipole_tensor.cc
        hamiltonian/exchange.cc
        hamiltonian/exchange_neartree.cc
        hamiltonian/uniaxial_anisotropy.cc
        hamiltonian/uniaxial_microscopic_anisotropy.cc
        hamiltonian/random_anisotropy.cc
        hamiltonian/zeeman.cc
        helpers/error.cc
        interface/fft.cc
        helpers/maths.cc
        helpers/neutrons.cc
        helpers/output.cc
        helpers/slice.cc
        helpers/stats.cc
        helpers/utils.cc
        interface/config.cc
        monitors/binary.cc
        monitors/boltzmann.cc
        monitors/energy.cc
        monitors/hdf5.cc
        monitors/field.cc
        monitors/magnetisation.cc
        monitors/magnetisation_rate.cc
        monitors/neutron_scattering.cc
        monitors/neutron_scattering_no_lattice.cc
        monitors/skyrmion.cc
        monitors/smr.cc
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
        physics/square_field_pulse.cc
        physics/two_temperature_model.cc
        solvers/cpu_llg_heun.cc
        solvers/cpu_rotations.cc
        solvers/cpu_monte_carlo_constrained.cc
        solvers/cpu_monte_carlo_metropolis.cc)

if(JAMS_BUILD_CUDA)
    target_sources(jams
            PRIVATE
            cuda/cuda_array_kernels.cu
            cuda/cuda_solver.cc
            cuda/cuda_common.cu
            hamiltonian/cuda_uniaxial_anisotropy.cu
            hamiltonian/cuda_uniaxial_microscopic_anisotropy.cu
            hamiltonian/cuda_dipole_fft.cu
            hamiltonian/cuda_exchange.cu
            hamiltonian/cuda_exchange_neartree.cu
            hamiltonian/cuda_random_anisotropy.cu
            hamiltonian/cuda_zeeman.cu
            hamiltonian/cuda_dipole_bruteforce.cu
            hamiltonian/cuda_dipole_sparse_tensor.cu
            monitors/cuda_spin_current.cc
            monitors/cuda_spin_current_kernel.cu
            monitors/cuda_thermal_current.cc
            monitors/cuda_thermal_current_kernel.cu
            solvers/cuda_llg_heun.cu
            thermostats/cuda_langevin_bose.cu
            thermostats/cuda_langevin_white.cc)
endif()
