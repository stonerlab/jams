add_library(jams_object_lib OBJECT EXCLUDE_FROM_ALL "")
target_compile_features(jams_object_lib PRIVATE cxx_std_11)
target_compile_definitions(jams_object_lib PRIVATE HAS_CUDA=1)
target_include_directories(jams_object_lib PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)
target_include_directories(jams_object_lib PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

# object libraries cannot use target_link_libraries to pull in include and compile definitions
# this hack allows us to pull them in manually
foreach(TGT fftw3 cblas jblib pcg symspg config++ hdf5 cuda cusparse curand cublas cufft)
    target_include_directories(jams_object_lib PRIVATE $<TARGET_PROPERTY:${TGT},INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(jams_object_lib PRIVATE $<TARGET_PROPERTY:${TGT},INTERFACE_COMPILE_DEFINITIONS>)
endforeach()

add_executable(jams "main.cc" $<TARGET_OBJECTS:jams_object_lib>)
target_link_libraries(jams fftw3 cblas jblib pcg symspg config++ hdf5 cuda cusparse curand cublas cufft)
target_compile_features(jams PRIVATE cxx_std_11)
target_compile_definitions(jams PRIVATE HAS_CUDA=1)
target_include_directories(jams PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)
target_include_directories(jams PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

if (OpenMP_CXX_FOUND)
    # target_link_libraries(jams OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(jams ${OpenMP_CXX_LIB_NAMES})
    target_compile_definitions(jams PRIVATE HAS_OMP=1)
endif()

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET jams_object_lib
            PROPERTY
            BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
target_sources(jams_object_lib
        PRIVATE
        containers/bimap.h
        containers/cell.cc
        containers/cell.h
        containers/mat3.h
        containers/mat3.h
        containers/material.h
        containers/name_id_map.h
        containers/neartree.h
        containers/sparsematrix.cc
        containers/sparsematrix.h
        containers/sparsematrix4d.h
        containers/unitcell.cc
        containers/unitcell.h
        containers/vec3.h
        core/base.h
        core/globals.h
        core/hamiltonian.cc
        core/hamiltonian.h
        core/interactions.cc
        core/interactions.h
        core/jams++.cc
        core/jams++.h
        core/lattice.cc
        core/lattice.h
        core/monitor.cc
        core/monitor.h
        core/physics.cc
        core/physics.h
        core/random.cc
        core/solver.cc
        core/solver.h
        core/thermostat.cc
        core/thermostat.h
        core/types.h
        cuda/cuda-complex-operators.h
        cuda/cuda-sparse-helpers.h
        cuda/cuda_array_kernels.cu
        cuda/cuda_array_kernels.h
        cuda/cuda_defs.cc
        cuda/cuda_defs.h
        cuda/cuda_solver.cc
        cuda/cuda_solver.h
        cuda/cuda_sparsematrix.cu
        cuda/cuda_sparsematrix.h
        cuda/cuda_vector_ops.h
        cuda/wrappers/stream.h
        hamiltonian/anisotropy_uniaxial.cu
        hamiltonian/anisotropy_uniaxial.h
        hamiltonian/anisotropy_uniaxial_kernel.h
        hamiltonian/cuda_dipole_fft.cu
        hamiltonian/cuda_dipole_fft.h
        hamiltonian/dipole.cc
        hamiltonian/dipole.h
        hamiltonian/dipole_bruteforce.cu
        hamiltonian/dipole_bruteforce.h
        hamiltonian/dipole_bruteforce_kernel.h
        hamiltonian/dipole_cpu_bruteforce.cc
        hamiltonian/dipole_cpu_bruteforce.h
        hamiltonian/dipole_cuda_sparse_tensor.cu
        hamiltonian/dipole_cuda_sparse_tensor.h
        hamiltonian/dipole_ewald.cc
        hamiltonian/dipole_ewald.h
        hamiltonian/dipole_fft.cc
        hamiltonian/dipole_fft.h
        hamiltonian/dipole_tensor.cc
        hamiltonian/dipole_tensor.h
        hamiltonian/exchange.cc
        hamiltonian/exchange.h
        hamiltonian/exchange_neartree.cc
        hamiltonian/exchange_neartree.h
        hamiltonian/strategy.h
        hamiltonian/zeeman.cu
        hamiltonian/zeeman.h
        hamiltonian/zeeman_kernel.h
        helpers/consts.h
        helpers/defaults.h
        helpers/duration.h
        helpers/error.cc
        helpers/error.h
        helpers/exception.h
        helpers/fft.cc
        helpers/fft.h
        helpers/load.h
        helpers/maths.cc
        helpers/maths.h
        helpers/montecarlo.h
        helpers/permutations.h
        helpers/random.h
        helpers/runningstat.h
        helpers/slice.cc
        helpers/slice.h
        helpers/stats.cc
        helpers/stats.h
        helpers/utils.cc
        helpers/utils.h
        interface/blas.h
        interface/config.cc
        interface/config.h
        monitors/binary.cc
        monitors/binary.h
        monitors/boltzmann.cc
        monitors/boltzmann.h
        monitors/cuda-spin-current.cc
        monitors/cuda-spin-current.h
        monitors/cuda-spin-current-kernel.cu
        monitors/energy.cc
        monitors/energy.h
        monitors/hdf5.cc
        monitors/hdf5.h
        monitors/magnetisation.cc
        monitors/magnetisation.h
        monitors/magnetisation_rate.cc
        monitors/magnetisation_rate.h
        monitors/skyrmion.cc
        monitors/skyrmion.h
        monitors/smr.cc
        monitors/smr.h
        monitors/spin_pumping.cc
        monitors/spin_pumping.h
        monitors/spin_temperature.cc
        monitors/spin_temperature.h
        monitors/structurefactor.cc
        monitors/structurefactor.h
        monitors/torque.cc
        monitors/torque.h
        monitors/vtu.cc
        monitors/vtu.h
        monitors/xyz.cc
        monitors/xyz.h
        physics/empty.h
        physics/field_cool.cc
        physics/field_cool.h
        physics/flips.cc
        physics/flips.h
        physics/fmr.cc
        physics/fmr.h
        physics/mean_first_passage_time.cc
        physics/mean_first_passage_time.h
        physics/ping.cc
        physics/ping.h
        physics/square_field_pulse.cc
        physics/square_field_pulse.h
        physics/two_temperature_model.cc
        physics/two_temperature_model.h
        solvers/cpu_llg_heun.cc
        solvers/cpu_llg_heun.h
        solvers/cpu_monte_carlo_constrained.cc
        solvers/cpu_monte_carlo_constrained.h
        solvers/cpu_monte_carlo_metropolis.cc
        solvers/cpu_monte_carlo_metropolis.h
        solvers/cuda_llg_heun.cu
        solvers/cuda_llg_heun.h
        solvers/cuda_llg_heun_kernel.h
        thermostats/cuda_langevin_bose.cu
        thermostats/cuda_langevin_bose.h
        thermostats/cuda_langevin_bose_kernel.h
        thermostats/cuda_langevin_white.cc
        thermostats/cuda_langevin_white.h
        )