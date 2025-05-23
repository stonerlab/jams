include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Sources.cmake)

add_executable(jams ""
        hamiltonian/cuda_crystal_field.h)
target_sources(jams PRIVATE main.cc ${JAMS_SOURCES_CXX})

target_include_directories(jams
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

target_link_libraries(jams
        fftw_external
        cblas_external
        hdf5_external
        pcg
        spglib
        libconfig
        highfive)

if(JAMS_BUILD_CUDA)
    target_sources(jams PRIVATE ${JAMS_SOURCES_CUDA})
    target_compile_definitions(jams PRIVATE HAS_CUDA)
    target_link_libraries(jams
            cuda_external
            cusparse_external
            curand_external
            cublas_external
            cufft_external)

    if (JAMS_BUILD_MIXED_PREC)
        target_compile_definitions(jams PRIVATE HAS_CUSPARSE_MIXED_PREC)
    endif()

    if(APPLE)
        # We need to add the path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_target_properties(jams PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endif()

if (JAMS_BUILD_OMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(jams PRIVATE HAS_OMP)
    target_compile_options(jams PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(jams OpenMP::OpenMP_CXX)
endif()

if (MKL_FOUND)
    target_compile_definitions(jams PRIVATE HAS_MKL)
endif()

if (JAMS_BUILD_FASTMATH)
    jams_set_fast_math(jams)
endif()