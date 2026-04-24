include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Sources.cmake)

add_library(jams_core STATIC "")
target_sources(jams_core PRIVATE ${JAMS_SOURCES_CXX})
target_compile_features(jams_core PUBLIC cxx_std_17)

target_include_directories(jams_core
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

target_link_libraries(jams_core
        PUBLIC
        fftw_external
        cblas_external
        hdf5_external
        pcg
        spglib
        libconfig
        highfive)

if(JAMS_BUILD_CUDA)
    target_sources(jams_core PRIVATE ${JAMS_SOURCES_CUDA})
    target_compile_definitions(jams_core PUBLIC HAS_CUDA=1)
    target_link_libraries(jams_core
            PUBLIC
            cuda_external
            cusparse_external
            curand_external
            cublas_external
            cufft_external)

    if (JAMS_BUILD_MIXED_PREC)
        target_compile_definitions(jams_core PUBLIC HAS_CUSPARSE_MIXED_PREC=1)
    endif()
endif()

if (JAMS_BUILD_MIXED_PREC)
    target_compile_definitions(jams_core PUBLIC DO_MIXED_PRECISION=1)
endif()

if (JAMS_BUILD_OMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(jams_core PUBLIC HAS_OMP=1)
    target_compile_options(jams_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(jams_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (MKL_FOUND)
    target_compile_definitions(jams_core PUBLIC HAS_MKL=1)
endif()

if (JAMS_BUILD_FASTMATH)
    jams_set_fast_math(jams_core)
endif()

add_executable(jams "")
target_sources(jams PRIVATE main.cc)
target_link_libraries(jams PRIVATE jams_core)

if(JAMS_BUILD_CUDA AND APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_target_properties(jams PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

include(GNUInstallDirs)
install(TARGETS jams
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
