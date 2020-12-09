function(missing_external_dependency_error name)
    unset(MESSAGE_QUIET)
    message(FATAL_ERROR
            " \n"
            " -----------------------------------------------------------\n"
            " CMAKE CONFIGURE ERROR\n"
            " Could not find the dependency '${name}'.\n"
            " \n"
            " Check that it is installed and is visible to CMake. If \n"
            " you're on a HPC cluster make sure you have loaded the \n"
            " relavent module, for example try:\n"
            " \n"
            " module load ${name}\n"
            " -----------------------------------------------------------\n")
endfunction()

message("-----------------------------------------------------------")
message("CMAKE FINDING JAMS EXTERNAL DEPENDENCIES")
message("-----------------------------------------------------------")


message(STATUS "posix")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/posix.cmake")
unset(MESSAGE_QUIET)
message(STATUS "  headers found")

if(JAMS_BUILD_OMP)
    message(STATUS "openmp")
    set(MESSAGE_QUIET ON)
    include("${PROJECT_SOURCE_DIR}/cmake/External/omp.cmake")
    unset(MESSAGE_QUIET)
    if (OpenMP_FOUND)
        message(STATUS "  flags: " ${OpenMP_CXX_FLAGS})
    else()
        missing_external_dependency_error("openmp")
    endif()
endif()

if (JAMS_BUILD_CUDA)
    message(STATUS "cuda")
    set(MESSAGE_QUIET ON)
    include("${PROJECT_SOURCE_DIR}/cmake/External/cuda.cmake")
    unset(MESSAGE_QUIET)
    if (CUDA_FOUND)
        message(STATUS "  cusparse: " ${JAMS_CUDA_cusparse_LIBRARIES})
        message(STATUS "  curand:   " ${JAMS_CUDA_curand_LIBRARIES})
        message(STATUS "  cublas:   " ${JAMS_CUDA_cublas_LIBRARIES})
        message(STATUS "  cufft:    " ${JAMS_CUDA_cufft_LIBRARIES})
    else()
        missing_external_dependency_error("cuda")
    endif()
endif()

message(STATUS "hdf5")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/hdf5.cmake")
unset(MESSAGE_QUIET)

if (HDF5_FOUND)
    message(STATUS "  version: " ${JAMS_HDF5_VERSION})
    message(STATUS "  libs: " ${JAMS_HDF5_LIBRARIES})
else()
    missing_external_dependency_error("hdf5")
endif()

message(STATUS "fftw")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/fftw.cmake")
unset(MESSAGE_QUIET)

if (JAMS_FFTW3_FOUND)
    message(STATUS "  vendor: " ${JAMS_FFTW3_VENDOR})
    message(STATUS "  libs: " ${JAMS_FFTW3_LIBRARIES})
else()
    missing_external_dependency_error("fftw3")
endif()

message(STATUS "cblas")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/cblas.cmake")
unset(MESSAGE_QUIET)

if (JAMS_CBLAS_FOUND)
    message(STATUS "  vendor: " ${JAMS_CBLAS_VENDOR})
    message(STATUS "  libs: " ${JAMS_CBLAS_LIBRARIES})
else()
    missing_external_dependency_error("cblas")
endif()

message("-----------------------------------------------------------")
message("CMAKE DOWNLOAD AND CONFIGURE JAMS BUILT-IN DEPENDENCIES")
message("-----------------------------------------------------------")

message(STATUS "libconfig")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/libconfig.cmake")
unset(MESSAGE_QUIET)
message(STATUS "  url: " ${JAMS_LIBCONFIG_URL})
message(STATUS "  version: " ${JAMS_LIBCONFIG_VERSION})

message(STATUS "highfive")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/highfive.cmake")
unset(MESSAGE_QUIET)
message(STATUS "  url: " ${JAMS_HIGHFIVE_URL})
message(STATUS "  version: " ${JAMS_HIGHFIVE_VERSION})

message(STATUS "spglib")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/spglib.cmake")
unset(MESSAGE_QUIET)
message(STATUS "  url: " ${JAMS_SPGLIB_URL})
message(STATUS "  version: " ${JAMS_SPGLIB_VERSION})

message(STATUS "pcg")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/pcg.cmake")
unset(MESSAGE_QUIET)
message(STATUS "  url: " ${JAMS_PCG_URL})
message(STATUS "  version: " ${JAMS_PCG_VERSION})

if(JAMS_BUILD_TESTS)
    message(STATUS "gtest")
    set(MESSAGE_QUIET ON)
    include("${PROJECT_SOURCE_DIR}/cmake/External/gtest.cmake")
    unset(MESSAGE_QUIET)
    message(STATUS "  url: " ${JAMS_GTEST_URL})
    message(STATUS "  version: " ${JAMS_GTEST_VERSION})
endif()