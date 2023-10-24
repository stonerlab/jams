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

function(external_dependency_version_error PACKAGE_NAME PACKAGE_VERSION MIN_VERSION)
    # remove any "v" prefixes from the version string
    string(REGEX REPLACE "^v" "" CLEAN_PACKAGE_VERSION ${PACKAGE_VERSION})

    if(${CLEAN_PACKAGE_VERSION} VERSION_LESS "${MIN_VERSION}")
        unset(MESSAGE_QUIET)

        message(FATAL_ERROR
                " \n"
                " -----------------------------------------------------------\n"
                " CMAKE CONFIGURE ERROR\n"
                " Unsupported ${PACKAGE_NAME} version.\n"
                " \n"
                " JAMS requires ${PACKAGE_NAME} version >= ${MIN_VERSION}\n"
                " ${PACKAGE_NAME} version found: ${CLEAN_PACKAGE_VERSION}\n"
                " -----------------------------------------------------------\n")
    endif()
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

    external_dependency_version_error("cuda" ${JAMS_CUDA_VERSION} ${JAMS_CUDA_VERSION_MIN})
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

    message(STATUS "  include directories: ")
    get_target_property(JAMS_HDF5_INCLUDE_DIRECTORIES hdf5_external INTERFACE_INCLUDE_DIRECTORIES)
    foreach(INC ${JAMS_HDF5_INCLUDE_DIRECTORIES})
        message(STATUS "    ${INC}")
    endforeach()

    message(STATUS "  link libraries:")
    get_target_property(JAMS_HDF5_LINK_LIBRARIES hdf5_external INTERFACE_LINK_LIBRARIES)
    foreach(LIB ${JAMS_HDF5_LINK_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
    external_dependency_version_error("HDF5" ${JAMS_HDF5_VERSION} ${JAMS_HDF5_VERSION_MIN})
else()
    missing_external_dependency_error("hdf5")
endif()

message(STATUS "fftw")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/fftw.cmake")
unset(MESSAGE_QUIET)

if (JAMS_FFTW3_FOUND)
    message(STATUS "  vendor: " ${JAMS_FFTW3_VENDOR})

    message(STATUS "  include directories: ")
    get_target_property(JAMS_FFTW3_INCLUDE_DIRECTORIES fftw_external INTERFACE_INCLUDE_DIRECTORIES)
    foreach(INC ${JAMS_FFTW3_INCLUDE_DIRECTORIES})
        message(STATUS "    ${INC}")
    endforeach()

    message(STATUS "  link libraries:")
    get_target_property(JAMS_FFTW3_LINK_LIBRARIES fftw_external INTERFACE_LINK_LIBRARIES)
    foreach(LIB ${JAMS_FFTW3_LINK_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
else()
    missing_external_dependency_error("fftw3")
endif()

message(STATUS "cblas")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/cblas.cmake")
unset(MESSAGE_QUIET)

if (JAMS_CBLAS_FOUND)
    message(STATUS "  vendor: " ${JAMS_CBLAS_VENDOR})
    get_target_property(JAMS_CBLAS_INCLUDE_DIRECTORIES cblas_external INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "  include directories: ")
    message(STATUS "    ${JAMS_CBLAS_INCLUDE_DIRECTORIES}")
    get_target_property(JAMS_CBLAS_LINK_LIBRARIES cblas_external INTERFACE_LINK_LIBRARIES)
    message(STATUS "  link libraries:")
    foreach(LIB ${JAMS_CBLAS_LINK_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
else()
    missing_external_dependency_error("cblas")
endif()

message(STATUS "libconfig")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/libconfig.cmake")
unset(MESSAGE_QUIET)
if (JAMS_LIBCONFIG_VERSION)
    message(STATUS "  url: " ${JAMS_LIBCONFIG_URL})
    message(STATUS "  version: " ${JAMS_LIBCONFIG_VERSION})
    external_dependency_version_error("libconfig" ${JAMS_LIBCONFIG_VERSION} ${JAMS_LIBCONFIG_VERSION_MIN})
else()
    if (NOT JAMS_LIBCONFIG_LIBRARIES)
        missing_external_dependency_error("libconfig")
    endif()
    message(STATUS "  link libraries:")
    foreach(LIB ${JAMS_LIBCONFIG_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
endif()

message(STATUS "spglib")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/spglib.cmake")
unset(MESSAGE_QUIET)
if (JAMS_SPGLIB_VERSION)
    message(STATUS "  url: " ${JAMS_SPGLIB_URL})
    message(STATUS "  version: " ${JAMS_SPGLIB_VERSION})
else()
    if (NOT JAMS_SPGLIB_LIBRARIES)
        missing_external_dependency_error("spglib")
    endif()
    message(STATUS "  link libraries:")
    foreach(LIB ${JAMS_SPGLIB_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
endif()

message(STATUS "pcg")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/pcg.cmake")
unset(MESSAGE_QUIET)
if (JAMS_PCG_VERSION)
    message(STATUS "  url: " ${JAMS_PCG_URL})
    message(STATUS "  version: " ${JAMS_PCG_VERSION})
else()
    if (NOT JAMS_PCG_LIBRARIES)
        missing_external_dependency_error("pcg")
    endif()
    message(STATUS "  link libraries:")
    foreach(LIB ${JAMS_PCG_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
endif()

message(STATUS "highfive")
set(MESSAGE_QUIET ON)
include("${PROJECT_SOURCE_DIR}/cmake/External/highfive.cmake")
unset(MESSAGE_QUIET)
if (JAMS_HIGHFIVE_VERSION)
    message(STATUS "  url: " ${JAMS_HIGHFIVE_URL})
    message(STATUS "  version: " ${JAMS_HIGHFIVE_VERSION})
else()
    if (NOT JAMS_HIGHFIVE_LIBRARIES)
        missing_external_dependency_error("highfive")
    endif()
    message(STATUS "  link libraries:")
    foreach(LIB ${JAMS_HIGHFIVE_LIBRARIES})
        message(STATUS "    ${LIB}")
    endforeach()
endif()


if(JAMS_BUILD_TESTS)
    message(STATUS "gtest")
    set(MESSAGE_QUIET ON)
    include("${PROJECT_SOURCE_DIR}/cmake/External/gtest.cmake")
    unset(MESSAGE_QUIET)
    message(STATUS "  url: " ${JAMS_GTEST_URL})
    message(STATUS "  version: " ${JAMS_GTEST_VERSION})
endif()
