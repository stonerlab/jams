include("${PROJECT_SOURCE_DIR}/cmake/External/posix.cmake")

if(JAMS_BUILD_OMP)
    include("${PROJECT_SOURCE_DIR}/cmake/External/omp.cmake")
endif()

if (JAMS_BUILD_CUDA)
    include("${PROJECT_SOURCE_DIR}/cmake/External/cuda.cmake")
endif()

include("${PROJECT_SOURCE_DIR}/cmake/External/hdf5.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/External/libconfig.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/External/highfive.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/External/spglib.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/External/pcg.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/External/fftw.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/External/cblas.cmake")
