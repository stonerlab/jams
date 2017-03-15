# -- Google-test
include("cmake/External/gtest.cmake")

# -- Libconfig++
find_package(CONFIG++ REQUIRED)
include_directories(SYSTEM ${CONFIG++_INCLUDE_DIR})
list(APPEND JAMS_LINKER_LIBS ${CONFIG++_LIBRARY})

# -- symspg
find_package(SYMSPG REQUIRED)
include_directories(SYSTEM ${SYMSPG_INCLUDE_DIR})
list(APPEND JAMS_LINKER_LIBS ${SYMSPG_LIBRARY})


# -- HDF5
find_package(HDF5 COMPONENTS CXX REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIR})
list(APPEND JAMS_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

# -- CUDA
find_package(CUDA QUIET REQUIRED)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND JAMS_LINKER_LIBS ${CUDA_LIBRARIES})
list(APPEND JAMS_LINKER_LIBS ${CUDA_CUFFT_LIBRARIES})
list(APPEND JAMS_LINKER_LIBS ${CUDA_CUBLAS_LIBRARIES})
list(APPEND JAMS_LINKER_LIBS ${CUDA_curand_LIBRARY})
list(APPEND JAMS_LINKER_LIBS ${CUDA_cusparse_LIBRARY})

# -- BLAS
#if(NOT APPLE)
#  set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
#  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")
#
#  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
#    find_package(Atlas REQUIRED)
#    include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
#    list(APPEND JAMS_LINKER_LIBS ${Atlas_LIBRARIES})
#  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
#    find_package(OpenBLAS REQUIRED)
#    include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
#    list(APPEND JAMS_LINKER_LIBS ${OpenBLAS_LIB})
#  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND JAMS_LINKER_LIBS ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
 # endif()
#elseif(APPLE)
#  find_package(vecLib REQUIRED)
#  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
#  list(APPEND JAMS_LINKER_LIBS ${vecLib_LINKER_LIBS})
#
#  if(VECLIB_FOUND)
#    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
#      add_definitions(-DUSE_ACCELERATE)
#    endif()
#  endif()
#endif()