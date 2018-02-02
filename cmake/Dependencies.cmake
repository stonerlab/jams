# -- Threads
find_package(Threads)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OMP)
endif()

# -- Libconfig++
find_package(CONFIG++ REQUIRED)
include_directories(SYSTEM ${CONFIG++_INCLUDE_DIR})
list(APPEND JAMS_LINKER_LIBS ${CONFIG++_LIBRARY})

# -- symspg
find_package(SYMSPG REQUIRED)
include_directories(SYSTEM ${SYMSPG_INCLUDE_DIR})
list(APPEND JAMS_LINKER_LIBS ${SYMSPG_LIBRARY})

# -- CUDA
find_package(CUDA QUIET REQUIRED)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND JAMS_LINKER_LIBS ${CUDA_LIBRARIES})
list(APPEND JAMS_LINKER_LIBS ${CUDA_CUFFT_LIBRARIES})
list(APPEND JAMS_LINKER_LIBS ${CUDA_CUBLAS_LIBRARIES})
list(APPEND JAMS_LINKER_LIBS ${CUDA_curand_LIBRARY})
list(APPEND JAMS_LINKER_LIBS ${CUDA_cusparse_LIBRARY})
add_definitions(-DCUDA)

find_package(MKL)
if(MKL_FOUND)
	include_directories(SYSTEM ${MKL_INCLUDE_DIR})
	include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})
	list(APPEND JAMS_LINKER_LIBS ${MKL_LIBRARIES})
	add_definitions(-DUSE_MKL)
endif(MKL_FOUND)

if(NOT MKL_FOUND)
	find_package(FFTW3 REQUIRED)
	include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})
	list(APPEND JAMS_LINKER_LIBS ${FFTW3_LIBRARY})
    find_package(BLAS REQUIRED)
    include_directories(SYSTEM ${BLAS_INCLUDE_DIR})
    list(APPEND JAMS_LINKER_LIBS ${BLAS_LIBRARIES})
endif(NOT MKL_FOUND)
