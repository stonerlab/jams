# -- Threads
find_package(Threads QUIET)
find_package(OpenMP QUIET)
add_library(openmp INTERFACE IMPORTED)
if (OPENMP_FOUND)
    set_target_properties(openmp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
    target_compile_definitions(openmp -DUSE_OMP)
endif()

add_library(pcg INTERFACE)
target_include_directories(pcg INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

add_library(jblib INTERFACE)
target_include_directories(jblib INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

# -- Libconfig++
find_package(CONFIG++ QUIET REQUIRED)
add_library(config++ INTERFACE IMPORTED)
set_property(TARGET config++ PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONFIG++_INCLUDE_DIR})
set_property(TARGET config++ PROPERTY INTERFACE_LINK_LIBRARIES ${CONFIG++_LIBRARY})

# -- symspg
find_package(SYMSPG QUIET REQUIRED)
add_library(symspg INTERFACE IMPORTED)
set_property(TARGET symspg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SYMSPG_INCLUDE_DIR})
set_property(TARGET symspg PROPERTY INTERFACE_LINK_LIBRARIES ${SYMSPG_LIBRARY})

# -- hdf5
find_package(HDF5 COMPONENTS CXX QUIET REQUIRED)
add_library(hdf5 INTERFACE IMPORTED)
set_property(TARGET hdf5 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIR})
set_property(TARGET hdf5 PROPERTY INTERFACE_LINK_LIBRARIES ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

# -- CUDA
find_package(CUDA QUIET REQUIRED)
add_library(cuda INTERFACE IMPORTED)
set_property(TARGET cuda PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})
set_property(TARGET cuda PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})

find_package(MKL QUIET)
if(MKL_FOUND)
	include_directories(SYSTEM ${MKL_INCLUDE_DIR})
	include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})
	list(APPEND JAMS_LINKER_LIBS ${MKL_LIBRARIES})
	add_definitions(-DUSE_MKL)
endif(MKL_FOUND)

if(NOT MKL_FOUND)
	find_package(FFTW3 QUIET REQUIRED)
	include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})
	list(APPEND JAMS_LINKER_LIBS ${FFTW3_LIBRARY})
    find_package(BLAS QUIET REQUIRED)
    include_directories(SYSTEM ${BLAS_INCLUDE_DIR})
    list(APPEND JAMS_LINKER_LIBS ${BLAS_LIBRARIES})
endif(NOT MKL_FOUND)
