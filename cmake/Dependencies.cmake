# -- Threads
find_package(Threads QUIET)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_compile_definitions(OpenMP::OpenMP_CXX PUBLIC HAS_OMP=1)
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

# -- FFTW
add_library(fftw3 INTERFACE)
if(MKL_FOUND)
    set_property(TARGET fftw3 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR})
    set_property(TARGET fftw3 PROPERTY INTERFACE_LINK_LIBRARIES ${MKL_LIBRARIES})
    target_compile_definitions(fftw3 INTERFACE HAS_MKL=1)
else()
    find_package(FFTW3 QUIET REQUIRED)
    set_property(TARGET fftw3 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR})
    set_property(TARGET fftw3 PROPERTY INTERFACE_LINK_LIBRARIES ${FFTW3_LIBRARY})
endif(MKL_FOUND)

# -- BLAS
add_library(cblas INTERFACE)
if(MKL_FOUND)
    set_property(TARGET cblas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MKL_INCLUDE_DIR})
    set_property(TARGET cblas PROPERTY INTERFACE_LINK_LIBRARIES ${MKL_LIBRARIES})
    target_compile_definitions(cblas INTERFACE HAS_MKL=1)
else()
    find_package(CBLAS QUIET REQUIRED)
    set_property(TARGET cblas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BLAS_INCLUDE_DIR})
    set_property(TARGET cblas PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})
endif(MKL_FOUND)