add_library(cuda_external INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuda_external INTERFACE CUDA::cudart)
set(JAMS_CUDA_VERSION ${CUDAToolkit_VERSION})

foreach(LIB cusparse curand cublas cufft)
    add_library(${LIB}_external INTERFACE)
    target_link_libraries(${LIB}_external INTERFACE CUDA::${LIB})
    set(JAMS_CUDA_${LIB}_LIBRARIES CUDA::${LIB})
endforeach()