add_library(cuda_external INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuda_external INTERFACE CUDA::cudart)

set(JAMS_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
if (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    list(APPEND JAMS_CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
list(REMOVE_DUPLICATES JAMS_CUDA_INCLUDE_DIRS)

target_include_directories(cuda_external INTERFACE ${JAMS_CUDA_INCLUDE_DIRS})

# CUDA 13 moves the bundled CCCL headers (including Thrust) under include/cccl.
# Add that directory for host-compiled translation units while remaining a no-op
# on older toolkits where the legacy include layout is still present.
foreach(CUDA_INCLUDE_DIR IN LISTS JAMS_CUDA_INCLUDE_DIRS)
    if (EXISTS "${CUDA_INCLUDE_DIR}/cccl/thrust/device_vector.h")
        target_include_directories(cuda_external INTERFACE "${CUDA_INCLUDE_DIR}/cccl")
        break()
    endif()
endforeach()

set(JAMS_CUDA_VERSION ${CUDAToolkit_VERSION})

foreach(LIB cusparse curand cublas cufft)
    add_library(${LIB}_external INTERFACE)
    target_link_libraries(${LIB}_external INTERFACE CUDA::${LIB})
    set(JAMS_CUDA_${LIB}_LIBRARIES CUDA::${LIB})
endforeach()
