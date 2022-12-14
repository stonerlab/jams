add_library(cuda_external INTERFACE)

find_package(CUDA QUIET)

target_include_directories(cuda_external INTERFACE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_external INTERFACE cuda)
set(JAMS_CUDA_VERSION ${CUDA_VERSION})

foreach(LIB cusparse curand cublas cufft)
    add_library(${LIB}_external INTERFACE)
    target_include_directories(${LIB}_external INTERFACE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${LIB}_external INTERFACE ${CUDA_${LIB}_LIBRARY})
    set(JAMS_CUDA_${LIB}_LIBRARIES ${CUDA_${LIB}_LIBRARY})
endforeach()