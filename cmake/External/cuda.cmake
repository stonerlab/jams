add_library(cuda INTERFACE)

find_package(CUDA REQUIRED)

target_include_directories(cuda INTERFACE ${CUDA_INCLUDE_DIRS})
foreach(LIB cusparse curand cublas cufft)
    add_library(${LIB} INTERFACE)
    target_include_directories(${LIB} INTERFACE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${LIB} INTERFACE ${CUDA_${LIB}_LIBRARY})
    set(JAMS_CUDA_${LIB}_LIBRARIES ${CUDA_${LIB}_LIBRARY})
endforeach()