include_directories(${CMAKE_BINARY_DIR}/generated)
jams_pickup_jams_sources(${PROJECT_SOURCE_DIR})

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-missing-braces -Wmissing-field-initializers")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -std=c++11)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -g -std=c++11)
endif()
add_definitions(-DCUDA)

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30") # Kepler K20/K40/K80
list(APPEND srcs ${cuda})

cuda_add_executable(jams ${srcs})
target_link_libraries(jams ${JAMS_LINKER_LIBS})
target_link_libraries(jams pcg symspg config++ hdf5 cuda)
jams_use_cxx11(jams)