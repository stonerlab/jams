include_directories(${CMAKE_BINARY_DIR}/generated)
jams_pickup_jams_sources(${PROJECT_SOURCE_DIR})

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-missing-braces -Wmissing-field-initializers")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -g -std=c++11)

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30") # Kepler K20/K40/K80
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60") # Pascal (P100)
list(APPEND srcs ${cuda})

cuda_add_executable(jams ${srcs})
target_link_libraries(jams ${JAMS_LINKER_LIBS})
jams_use_cxx11(jams)