enable_testing()

# -- Google-test
include("${PROJECT_SOURCE_DIR}/cmake/External/gtest.cmake")

set(PROJECT_TEST_NAME ${CMAKE_PROJECT_NAME}_test)

# put source files into 'srcs' and 'cuda'
jams_pickup_jams_sources(${PROJECT_SOURCE_DIR})
list(REMOVE_ITEM srcs ${PROJECT_SOURCE_DIR}/src/jams/main.cc)


add_definitions(-DCUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -g -std=c++11)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30") # Kepler K20/K40/K80
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60") # Pascal (P100)
list(APPEND srcs ${cuda})


include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/test)

file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/jams/*.cc)

cuda_add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${srcs})

add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME} gtest)

target_link_libraries(${PROJECT_TEST_NAME}
    ${gtest_BINARY_DIR}/libgtest.a
    ${gtest_BINARY_DIR}/libgtest_main.a
    )

include_directories("${gtest_SOURCE_DIR}/include"
                    "${gmock_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_TEST_NAME} ${JAMS_LINKER_LIBS} ${CMAKE_THREAD_LIBS_INIT})

jams_use_cxx11(jams_test)

add_test(test1 ${PROJECT_TEST_NAME})