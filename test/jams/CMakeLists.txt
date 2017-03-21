enable_testing()
set(PROJECT_TEST_NAME ${CMAKE_PROJECT_NAME}_test)
include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES})
file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/jams/*.cc)
MESSAGE(STATUS "DEBUG:: ${gtest_BINARY_DIR}")

cuda_add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
add_dependencies(${PROJECT_TEST_NAME} gtest)

target_link_libraries(${PROJECT_TEST_NAME}
    ${gtest_BINARY_DIR}/libgtest.a
    ${gtest_BINARY_DIR}/libgtest_main.a
    )

include_directories("${gtest_SOURCE_DIR}/include"
                    "${gmock_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})

add_test(test1 ${PROJECT_TEST_NAME})