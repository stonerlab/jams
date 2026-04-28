enable_testing()
include(GoogleTest)

if (JAMS_BUILD_CUDA)
    # current versions of cmake mistakenly pass -pthread to nvcc at link time
    # but nvcc does not accept -pthread
    set(gtest_disable_pthreads ON)
endif()

add_executable(jams_low_level_tests "jams_low_level_test.cc")

target_link_libraries(jams_low_level_tests
        jams_low_level
        gtest
        gmock)

target_compile_features(jams_low_level_tests PRIVATE cxx_std_17)

target_include_directories(jams_low_level_tests PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

target_include_directories(jams_low_level_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)

if(JAMS_BUILD_CUDA AND APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_target_properties(jams_low_level_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

gtest_discover_tests(jams_low_level_tests
        PROPERTIES LABELS low_level)

add_executable(jams_tests "jams_test.cc")

target_link_libraries(jams_tests
        jams_core
        gtest
        gmock)

target_compile_features(jams_tests PRIVATE cxx_std_17)

target_include_directories(jams_tests PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)

if(JAMS_BUILD_CUDA AND APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_target_properties(jams_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

gtest_discover_tests(jams_tests)
