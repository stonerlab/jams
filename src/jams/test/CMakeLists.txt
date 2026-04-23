include(GoogleTest)

if (JAMS_BUILD_CUDA)
    # current versions of cmake mistakenly pass -pthread to nvcc at link time
    # but nvcc does not accept -pthread
    set(gtest_disable_pthreads ON)
endif()

add_executable(jams_tests "jams_test.cc")

target_link_libraries(jams_tests
        jams_core
        gtest
        gmock)

target_include_directories(jams_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)

if(JAMS_BUILD_CUDA AND APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_target_properties(jams_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

set(jams_discovered_tests)
gtest_discover_tests(
        jams_tests
        TEST_LIST jams_discovered_tests)

set(jams_test_labels_script "${CMAKE_CURRENT_BINARY_DIR}/jams_test_labels.cmake")
file(WRITE "${jams_test_labels_script}" [=[
foreach(test_name IN LISTS jams_discovered_tests)
    set(test_labels unit)
    set(run_serial FALSE)

    if (test_name MATCHES "GPUTests"
            OR test_name MATCHES "(^|[./])Gpu"
            OR test_name MATCHES "(^|[./])Cuda"
            OR test_name MATCHES "_gpu")
        list(APPEND test_labels gpu)
        set(run_serial TRUE)
    else()
        list(APPEND test_labels cpu)
    endif()

    if (test_name MATCHES "SLOW")
        list(APPEND test_labels slow)
    endif()

    set_tests_properties("${test_name}" PROPERTIES LABELS "${test_labels}")

    if (run_serial)
        set_tests_properties("${test_name}" PROPERTIES RUN_SERIAL TRUE)
    endif()
endforeach()
]=])
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${jams_test_labels_script}")
