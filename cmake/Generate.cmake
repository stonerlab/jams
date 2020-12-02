include("${PROJECT_SOURCE_DIR}/cmake/Utils.cmake")

jams_extract_git_info()

string(TIMESTAMP JAMS_BUILD_TIME "%Y-%m-%d %H:%M:%S")

configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/Templates/version.h.in
        ${CMAKE_BINARY_DIR}/generated/version.h)

