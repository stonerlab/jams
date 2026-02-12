macro(ensure_out_of_source_build)
     string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${CMAKE_BINARY_DIR}" insource)
     get_filename_component(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
     string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${PARENTDIR}" insourcesubdir)
    if(insource OR insourcesubdir)
        message(FATAL_ERROR
                " \n"
                " -----------------------------------------------------------\n"
                " CMAKE CONFIGURE ERROR\n"
                " Building inside the project source folder is not allowed.\n"
                " You want to do something like:\n"
                " \n"
                " mkdir ${CMAKE_SOURCE_DIR}/build\n"
                " cd ${CMAKE_SOURCE_DIR}/build\n"
                " cmake -DCMAKE_BUILD_TYPE=Release ..\n"
                " -----------------------------------------------------------\n")

endif(insource OR insourcesubdir)
endmacro()

function(jams_extract_git_info)
    set(JAMS_GIT_BRANCH "unknown")
    set(JAMS_GIT_COMMIT_HASH "unknown")
    find_package(Git)

    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE JAMS_GIT_BRANCH
                RESULT_VARIABLE __git_result)

        if(NOT ${__git_result} EQUAL 0)
            set(JAMS_GIT_BRANCH "unknown")
        endif()

        # Get the latest abbreviated commit hash of the working branch
        execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE JAMS_GIT_COMMIT_HASH
                RESULT_VARIABLE __git_result)

        if(NOT ${__git_result} EQUAL 0)
            set(JAMS_GIT_COMMIT_HASH "unknown")
        endif()

        execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --tags --long --dirty
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE JAMS_GIT_DESCRIPTION
                RESULT_VARIABLE __git_result)

        if(NOT ${__git_result} EQUAL 0)
            set(JAMS_GIT_DESCRIPTION "unknown")
        endif()
    endif()

    set(JAMS_GIT_BRANCH ${JAMS_GIT_BRANCH} PARENT_SCOPE)
    set(JAMS_GIT_COMMIT_HASH ${JAMS_GIT_COMMIT_HASH} PARENT_SCOPE)
    set(JAMS_GIT_DESCRIPTION ${JAMS_GIT_DESCRIPTION} PARENT_SCOPE)
endfunction()

function(jams_set_fast_math target)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
        set(JAMS_FAST_MATH_OPT -O3 -ffast-math CACHE STRING "enabled compiler fast math options")
        target_compile_options(${target} PRIVATE $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${JAMS_FAST_MATH_OPT}>)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(JAMS_FAST_MATH_OPT -fno-math-errno -fno-rounding-math -fno-signaling-nans -fno-signed-zeros -fcx-limited-range CACHE STRING "enabled compiler fast math options")
        target_compile_options(${target} PRIVATE $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${JAMS_FAST_MATH_OPT}>)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(JAMS_FAST_MATH_OPT -ftz -msse3 -no-prec-div -fast -fp-model fast=2 CACHE STRING "enabled compiler fast math options")
        target_compile_options(${target} PRIVATE $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${JAMS_FAST_MATH_OPT}>)
    endif()
endfunction()

function(prepend variable prefix)
    set(__list "")
    foreach(__s ${${variable}})
        list(APPEND __list "${prefix}/${__s}")
    endforeach()
    set(${variable} "${__list}" PARENT_SCOPE)
endfunction()

# frmo caffe
function(jams_convert_absolute_paths variable)
    set(__dlist "")
    foreach(__s ${${variable}})
        get_filename_component(__abspath ${__s} ABSOLUTE)
        list(APPEND __list ${__abspath})
    endforeach()
    set(${variable} ${__list} PARENT_SCOPE)
endfunction()