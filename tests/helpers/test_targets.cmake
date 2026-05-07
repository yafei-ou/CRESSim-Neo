function(add_cressim_test)
    set(options INCLUDE_SRC)
    set(one_value_args SUITE NAME SOURCE)
    set(multi_value_args INCLUDE_DIRS LIBS ARGS)
    cmake_parse_arguments(TEST "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT TEST_SUITE OR NOT TEST_NAME OR NOT TEST_SOURCE)
        message(FATAL_ERROR "add_cressim_test requires SUITE, NAME, and SOURCE.")
    endif()

    set(target_name "test_${TEST_SUITE}_${TEST_NAME}")
    set(source_path "${PROJECT_SOURCE_DIR}/tests/${TEST_SUITE}/${TEST_SOURCE}")
    set(include_dirs
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/tests"
    )

    if(TEST_INCLUDE_SRC)
        list(APPEND include_dirs "${PROJECT_SOURCE_DIR}/src")
    endif()

    if(TEST_INCLUDE_DIRS)
        list(APPEND include_dirs ${TEST_INCLUDE_DIRS})
    endif()

    add_executable(${target_name}
        ${source_path}
    )

    target_include_directories(${target_name} PRIVATE
        ${include_dirs}
    )

    if(TEST_LIBS)
        target_link_libraries(${target_name} PRIVATE ${TEST_LIBS})
    endif()

    set_target_properties(${target_name}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_test(
        NAME "${target_name}"
        COMMAND ${target_name} ${TEST_ARGS}
    )
endfunction()
