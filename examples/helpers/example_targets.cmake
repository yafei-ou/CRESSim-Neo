function(add_cressim_example)
    set(one_value_args NAME SOURCE)
    set(multi_value_args COMPILE_DEFINITIONS LIBS)
    cmake_parse_arguments(EXAMPLE "" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT EXAMPLE_NAME OR NOT EXAMPLE_SOURCE)
        message(FATAL_ERROR "add_cressim_example requires NAME and SOURCE.")
    endif()

    set(target_name "example_${EXAMPLE_NAME}")
    set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_SOURCE}")

    add_executable(${target_name}
        ${source_path}
    )

    target_include_directories(${target_name} PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/examples"
    )

    target_link_libraries(${target_name} PRIVATE
        cressim_neo_viewer
        ${EXAMPLE_LIBS}
    )

    if(EXAMPLE_COMPILE_DEFINITIONS)
        target_compile_definitions(${target_name} PRIVATE
            ${EXAMPLE_COMPILE_DEFINITIONS}
        )
    endif()

    set_target_properties(${target_name}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()
