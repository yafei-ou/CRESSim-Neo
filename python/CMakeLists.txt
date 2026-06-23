include(GNUInstallDirs)

set(TARGET_NAME "cressim_neo_python")
set(CRESSIM_NEO_PYTHON_PACKAGE_NAME "cressim_neo")
set(CRESSIM_NEO_PYTHON_BUILD_ROOT "${CMAKE_BINARY_DIR}/bin")
set(CRESSIM_NEO_PYTHON_PACKAGE_DIR
    "${CRESSIM_NEO_PYTHON_BUILD_ROOT}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}")
set(CRESSIM_NEO_PYTHON_PACKAGE_INIT_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/package/__init__.py.in")
set(CRESSIM_NEO_PYTHON_PACKAGE_INIT
    "${CMAKE_CURRENT_BINARY_DIR}/package/__init__.py")

configure_file(
    "${CRESSIM_NEO_PYTHON_PACKAGE_INIT_TEMPLATE}"
    "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
    @ONLY
)

pybind11_add_module(${TARGET_NAME} MODULE
    bindings.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TARGET_NAME} PRIVATE
    cressim_neo_engine
)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "_cressim_neo"
    ARCHIVE_OUTPUT_DIRECTORY "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_path('platlib'))"
    OUTPUT_VARIABLE CRESSIM_NEO_PYTHON_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(CRESSIM_NEO_PYTHON_INSTALL_DIR STREQUAL "")
    set(CRESSIM_NEO_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

install(
    TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
    RUNTIME DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
)

install(
    FILES "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
    DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/assets/shaders"
    DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
)

if(CRESSIM_NEO_HAS_ULTRASOUND AND TARGET cru_interface)
    install(
        FILES "$<TARGET_FILE:cru_interface>"
        DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
    )
endif()

if(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH AND EXISTS "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    if(WIN32)
        install(
            FILES "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
            RENAME "spv_dxcompiler.dll"
        )
    elseif(UNIX)
        install(
            FILES "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
            RENAME "libdxcompiler.so"
        )
    endif()
endif()

set(CRESSIM_NEO_PYTHON_STUB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")
set(CRESSIM_NEO_PYTHON_STUB_FILE "${CRESSIM_NEO_PYTHON_STUB_DIR}/_cressim_neo.pyi")

find_program(CRESSIM_NEO_PYBIND11_STUBGEN_EXECUTABLE NAMES pybind11-stubgen)
set(CRESSIM_NEO_PYBIND11_STUBGEN_MODE "")

if(CRESSIM_NEO_PYBIND11_STUBGEN_EXECUTABLE)
    set(CRESSIM_NEO_PYBIND11_STUBGEN_MODE "executable")
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen --help
        RESULT_VARIABLE CRESSIM_NEO_PYBIND11_STUBGEN_MODULE_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(CRESSIM_NEO_PYBIND11_STUBGEN_MODULE_RESULT EQUAL 0)
        set(CRESSIM_NEO_PYBIND11_STUBGEN_MODE "module")
    endif()
endif()

if(CRESSIM_NEO_PYBIND11_STUBGEN_MODE STREQUAL "executable")
    add_custom_command(
        OUTPUT "${CRESSIM_NEO_PYTHON_STUB_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CRESSIM_NEO_PYTHON_STUB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CRESSIM_NEO_PYTHON_PACKAGE_DIR}
            ${CRESSIM_NEO_PYBIND11_STUBGEN_EXECUTABLE}
            _cressim_neo
            -o "${CRESSIM_NEO_PYTHON_STUB_DIR}"
        DEPENDS ${TARGET_NAME}
        VERBATIM
    )
elseif(CRESSIM_NEO_PYBIND11_STUBGEN_MODE STREQUAL "module")
    add_custom_command(
        OUTPUT "${CRESSIM_NEO_PYTHON_STUB_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CRESSIM_NEO_PYTHON_STUB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CRESSIM_NEO_PYTHON_PACKAGE_DIR}
            ${Python3_EXECUTABLE}
            -m pybind11_stubgen
            _cressim_neo
            -o "${CRESSIM_NEO_PYTHON_STUB_DIR}"
        DEPENDS ${TARGET_NAME}
        VERBATIM
    )
else()
    message(WARNING
        "pybind11-stubgen was not found. Python bindings will build without generated .pyi stubs.")
endif()

if(NOT CRESSIM_NEO_PYBIND11_STUBGEN_MODE STREQUAL "")
    add_custom_target(${TARGET_NAME}_stubs
        DEPENDS "${CRESSIM_NEO_PYTHON_STUB_FILE}"
    )
    install(
        FILES "${CRESSIM_NEO_PYTHON_STUB_FILE}"
        DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
        RENAME "__init__.pyi"
        OPTIONAL
    )
endif()

add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
        "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/assets/shaders"
        "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/shaders"
)

if(CRESSIM_NEO_HAS_ULTRASOUND AND TARGET cru_interface)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:cru_interface>"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/libcru_interface.so"
    )
endif()

if(WIN32 AND CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH AND EXISTS "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/spv_dxcompiler.dll"
    )
elseif(UNIX AND CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH AND EXISTS "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/libdxcompiler.so"
    )
endif()

if(BUILD_TESTING)
    add_test(
        NAME test_python_runtime_frame_readback
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CRESSIM_NEO_PYTHON_BUILD_ROOT}
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/python/runtime_frame_readback.py
    )
    set_tests_properties(test_python_runtime_frame_readback PROPERTIES
        SKIP_RETURN_CODE 77
    )
endif()
