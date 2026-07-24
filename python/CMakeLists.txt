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
set(CRESSIM_NEO_PYTHON_PACKAGE_STUB
    "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/__init__.pyi")
set(CRESSIM_NEO_PYTHON_PACKAGE_EXTRA_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/package/cartpole.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/torch_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/soft_body_pusher_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/fluid_pouring_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/camera_centering_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/ultrasound_centering_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/soft_body_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/fluid_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/psm_builder.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/psm_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/psm_soft_grasp_env.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package/psm_blood_suction_env.py")

configure_file(
    "${CRESSIM_NEO_PYTHON_PACKAGE_INIT_TEMPLATE}"
    "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
    @ONLY
)

pybind11_add_module(${TARGET_NAME} MODULE
    bindings.cpp
)

set(CRESSIM_NEO_PYTHON_RUNTIME_TARGETS
    cressim_neo_engine
)
if(NOT ENGINE_STATIC)
    list(APPEND CRESSIM_NEO_PYTHON_RUNTIME_TARGETS
        cressim_neo_common
        cressim_neo_graphics
        cressim_neo_physics
        cressim_neo_gpu
    )
endif()
set(CRESSIM_NEO_PYTHON_OPTIONAL_RUNTIME_TARGETS "")

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${CRESSIM_NEO_PYTHON_RUNTIME_TARGETS}
)

# An Arch Linux GCC 16 upgrade may resolve -lstdc++ to libstdc++.a.
# Force the shared runtime so this Python extension remains compatible with
# native modules such as PyTorch.
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
        OUTPUT_VARIABLE CRESSIM_NEO_SHARED_LIBSTDCXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS "${CRESSIM_NEO_SHARED_LIBSTDCXX}")
        target_link_libraries(${TARGET_NAME} PRIVATE "${CRESSIM_NEO_SHARED_LIBSTDCXX}")
    endif()
endif()

if(TARGET cressim_neo_viewer)
    target_link_libraries(${TARGET_NAME} PRIVATE cressim_neo_viewer)
    target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_PYTHON_HAS_VIEWER=1)
    if(NOT ENGINE_STATIC)
        list(APPEND CRESSIM_NEO_PYTHON_OPTIONAL_RUNTIME_TARGETS cressim_neo_viewer)
    endif()
endif()

set_target_properties(${TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "_cressim_neo"
    ARCHIVE_OUTPUT_DIRECTORY "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
)

if(APPLE)
    set_target_properties(${TARGET_NAME}
        PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(${TARGET_NAME}
        PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_path('platlib', vars={'base': '', 'platbase': ''}).lstrip('/\\\\'))"
    OUTPUT_VARIABLE CRESSIM_NEO_PYTHON_DEFAULT_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(CRESSIM_NEO_PYTHON_DEFAULT_INSTALL_DIR STREQUAL "")
    set(CRESSIM_NEO_PYTHON_DEFAULT_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

set(CRESSIM_NEO_PYTHON_INSTALL_DIR
    "${CRESSIM_NEO_PYTHON_DEFAULT_INSTALL_DIR}"
    CACHE PATH "Python package install directory (relative to CMAKE_INSTALL_PREFIX or absolute)"
)

# scikit-build-core supplies a wheel staging prefix whose root is already the
# target interpreter's platlib directory.  Installing the normal CMake
# platlib-relative path inside it would nest lib/pythonX.Y/site-packages in the
# wheel and make the package unimportable after pip installation.
if(DEFINED SKBUILD)
    set(CRESSIM_NEO_PYTHON_INSTALL_DIR ".")
endif()

set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "")
set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH "")

if(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH)
    set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    if(WIN32)
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "spv_dxcompiler.dll")
    elseif(APPLE)
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "libdxcompiler.dylib")
    elseif(UNIX)
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "libdxcompiler.so")
    endif()
endif()

if(NOT CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH)
    if(WIN32)
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "spv_dxcompiler.dll")
    elseif(APPLE)
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "libdxcompiler.dylib")
    elseif(UNIX)
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME "libdxcompiler.so")
    endif()

    set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_BUILD_FALLBACK
        "${CMAKE_BINARY_DIR}/bin/${CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME}")
    if(EXISTS "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_BUILD_FALLBACK}")
        set(CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH
            "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_BUILD_FALLBACK}")
    endif()
endif()

install(
    TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}" COMPONENT Python
    RUNTIME DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}" COMPONENT Python
)

if(NOT ENGINE_STATIC)
    install(
        TARGETS ${CRESSIM_NEO_PYTHON_RUNTIME_TARGETS} ${CRESSIM_NEO_PYTHON_OPTIONAL_RUNTIME_TARGETS}
        LIBRARY DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}" COMPONENT Python
        RUNTIME DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}" COMPONENT Python
    )
endif()

install(
    FILES "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
    DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
    COMPONENT Python
)

install(
    FILES ${CRESSIM_NEO_PYTHON_PACKAGE_EXTRA_FILES}
    DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
    COMPONENT Python
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/assets/"
    DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}/assets"
    COMPONENT Python
)

if(CRESSIM_NEO_HAS_ULTRASOUND AND TARGET cru_interface)
    install(
        FILES "$<TARGET_FILE:cru_interface>"
        DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
        COMPONENT Python
    )
endif()

if(CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH AND EXISTS "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH}")
    install(
        FILES "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH}"
        DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
        RENAME "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME}"
        COMPONENT Python
    )
elseif(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH)
    message(WARNING
        "CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH does not exist for Python packaging: "
        "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
endif()

if(WIN32 AND CRESSIM_NEO_DXC_RUNTIME_PATH AND CRESSIM_NEO_DXIL_RUNTIME_PATH)
    if(EXISTS "${CRESSIM_NEO_DXC_RUNTIME_PATH}" AND EXISTS "${CRESSIM_NEO_DXIL_RUNTIME_PATH}")
        install(
            FILES "${CRESSIM_NEO_DXC_RUNTIME_PATH}" "${CRESSIM_NEO_DXIL_RUNTIME_PATH}"
            DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
            COMPONENT Python
        )
    endif()
endif()
install(
    FILES "${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
    COMPONENT Python
)

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
        OUTPUT "${CRESSIM_NEO_PYTHON_PACKAGE_STUB}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CRESSIM_NEO_PYTHON_PACKAGE_DIR}
            ${CRESSIM_NEO_PYBIND11_STUBGEN_EXECUTABLE}
            _cressim_neo
            -o "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/_cressim_neo.pyi"
            "${CRESSIM_NEO_PYTHON_PACKAGE_STUB}"
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/_cressim_neo.pyi"
        DEPENDS ${TARGET_NAME}
        VERBATIM
    )
elseif(CRESSIM_NEO_PYBIND11_STUBGEN_MODE STREQUAL "module")
    add_custom_command(
        OUTPUT "${CRESSIM_NEO_PYTHON_PACKAGE_STUB}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CRESSIM_NEO_PYTHON_PACKAGE_DIR}
            ${Python3_EXECUTABLE}
            -m pybind11_stubgen
            _cressim_neo
            -o "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/_cressim_neo.pyi"
            "${CRESSIM_NEO_PYTHON_PACKAGE_STUB}"
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/_cressim_neo.pyi"
        DEPENDS ${TARGET_NAME}
        VERBATIM
    )
else()
    message(WARNING
        "pybind11-stubgen was not found. Python bindings will build without generated .pyi stubs.")
endif()

if(NOT CRESSIM_NEO_PYBIND11_STUBGEN_MODE STREQUAL "")
    add_custom_target(${TARGET_NAME}_stubs ALL
        DEPENDS "${CRESSIM_NEO_PYTHON_PACKAGE_STUB}"
    )
    add_dependencies(${TARGET_NAME}_stubs ${TARGET_NAME})
    install(
        FILES "${CRESSIM_NEO_PYTHON_PACKAGE_STUB}"
        DESTINATION "${CRESSIM_NEO_PYTHON_INSTALL_DIR}/${CRESSIM_NEO_PYTHON_PACKAGE_NAME}"
        COMPONENT Python
        OPTIONAL
    )
endif()

add_custom_target(${TARGET_NAME}_package_files ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
        "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CRESSIM_NEO_PYTHON_PACKAGE_EXTRA_FILES}
        "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/assets"
        "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/assets"
    DEPENDS
        "${CRESSIM_NEO_PYTHON_PACKAGE_INIT}"
        ${CRESSIM_NEO_PYTHON_PACKAGE_EXTRA_FILES}
)

if(CRESSIM_NEO_HAS_ULTRASOUND AND TARGET cru_interface)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:cru_interface>"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/$<TARGET_FILE_NAME:cru_interface>"
    )
endif()

if(CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH AND EXISTS "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH}")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_PYTHON_DXC_RUNTIME_SOURCE_PATH}"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/${CRESSIM_NEO_PYTHON_DXC_RUNTIME_NAME}"
    )
endif()

if(WIN32 AND CRESSIM_NEO_DXC_RUNTIME_PATH AND CRESSIM_NEO_DXIL_RUNTIME_PATH AND
   EXISTS "${CRESSIM_NEO_DXC_RUNTIME_PATH}" AND EXISTS "${CRESSIM_NEO_DXIL_RUNTIME_PATH}")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_DXC_RUNTIME_PATH}"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/dxcompiler.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_DXIL_RUNTIME_PATH}"
            "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/dxil.dll"
    )
endif()

if(TARGET ${TARGET_NAME}_stubs)
    add_dependencies(${TARGET_NAME}_stubs ${TARGET_NAME}_package_files)
endif()

if(NOT ENGINE_STATIC)
    foreach(CRESSIM_NEO_PYTHON_RUNTIME_TARGET IN LISTS CRESSIM_NEO_PYTHON_RUNTIME_TARGETS)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${CRESSIM_NEO_PYTHON_RUNTIME_TARGET}>"
                "${CRESSIM_NEO_PYTHON_PACKAGE_DIR}/$<TARGET_FILE_NAME:${CRESSIM_NEO_PYTHON_RUNTIME_TARGET}>"
        )
    endforeach()
endif()
