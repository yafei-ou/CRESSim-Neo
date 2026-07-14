if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(CRESSim-Neo-Docs LANGUAGES NONE)
endif()

find_package(Doxygen REQUIRED)
find_program(SPHINX_BUILD_EXECUTABLE NAMES sphinx-build REQUIRED)

get_filename_component(CRESSIM_NEO_REPOSITORY_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(CRESSIM_NEO_DOCS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CRESSIM_NEO_DOCS_DOXYGEN_OUTPUT_DIR
    "${CRESSIM_NEO_DOCS_OUTPUT_DIR}/doxygen")
set(CRESSIM_NEO_DOCS_DOXYGEN_XML_DIR
    "${CRESSIM_NEO_DOCS_DOXYGEN_OUTPUT_DIR}/xml")
set(CRESSIM_NEO_DOCS_HTML_DIR "${CRESSIM_NEO_DOCS_OUTPUT_DIR}/html")
set(CRESSIM_NEO_DOCS_DOCTREE_DIR "${CRESSIM_NEO_DOCS_OUTPUT_DIR}/doctrees")
set(CRESSIM_NEO_DOCS_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file(Doxyfile.in "${CRESSIM_NEO_DOCS_DOXYFILE}" @ONLY)

file(GLOB_RECURSE CRESSIM_NEO_DOCS_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${CRESSIM_NEO_REPOSITORY_DIR}/include/*.h"
    "${CRESSIM_NEO_REPOSITORY_DIR}/include/*.hpp"
)

add_custom_command(
    OUTPUT "${CRESSIM_NEO_DOCS_DOXYGEN_XML_DIR}/index.xml"
    COMMAND Doxygen::doxygen "${CRESSIM_NEO_DOCS_DOXYFILE}"
    DEPENDS
        "${CRESSIM_NEO_DOCS_DOXYFILE}"
        ${CRESSIM_NEO_DOCS_PUBLIC_HEADERS}
    COMMENT "Generating C++ API XML with Doxygen"
    VERBATIM
)

add_custom_target(docs-doxygen
    DEPENDS "${CRESSIM_NEO_DOCS_DOXYGEN_XML_DIR}/index.xml"
)

add_custom_target(docs-html
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "CRESSIM_NEO_DOXYGEN_XML=${CRESSIM_NEO_DOCS_DOXYGEN_XML_DIR}"
        "${SPHINX_BUILD_EXECUTABLE}"
        --fail-on-warning
        --keep-going
        --fresh-env
        --builder html
        --doctree-dir "${CRESSIM_NEO_DOCS_DOCTREE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/source"
        "${CRESSIM_NEO_DOCS_HTML_DIR}"
    DEPENDS docs-doxygen
    COMMENT "Building Sphinx HTML documentation"
    VERBATIM
)
