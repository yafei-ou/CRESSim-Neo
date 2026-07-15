#!/usr/bin/env bash
# Configure the standard Linux build trees through a small interactive menu.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/build"

usage() {
    cat <<'EOF'
Usage: scripts/configure_builds.sh

Interactively configures any combination of:
  1. build/linux-release
  2. build/linux-debug
  3. build/docs-site

The documentation site imports the Python bindings. If documentation is
selected, the script asks which native build supplies that package and verifies
that it has already been configured and built before configuring build/docs-site.

Before configuring native builds, the script also asks which common CMake
options to enable (viewer, examples, Python bindings, tests, clang-tidy,
CUDA interop, and Ultrasound). Ultrasound automatically enables CUDA interop.
The displayed defaults match the project's CMake defaults.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

read -r -p $'Select builds to configure (comma-separated; default: 1,2,3):\n  1) Linux Release\n  2) Linux Debug\n  3) Documentation site\n> ' selections
selections="${selections:-1,2,3}"
selections="${selections//,/ }"

configure_release=false
configure_debug=false
configure_docs=false

for selection in ${selections}; do
    case "${selection}" in
        1|release) configure_release=true ;;
        2|debug) configure_debug=true ;;
        3|docs|documentation) configure_docs=true ;;
        *)
            echo "Unknown selection: ${selection}" >&2
            echo "Use 1, 2, and/or 3 (for example: 1,3)." >&2
            exit 2
            ;;
    esac
done

ask_cmake_option() {
    local variable_name="$1"
    local prompt="$2"
    local default_value="$3"
    local answer
    local prompt_suffix

    if [[ "${default_value}" == ON ]]; then
        prompt_suffix='[Y/n]'
    else
        prompt_suffix='[y/N]'
    fi

    read -r -p "${prompt} ${prompt_suffix} " answer
    if [[ -z "${answer}" ]]; then
        answer="${default_value}"
    fi
    case "${answer}" in
        ON|on|y|Y|yes|YES) printf -v "${variable_name}" '%s' ON ;;
        OFF|off|n|N|no|NO) printf -v "${variable_name}" '%s' OFF ;;
        *)
            echo "Please answer y or n." >&2
            exit 2
            ;;
    esac
}

if "${configure_release}" || "${configure_debug}"; then
    echo
    echo "Common native CMake options (applied to every Linux build selected):"
    ask_cmake_option ENGINE_STATIC "Build the engine as a static library?" ON
    ask_cmake_option CRESSIM_NEO_BUILD_VIEWER "Build the viewer?" ON
    ask_cmake_option CRESSIM_NEO_BUILD_EXAMPLES "Build examples?" ON
    ask_cmake_option CRESSIM_NEO_BUILD_PYTHON "Build Python bindings?" OFF
    ask_cmake_option BUILD_TESTING "Enable tests?" ON
    ask_cmake_option CRESSIM_NEO_ENABLE_CLANG_TIDY "Enable clang-tidy during builds?" OFF
    ask_cmake_option CRESSIM_NEO_ENABLE_CUDA_INTEROP "Enable CUDA interop?" OFF
    ask_cmake_option CRESSIM_NEO_ENABLE_ULTRASOUND "Enable CRESSim-Ultrasound?" OFF

    if [[ "${CRESSIM_NEO_ENABLE_ULTRASOUND}" == ON && "${CRESSIM_NEO_ENABLE_CUDA_INTEROP}" == OFF ]]; then
        CRESSIM_NEO_ENABLE_CUDA_INTEROP=ON
        echo "CUDA interop was enabled because CRESSim-Ultrasound requires it."
    fi
fi

configure_native_build() {
    local name="$1"
    local build_type="$2"
    local build_dir="${BUILD_ROOT}/linux-${name}"

    echo
    echo "Configuring ${build_type} build: ${build_dir}"
    cmake -S "${REPO_ROOT}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE="${build_type}" \
        -DENGINE_STATIC="${ENGINE_STATIC}" \
        -DCRESSIM_NEO_BUILD_VIEWER="${CRESSIM_NEO_BUILD_VIEWER}" \
        -DCRESSIM_NEO_BUILD_EXAMPLES="${CRESSIM_NEO_BUILD_EXAMPLES}" \
        -DCRESSIM_NEO_BUILD_DOCS=OFF \
        -DCRESSIM_NEO_BUILD_PYTHON="${CRESSIM_NEO_BUILD_PYTHON}" \
        -DBUILD_TESTING="${BUILD_TESTING}" \
        -DCRESSIM_NEO_ENABLE_CLANG_TIDY="${CRESSIM_NEO_ENABLE_CLANG_TIDY}" \
        -DCRESSIM_NEO_ENABLE_CUDA_INTEROP="${CRESSIM_NEO_ENABLE_CUDA_INTEROP}" \
        -DCRESSIM_NEO_ENABLE_ULTRASOUND="${CRESSIM_NEO_ENABLE_ULTRASOUND}"
}

native_build_has_python_enabled() {
    local build_name="$1"
    local cache_file="${BUILD_ROOT}/linux-${build_name}/CMakeCache.txt"

    [[ -f "${cache_file}" ]] &&
        rg -q '^CRESSIM_NEO_BUILD_PYTHON:BOOL=ON$' "${cache_file}"
}

require_docs_python_package() {
    local build_name="$1"
    local build_dir="${BUILD_ROOT}/linux-${build_name}"
    local cache_file="${build_dir}/CMakeCache.txt"
    local package_dir="${build_dir}/bin/cressim_neo"

    if [[ ! -d "${build_dir}" || ! -f "${cache_file}" ]]; then
        echo "Documentation requires an existing configured build at: ${build_dir}" >&2
        echo "Configure it first by selecting Linux ${build_name^}." >&2
        exit 1
    fi
    if ! native_build_has_python_enabled "${build_name}"; then
        echo "Documentation requires Python bindings in: ${build_dir}" >&2
        echo "Reconfigure it with CRESSIM_NEO_BUILD_PYTHON=ON, then build the package." >&2
        exit 1
    fi
    if [[ ! -f "${package_dir}/__init__.py" ]] || ! compgen -G "${package_dir}/_cressim_neo*.so" > /dev/null; then
        echo "Documentation requires a built Python package in: ${package_dir}" >&2
        echo "Build it first with: cmake --build ${build_dir} --target cressim_neo_python cressim_neo_python_package_files" >&2
        exit 1
    fi
}

if "${configure_release}"; then
    configure_native_build release Release
fi

if "${configure_debug}"; then
    configure_native_build debug Debug
fi

if "${configure_docs}"; then
    default_docs_build="debug"
    if ! "${configure_debug}" && "${configure_release}"; then
        default_docs_build="release"
    fi

    read -r -p "Use which Python package for documentation (debug/release; default: ${default_docs_build})? " docs_build
    docs_build="${docs_build:-${default_docs_build}}"
    case "${docs_build}" in
        debug) docs_build_type="Debug" ;;
        release) docs_build_type="Release" ;;
        *)
            echo "Documentation package must be 'debug' or 'release'." >&2
            exit 2
            ;;
    esac

    docs_package_dir="${BUILD_ROOT}/linux-${docs_build}/bin"
    require_docs_python_package "${docs_build}"

    echo
    echo "Configuring documentation build: ${BUILD_ROOT}/docs-site"
    cmake -S "${REPO_ROOT}/docs" -B "${BUILD_ROOT}/docs-site" \
        -DCRESSIM_NEO_PYTHON_PACKAGE_DIR="${docs_package_dir}"
fi

echo
echo "Configuration complete."
if "${configure_docs}"; then
    echo "Build the documentation with: cmake --build build/docs-site --target docs-html"
fi
