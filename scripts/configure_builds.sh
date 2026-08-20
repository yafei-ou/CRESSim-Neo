#!/usr/bin/env bash
# Configure the supported shared-library native development builds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/build"

usage() {
    cat <<'EOF'
Usage: scripts/configure_builds.sh

Interactive helper for configuring Release and/or Debug shared-library CMake
builds. Building and installation use standard CMake commands after setup.

Existing build directories retain their CMake generator. New build directories
use Ninja when it is available, unless Unix Makefiles is selected.

Static C++ SDK installation is not a supported workflow in this helper.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

print_command() {
    printf ' +'
    printf ' %q' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    "$@"
}

ask_yes_no() {
    local variable_name="$1"
    local prompt="$2"
    local default_value="$3"
    local answer
    local suffix='[y/N]'

    if [[ "${default_value}" == ON ]]; then
        suffix='[Y/n]'
    fi

    read -r -p "${prompt} ${suffix} " answer
    if [[ -z "${answer}" ]]; then
        answer="${default_value}"
    fi

    case "${answer}" in
        ON|on|y|Y|yes|YES) printf -v "${variable_name}" '%s' ON ;;
        OFF|off|n|N|no|NO) printf -v "${variable_name}" '%s' OFF ;;
        *) die "Please answer y or n." ;;
    esac
}

cache_value() {
    local cache_file="$1"
    local variable_name="$2"

    [[ -f "${cache_file}" ]] || return 0
    sed -n "s/^${variable_name}:[^=]*=//p" "${cache_file}" | head -n 1
}

build_dir_for() {
    printf '%s/linux-%s' "${BUILD_ROOT}" "$1"
}

configure_native() {
    local build_name="$1"
    local build_type="$2"
    local build_dir
    local cache_file
    local existing_generator
    local -a command

    build_dir="$(build_dir_for "${build_name}")"
    cache_file="${build_dir}/CMakeCache.txt"
    command=(cmake -S "${REPO_ROOT}" -B "${build_dir}")

    if [[ -f "${cache_file}" ]]; then
        existing_generator="$(cache_value "${cache_file}" CMAKE_GENERATOR)"
        echo "Reusing ${build_name} build directory and generator: ${existing_generator}"
    else
        command+=(-G "${NEW_BUILD_GENERATOR}")
        echo "Creating ${build_name} build directory with generator: ${NEW_BUILD_GENERATOR}"
    fi

    command+=(
        "-DCMAKE_BUILD_TYPE=${build_type}"
        -DENGINE_STATIC=OFF
        "-DCRESSIM_NEO_BUILD_VIEWER=${CRESSIM_NEO_BUILD_VIEWER}"
        "-DCRESSIM_NEO_BUILD_EXAMPLES=${CRESSIM_NEO_BUILD_EXAMPLES}"
        -DCRESSIM_NEO_BUILD_DOCS=OFF
        "-DCRESSIM_NEO_BUILD_PYTHON=${CRESSIM_NEO_BUILD_PYTHON}"
        "-DBUILD_TESTING=${BUILD_TESTING}"
        "-DCRESSIM_NEO_ENABLE_CLANG_TIDY=${CRESSIM_NEO_ENABLE_CLANG_TIDY}"
        "-DCRESSIM_NEO_ENABLE_CUDA_INTEROP=${CRESSIM_NEO_ENABLE_CUDA_INTEROP}"
        "-DCRESSIM_NEO_ENABLE_ULTRASOUND=${CRESSIM_NEO_ENABLE_ULTRASOUND}"
        "-DCRESSIM_NEO_CUDA_RUNTIME_PROVIDER=${CRESSIM_NEO_CUDA_RUNTIME_PROVIDER}"
    )
    if [[ -n "${INSTALL_PREFIX}" ]]; then
        command+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}")
    fi

    echo
    echo "Configuring ${build_type} shared SDK build: ${build_dir}"
    run_command "${command[@]}"
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

echo "CRESSim-Neo shared SDK configuration"
read -r -p $'Select native profiles (comma-separated; default: release):\n  release) Release shared SDK\n  debug)   Debug shared SDK\n  both)    Release and Debug shared SDK\n> ' selections
selections="${selections:-release}"
selections="${selections//,/ }"

NATIVE_BUILDS=()
for selection in ${selections}; do
    case "${selection}" in
        release) NATIVE_BUILDS+=(release) ;;
        debug) NATIVE_BUILDS+=(debug) ;;
        both) NATIVE_BUILDS+=(release debug) ;;
        *) die "Unknown profile '${selection}'. Use release, debug, or both." ;;
    esac
done

unique_native_builds=()
for build_name in "${NATIVE_BUILDS[@]}"; do
    if [[ " ${unique_native_builds[*]} " != *" ${build_name} "* ]]; then
        unique_native_builds+=("${build_name}")
    fi
done
NATIVE_BUILDS=("${unique_native_builds[@]}")

echo
echo "Supported shared-SDK features (applied to every selected native build):"
ask_yes_no CRESSIM_NEO_BUILD_VIEWER "Build the interactive viewer?" ON
ask_yes_no CRESSIM_NEO_BUILD_EXAMPLES "Build examples?" ON
ask_yes_no CRESSIM_NEO_BUILD_PYTHON "Build Python bindings?" OFF
ask_yes_no BUILD_TESTING "Enable tests?" OFF
ask_yes_no CRESSIM_NEO_ENABLE_CLANG_TIDY "Enable clang-tidy during builds?" OFF
ask_yes_no CRESSIM_NEO_ENABLE_CUDA_INTEROP "Enable CUDA interop?" OFF
ask_yes_no CRESSIM_NEO_ENABLE_ULTRASOUND "Enable CRESSim-Ultrasound?" OFF

if [[ "${CRESSIM_NEO_ENABLE_ULTRASOUND}" == ON && "${CRESSIM_NEO_ENABLE_CUDA_INTEROP}" == OFF ]]; then
    CRESSIM_NEO_ENABLE_CUDA_INTEROP=ON
    echo "CUDA interop was enabled because CRESSim-Ultrasound requires it."
fi

CRESSIM_NEO_CUDA_RUNTIME_PROVIDER=AUTO
if [[ "${CRESSIM_NEO_ENABLE_CUDA_INTEROP}" == ON ]]; then
    read -r -p "CUDA runtime provider (auto/system/managed; default: auto)? " cuda_provider_answer
    cuda_provider_answer="${cuda_provider_answer:-auto}"
    case "${cuda_provider_answer,,}" in
        auto) CRESSIM_NEO_CUDA_RUNTIME_PROVIDER=AUTO ;;
        system) CRESSIM_NEO_CUDA_RUNTIME_PROVIDER=SYSTEM ;;
        managed) CRESSIM_NEO_CUDA_RUNTIME_PROVIDER=MANAGED ;;
        *) die "CUDA runtime provider must be auto, system, or managed." ;;
    esac
fi

needs_new_generator=OFF
for build_name in "${NATIVE_BUILDS[@]}"; do
    if [[ ! -f "$(build_dir_for "${build_name}")/CMakeCache.txt" ]]; then
        needs_new_generator=ON
    fi
done
if [[ "${needs_new_generator}" == ON ]]; then
    if command -v ninja >/dev/null 2>&1; then
        default_generator=ninja
    else
        default_generator=make
    fi
    read -r -p "Generator for new build directories (ninja/make; default: ${default_generator})? " generator_answer
    generator_answer="${generator_answer:-${default_generator}}"
    case "${generator_answer}" in
        ninja)
            command -v ninja >/dev/null 2>&1 || die "Ninja was selected but is not on PATH."
            NEW_BUILD_GENERATOR=Ninja
            ;;
        make|makefiles)
            NEW_BUILD_GENERATOR='Unix Makefiles'
            ;;
        *) die "Generator must be 'ninja' or 'make'." ;;
    esac
else
    NEW_BUILD_GENERATOR=""
    echo "All selected native build directories already exist; their generators will be preserved."
fi

read -r -p "Install prefix (empty: each build directory's configured default)? " INSTALL_PREFIX
if [[ -n "${INSTALL_PREFIX}" && "${INSTALL_PREFIX}" != /* ]]; then
    die "Install prefix must be an absolute path or left empty."
fi

for build_name in "${NATIVE_BUILDS[@]}"; do
    case "${build_name}" in
        release) configure_native release Release ;;
        debug) configure_native debug Debug ;;
    esac
done

echo
echo "Configuration complete."
for build_name in "${NATIVE_BUILDS[@]}"; do
    build_dir="$(build_dir_for "${build_name}")"
    install_prefix="$(cache_value "${build_dir}/CMakeCache.txt" CMAKE_INSTALL_PREFIX)"
    echo "${build_name^}: ${build_dir}"
    echo "  Build:       cmake --build ${build_dir} --parallel"
    echo "  C++ SDK:     cmake --install ${build_dir} --component CXXSDK"
    if [[ "${CRESSIM_NEO_BUILD_PYTHON}" == ON ]]; then
        echo "  Python:      cmake --install ${build_dir} --component Python"
    fi
    [[ -n "${install_prefix}" ]] && echo "  Prefix:      ${install_prefix}"
done
