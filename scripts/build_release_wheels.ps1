[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('base', 'cu126', 'cu130', 'cu132')]
    [string]$Lane,

    [string]$OutputRoot = 'dist/windows'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$packageDirectory = switch ($Lane) {
    'base'  { Join-Path $repoRoot 'packaging/base' }
    'cu126' { Join-Path $repoRoot 'packaging/cuda126' }
    'cu130' { Join-Path $repoRoot 'packaging/cuda130' }
    'cu132' { Join-Path $repoRoot 'packaging/cuda132' }
}

$outputRootPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
$outputDirectory = [System.IO.Path]::GetFullPath((Join-Path $outputRootPath $Lane))
$workspacePrefix = $repoRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
    [System.IO.Path]::DirectorySeparatorChar
if (-not $outputDirectory.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must be inside the repository: $outputDirectory"
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'python is required to run cibuildwheel.'
}
& python -c 'import cibuildwheel'
if ($LASTEXITCODE -ne 0) {
    throw 'cibuildwheel is required. Install it in the active Python environment: python -m pip install cibuildwheel'
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw 'MSVC is required. Run this script from a Visual Studio developer PowerShell.'
}
if ($env:VSCMD_ARG_TGT_ARCH -notin @('x64', 'amd64')) {
    throw (
        'An x64 MSVC environment is required. Open "x64 Native Tools Command Prompt for VS 2022" ' +
        'or start a Visual Studio developer PowerShell with the x64 target, then rerun this script.'
    )
}

$previousCudaPath = $env:CUDA_PATH
$previousCudaToolkitRoot = $env:CUDAToolkit_ROOT
$previousCudaCompiler = $env:CUDACXX
$previousCibuildwheelCachePath = $env:CIBW_CACHE_PATH
$previousCibuildwheelWindowsArchitectures = $env:CIBW_ARCHS_WINDOWS
try {
    # The bundled DXC runtime is x86-64 only. Do not let cibuildwheel's Windows
    # default add unsupported 32-bit builds alongside the AMD64 release wheels.
    $env:CIBW_ARCHS_WINDOWS = 'AMD64'

    if ([string]::IsNullOrWhiteSpace($env:CIBW_CACHE_PATH)) {
        # Avoid an incomplete or restricted per-user cache (notably the Microsoft Store
        # Python cache). This directory is ignored with the other build products and is
        # retained so downloaded CPython toolchains can be reused by later release builds.
        $env:CIBW_CACHE_PATH = Join-Path $repoRoot 'build\cibuildwheel-cache'
    }

    if ($Lane -ne 'base') {
        $cudaVersion = $Lane.Substring(2, 2) + '.' + $Lane.Substring(4)
        $cudaRoot = Join-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' "v$cudaVersion"
        $cudaCompiler = Join-Path $cudaRoot 'bin\nvcc.exe'
        if (-not (Test-Path -LiteralPath $cudaCompiler -PathType Leaf)) {
            throw "CUDA Toolkit $cudaVersion is required for $Lane. Expected: $cudaCompiler"
        }

        # cibuildwheel workers inherit these variables. CMake uses CUDACXX during its first
        # configure and FindCUDAToolkit uses CUDAToolkit_ROOT for the matching import libraries.
        $env:CUDA_PATH = $cudaRoot
        $env:CUDAToolkit_ROOT = $cudaRoot
        $env:CUDACXX = $cudaCompiler
    }

    if (Test-Path -LiteralPath $outputDirectory) {
        Remove-Item -LiteralPath $outputDirectory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

    & python -m cibuildwheel --platform windows $packageDirectory --output-dir $outputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "cibuildwheel failed for $Lane."
    }

    $wheels = @(Get-ChildItem -LiteralPath $outputDirectory -Filter '*.whl' -File)
    if ($wheels.Count -eq 0) {
        throw "cibuildwheel completed without producing any wheels for $Lane."
    }
    $wheels |
        ForEach-Object {
            & (Join-Path $PSScriptRoot 'verify_release_wheel.ps1') -Lane $Lane -Wheel $_.FullName
        }
}
finally {
    $env:CUDA_PATH = $previousCudaPath
    $env:CUDAToolkit_ROOT = $previousCudaToolkitRoot
    $env:CUDACXX = $previousCudaCompiler
    $env:CIBW_CACHE_PATH = $previousCibuildwheelCachePath
    $env:CIBW_ARCHS_WINDOWS = $previousCibuildwheelWindowsArchitectures
}
