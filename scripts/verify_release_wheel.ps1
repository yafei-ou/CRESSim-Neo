[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('base', 'cu126', 'cu130', 'cu132')]
    [string]$Lane,

    [Parameter(Mandatory = $true)]
    [string]$Wheel
)

$ErrorActionPreference = 'Stop'

$wheelPath = [System.IO.Path]::GetFullPath($Wheel)
if (-not (Test-Path -LiteralPath $wheelPath -PathType Leaf)) {
    throw "Wheel does not exist: $wheelPath"
}
if ([System.IO.Path]::GetFileName($wheelPath) -notlike '*-win_amd64.whl') {
    throw "Windows release wheel must target win_amd64: $wheelPath"
}
if (-not (Get-Command dumpbin -ErrorAction SilentlyContinue)) {
    throw 'dumpbin is required. Run this script from a Visual Studio developer PowerShell.'
}

$expectedCudart = switch ($Lane) {
    'cu126' { 'cudart64_12.dll' }
    'cu130' { 'cudart64_13.dll' }
    'cu132' { 'cudart64_13.dll' }
    default { $null }
}
$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("cressim-neo-wheel-" + [guid]::NewGuid())
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($wheelPath, $temporaryDirectory)
    $payloadFiles = Get-ChildItem -LiteralPath $temporaryDirectory -File -Recurse
    if (-not ($payloadFiles | Where-Object {
        $_.FullName -match '[\\/]cressim_neo[\\/]__init__\.pyi$'
    })) {
        throw "Release wheel is missing cressim_neo/__init__.pyi: $wheelPath"
    }
    $bundledCudaLibraries = $payloadFiles | Where-Object {
        $_.Name -match '^(cudart64|cufft64|curand64|nvjitlink64).*\.dll$'
    }
    if ($bundledCudaLibraries) {
        throw "CUDA runtime DLLs must not be bundled: $($bundledCudaLibraries.FullName -join ', ')"
    }

    $nativeFiles = $payloadFiles | Where-Object { $_.Extension -in '.dll', '.pyd' }
    $dependencies = @()
    foreach ($nativeFile in $nativeFiles) {
        $dumpbinOutput = & dumpbin /DEPENDENTS $nativeFile.FullName 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "dumpbin failed for $($nativeFile.FullName): $dumpbinOutput"
        }
        $dependencies += $dumpbinOutput |
            ForEach-Object { $_.ToString().Trim() } |
            Where-Object { $_ -match '\.dll$' } |
            ForEach-Object { $_.ToLowerInvariant() }
    }
    $dependencies = $dependencies | Select-Object -Unique
    $packagedDllNames = $nativeFiles |
        Where-Object { $_.Extension -eq '.dll' } |
        ForEach-Object { $_.Name.ToLowerInvariant() }
    $missingCressimDependencies = $dependencies |
        Where-Object { $_ -match '^cressim_neo_.*\.dll$' -and $_ -notin $packagedDllNames }
    if ($missingCressimDependencies) {
        throw "Wheel is missing CRESSim runtime DLLs: $($missingCressimDependencies -join ', ')"
    }
    $cudaDependencies = $dependencies | Where-Object {
        $_ -match '^(cudart64|cufft64|curand64|nvjitlink64).*\.dll$'
    }

    if ($Lane -eq 'base') {
        if ($cudaDependencies) {
            throw "Base wheel must not depend on CUDA DLLs: $($cudaDependencies -join ', ')"
        }
    }
    elseif ($dependencies -notcontains $expectedCudart) {
        throw "$Lane wheel does not depend on $expectedCudart. Found CUDA dependencies: $($cudaDependencies -join ', ')"
    }

    Write-Host "Verified Windows $Lane wheel: $wheelPath"
}
finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
    }
}
