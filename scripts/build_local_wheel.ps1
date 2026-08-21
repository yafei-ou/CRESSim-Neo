[CmdletBinding()]
param([string]$Output = 'dist')

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([IO.Path]::IsPathRooted($Output)) {
    $wheelDirectory = $Output
}
else {
    $wheelDirectory = Join-Path $repositoryRoot $Output
}
New-Item -ItemType Directory -Force -Path $wheelDirectory | Out-Null

Push-Location $repositoryRoot
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python was not found. Activate the intended environment first."
    }

    & python -m pip wheel --no-deps --wheel-dir $wheelDirectory `
        '-Ccmake.define.CRESSIM_NEO_BUILD_VIEWER=ON' .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
