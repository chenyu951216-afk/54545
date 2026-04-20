param(
    [string]$OutputDir = "_zeabur_wheel_bundle"
)

$ErrorActionPreference = "Stop"

$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$target = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspace $OutputDir))
}

if (-not $target.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must stay inside the workspace: $workspace"
}

if (Test-Path -LiteralPath $target) {
    try {
        Remove-Item -LiteralPath $target -Recurse -Force
    } catch {
        $fallbackName = ([System.IO.Path]::GetFileName($target)) + "_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        $target = Join-Path ([System.IO.Path]::GetDirectoryName($target)) $fallbackName
    }
}

New-Item -ItemType Directory -Path $target | Out-Null
$tempRoot = Join-Path $target ".tmp"
New-Item -ItemType Directory -Path $tempRoot | Out-Null

$requiredRootFiles = @(
    ".dockerignore",
    "Dockerfile",
    "worker.Dockerfile",
    "pyproject.toml",
    "README.md",
    "requirements.txt"
)

foreach ($relativePath in $requiredRootFiles) {
    $sourcePath = Join-Path $workspace $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required file is missing: $relativePath"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $target $relativePath) -Force
}

$wheelOut = Join-Path $target "wheelhouse"
New-Item -ItemType Directory -Path $wheelOut | Out-Null
Push-Location $workspace
try {
    $env:TMP = $tempRoot
    $env:TEMP = $tempRoot
    $env:PIP_BUILD_TRACKER = Join-Path $tempRoot "build-tracker"
    New-Item -ItemType Directory -Path $env:PIP_BUILD_TRACKER -Force | Out-Null
    py -m pip wheel . --no-deps --no-build-isolation --wheel-dir $wheelOut
} finally {
    Pop-Location
}

$wheel = Get-ChildItem -LiteralPath $wheelOut -Filter "*.whl" | Select-Object -First 1
if (-not $wheel) {
    throw "Wheel build failed: no .whl file produced."
}

Copy-Item -LiteralPath $wheel.FullName -Destination (Join-Path $target $wheel.Name) -Force
Remove-Item -LiteralPath $wheelOut -Recurse -Force
Remove-Item -LiteralPath $tempRoot -Recurse -Force

$manifestPath = Join-Path $target "UPLOAD_MANIFEST.txt"
$manifestLines = @(
    "Zeabur wheel bundle created at: $(Get-Date -Format s)",
    "Workspace: $workspace",
    "Bundle: $target",
    "Wheel: $($wheel.Name)",
    "",
    "Upload this whole folder.",
    "This bundle avoids relying on nested src uploads.",
    "Dockerfile and worker.Dockerfile will install the root-level wheel first."
)
Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8

Write-Host "Wheel bundle created: $target"
Write-Host "Wheel: $($wheel.Name)"
