param(
    [string]$OutputDir = "_zeabur_upload_bundle"
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

$pathsToCopy = @(
    ".dockerignore",
    "Dockerfile",
    "worker.Dockerfile",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "docs",
    "scripts",
    "src",
    "data/import_templates"
)

foreach ($relativePath in $pathsToCopy) {
    $sourcePath = Join-Path $workspace $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required path is missing: $relativePath"
    }
}

if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
}

New-Item -ItemType Directory -Path $target | Out-Null

foreach ($relativePath in $pathsToCopy) {
    $sourcePath = Join-Path $workspace $relativePath
    $destinationPath = Join-Path $target $relativePath
    $destinationParent = Split-Path -Parent $destinationPath
    if (-not (Test-Path -LiteralPath $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Recurse -Force
}

$requiredCheckPaths = @(
    "src/tw_stock_ai/adapters",
    "src/tw_stock_ai/ai_adapters",
    "src/tw_stock_ai/notifiers",
    "src/tw_stock_ai/prompts",
    "src/tw_stock_ai/routers",
    "src/tw_stock_ai/services",
    "src/tw_stock_ai/static",
    "src/tw_stock_ai/templates",
    "docs/zeabur.md"
)

$missingPaths = @()
foreach ($relativePath in $requiredCheckPaths) {
    $checkPath = Join-Path $target $relativePath
    if (-not (Test-Path -LiteralPath $checkPath)) {
        $missingPaths += $relativePath
    }
}

if ($missingPaths.Count -gt 0) {
    throw "Bundle verification failed. Missing: $($missingPaths -join ', ')"
}

$manifestPath = Join-Path $target "UPLOAD_MANIFEST.txt"
$manifestLines = @(
    "Zeabur upload bundle created at: $(Get-Date -Format s)",
    "Workspace: $workspace",
    "Bundle: $target",
    "",
    "Required paths verified:"
) + ($requiredCheckPaths | ForEach-Object { "- $_" }) + @(
    "",
    "Use this folder as the upload source.",
    "Do not upload individual changed files from the repo root."
)
Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8

Write-Host "Bundle created: $target"
Write-Host "Manifest: $manifestPath"
