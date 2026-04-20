param(
    [ValidateSet("web", "worker")]
    [string]$Service = "web",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "local_artifacts\\zeabur_${Service}_bundle"
}

$target = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspace $OutputDir))
}

powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "create_zeabur_embedded_bundle.ps1") -OutputDir $target

if ($Service -eq "worker") {
    Copy-Item -LiteralPath (Join-Path $target "worker.Dockerfile") -Destination (Join-Path $target "Dockerfile") -Force
    $manifestPath = Join-Path $target "UPLOAD_MANIFEST.txt"
    $manifestLines = @(
        "Zeabur worker bundle created at: $(Get-Date -Format s)",
        "Workspace: $workspace",
        "Bundle: $target",
        "",
        "This bundle is worker-specific.",
        "Root Dockerfile has been replaced with worker.Dockerfile to force the worker entrypoint in upload mode.",
        "Deploy this bundle only to the worker service."
    )
    Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8
    Write-Host "Worker bundle patched with worker Dockerfile: $target"
} else {
    $manifestPath = Join-Path $target "UPLOAD_MANIFEST.txt"
    $manifestLines = @(
        "Zeabur web bundle created at: $(Get-Date -Format s)",
        "Workspace: $workspace",
        "Bundle: $target",
        "",
        "This bundle is web-specific.",
        "Root Dockerfile keeps the web entrypoint.",
        "Deploy this bundle only to the web service."
    )
    Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8
    Write-Host "Web bundle ready: $target"
}
