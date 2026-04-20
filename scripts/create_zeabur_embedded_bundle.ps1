param(
    [string]$OutputDir = "local_artifacts\\zeabur_embedded_bundle"
)

$ErrorActionPreference = "Stop"

$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$target = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $workspace $OutputDir))
}

$artifactsRoot = Join-Path $workspace "local_artifacts"
$tempArchiveDir = Join-Path $artifactsRoot ("_tmp_archive_for_embedded_" + [Guid]::NewGuid().ToString("N"))

New-Item -ItemType Directory -Path $artifactsRoot -Force | Out-Null

try {
    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "create_zeabur_archive_bundle.ps1") -OutputDir $tempArchiveDir

    $archivePath = Join-Path $tempArchiveDir "app_bundle.zip"
    if (-not (Test-Path -LiteralPath $archivePath)) {
        throw "Archive bundle missing app_bundle.zip: $archivePath"
    }

    if (Test-Path -LiteralPath $target) {
        try {
            Remove-Item -LiteralPath $target -Recurse -Force
        } catch {
            $fallbackName = ([System.IO.Path]::GetFileName($target)) + "_" + (Get-Date -Format "yyyyMMdd_HHmmss")
            $target = Join-Path ([System.IO.Path]::GetDirectoryName($target)) $fallbackName
        }
    }
    New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($target)) -Force | Out-Null
    New-Item -ItemType Directory -Path $target | Out-Null

    $rootFiles = @(
        ".dockerignore",
        "Dockerfile",
        "worker.Dockerfile",
        "pyproject.toml",
        "README.md",
        "requirements.txt",
        "requirements-dev.txt"
    )

    foreach ($relativePath in $rootFiles) {
        Copy-Item -LiteralPath (Join-Path $workspace $relativePath) -Destination (Join-Path $target $relativePath) -Force
    }

    $base64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($archivePath))
    $payloadPath = Join-Path $target "app_bundle_payload.py"
    $payload = @"
from __future__ import annotations

import base64
import io
import pathlib
import sys
import zipfile

ARCHIVE_B64 = """$base64"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python app_bundle_payload.py <target_dir>")
    target = pathlib.Path(sys.argv[1]).resolve()
    archive_bytes = base64.b64decode(ARCHIVE_B64.encode("ascii"))
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        zf.extractall(target)


if __name__ == "__main__":
    main()
"@
    Set-Content -LiteralPath $payloadPath -Value $payload -Encoding utf8

    $manifestPath = Join-Path $target "UPLOAD_MANIFEST.txt"
    $manifestLines = @(
        "Zeabur embedded bundle created at: $(Get-Date -Format s)",
        "Workspace: $workspace",
        "Bundle: $target",
        "Payload: app_bundle_payload.py",
        "",
        "Upload this whole folder.",
        "Dockerfile and worker.Dockerfile will reconstruct the full source tree from app_bundle_payload.py."
    )
    Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8

    Write-Host "Embedded bundle created: $target"
    Write-Host "Payload: $payloadPath"
} finally {
    if (Test-Path -LiteralPath $tempArchiveDir) {
        try {
            Remove-Item -LiteralPath $tempArchiveDir -Recurse -Force
        } catch {
        }
    }
}
