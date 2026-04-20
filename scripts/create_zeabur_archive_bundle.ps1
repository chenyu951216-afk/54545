param(
    [string]$OutputDir = "_zeabur_archive_bundle"
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
    $sourcePath = Join-Path $workspace $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required file is missing: $relativePath"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $target $relativePath) -Force
}

$pathsToArchive = @(
    "src",
    "docs",
    "scripts",
    "data/import_templates"
)

$archivePath = Join-Path $target "app_bundle.zip"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

$zip = [System.IO.Compression.ZipFile]::Open($archivePath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($relativePath in $pathsToArchive) {
        $sourcePath = Join-Path $workspace $relativePath
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Required path is missing: $relativePath"
        }

        $files = Get-ChildItem -LiteralPath $sourcePath -Recurse -File
        foreach ($file in $files) {
            if ($file.FullName -like '*\__pycache__\*' -or $file.Extension -in @('.pyc', '.pyo', '.pyd')) {
                continue
            }
            $relativeEntry = $file.FullName.Substring($workspace.Length).TrimStart('\', '/').Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $file.FullName,
                $relativeEntry,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
} finally {
    $zip.Dispose()
}

$requiredArchiveEntries = @(
    "src/tw_stock_ai/adapters/__init__.py",
    "src/tw_stock_ai/ai_adapters/__init__.py",
    "src/tw_stock_ai/routers/api.py",
    "src/tw_stock_ai/services/jobs.py",
    "src/tw_stock_ai/static/style.css",
    "src/tw_stock_ai/templates/base.html",
    "src/tw_stock_ai/prompts/candidate_selection_reason.txt"
)

$zip = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $entryNames = @{}
    foreach ($entry in $zip.Entries) {
        $entryNames[$entry.FullName.Replace("\", "/")] = $true
    }
    $missingEntries = @()
    foreach ($entry in $requiredArchiveEntries) {
        if (-not $entryNames.ContainsKey($entry)) {
            $missingEntries += $entry
        }
    }
    if ($missingEntries.Count -gt 0) {
        throw "Archive verification failed. Missing: $($missingEntries -join ', ')"
    }
    $backslashEntries = @($zip.Entries | Where-Object { $_.FullName.Contains('\') } | Select-Object -ExpandProperty FullName)
    if ($backslashEntries.Count -gt 0) {
        throw "Archive verification failed. Backslash entry names detected: $($backslashEntries -join ', ')"
    }
} finally {
    $zip.Dispose()
}

$manifestPath = Join-Path $target "UPLOAD_MANIFEST.txt"
$manifestLines = @(
    "Zeabur archive bundle created at: $(Get-Date -Format s)",
    "Workspace: $workspace",
    "Bundle: $target",
    "Archive: app_bundle.zip",
    "",
    "Upload this whole folder.",
    "The Dockerfiles will auto-extract app_bundle.zip during build.",
    "This avoids relying on Zeabur upload preserving nested src directories."
)
Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding utf8

Write-Host "Archive bundle created: $target"
Write-Host "Archive: $archivePath"
