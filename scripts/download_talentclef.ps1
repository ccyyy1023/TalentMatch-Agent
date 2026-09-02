param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\data\external\talentclef2026")
)

$ErrorActionPreference = "Stop"
$expectedMd5 = "431ec3b693ae1ba24fe04793f9c1f750"
$url = "https://zenodo.org/records/19652670/files/TaskA.zip?download=1"
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$archivePath = Join-Path $destinationPath "TaskA.zip"
$extractPath = Join-Path $destinationPath "TaskA"

New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri $url -OutFile $archivePath
}

$actualMd5 = (Get-FileHash -LiteralPath $archivePath -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    throw "TaskA.zip checksum mismatch. Expected $expectedMd5 but got $actualMd5."
}

if (-not (Test-Path -LiteralPath (Join-Path $extractPath "development"))) {
    New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force
}

$developmentEnglish = Join-Path $extractPath "development\en"
if (-not (Test-Path -LiteralPath (Join-Path $developmentEnglish "qrels.tsv"))) {
    throw "TalentCLEF extraction validation failed: development/en/qrels.tsv not found."
}

Write-Host "TalentCLEF Task A v0.3.0 is ready at $extractPath"
Write-Host "Verified MD5: $actualMd5"
