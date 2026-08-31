$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$targetDirectory = Join-Path $projectRoot "data\external\jth"
$recordUrl = "https://zenodo.org/api/records/21390581"

New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
$record = Invoke-RestMethod -Uri $recordUrl
$results = foreach ($file in $record.files) {
    $targetFile = Join-Path $targetDirectory $file.key
    Invoke-WebRequest -Uri $file.links.self -OutFile $targetFile
    $actual = (Get-FileHash -Algorithm MD5 -LiteralPath $targetFile).Hash.ToLowerInvariant()
    $expected = $file.checksum -replace "^md5:", ""
    if ($actual -ne $expected) {
        throw "Checksum mismatch: $($file.key)"
    }
    [PSCustomObject]@{
        File = $file.key
        Bytes = (Get-Item -LiteralPath $targetFile).Length
        MD5 = $actual
        Verified = $true
    }
}

$results | Format-Table -AutoSize
Write-Host "JTH downloaded and verified. License: CC BY-NC 4.0 (non-commercial)."
