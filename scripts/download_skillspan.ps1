param(
    [string]$Target = "data/external/skillspan"
)

$ErrorActionPreference = "Stop"
$Commit = "2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b"
if (Test-Path -LiteralPath $Target) {
    $Actual = git -C $Target rev-parse HEAD
    if ($Actual -ne $Commit) {
        throw "SkillSpan exists at unexpected commit: $Actual"
    }
    Write-Host "SkillSpan already verified at $Commit"
    exit 0
}

git clone --no-checkout https://github.com/kris927b/SkillSpan.git $Target
git -C $Target fetch --depth 1 origin $Commit
git -C $Target checkout --detach $Commit
$Actual = git -C $Target rev-parse HEAD
if ($Actual -ne $Commit) {
    throw "SkillSpan commit verification failed: $Actual"
}
Write-Host "SkillSpan downloaded and verified at $Commit"
