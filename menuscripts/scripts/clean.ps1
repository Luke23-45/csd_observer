$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $root "build"
Set-Location $root

$legacyArtifacts = @(
    "main.aux", "main.bbl", "main.blg", "main.fdb_latexmk", "main.fls",
    "main.log", "main.out", "main.run.xml", "main.synctex.gz", "main.toc",
    "main.lof", "main.lot", "main.pdf"
)

if (Test-Path $buildPath) {
    Remove-Item -LiteralPath $buildPath -Recurse -Force
}

foreach ($artifact in $legacyArtifacts) {
    $artifactPath = Join-Path $root $artifact
    if (Test-Path $artifactPath) {
        Remove-Item -LiteralPath $artifactPath -Force
    }
}
