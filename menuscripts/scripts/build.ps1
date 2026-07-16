param(
    [string]$MainFile = "main.tex",
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $root $BuildDir
$mainBase = [System.IO.Path]::GetFileNameWithoutExtension($MainFile)

New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
Set-Location $root

function Invoke-PdfLatexBuild {
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $buildPath $MainFile
    if (Test-Path ".\bib\references.bib") {
        $previousBibInputs = $env:BIBINPUTS
        $previousBstInputs = $env:BSTINPUTS
        try {
            $env:BIBINPUTS = "$root;$previousBibInputs"
            $env:BSTINPUTS = "$root;$previousBstInputs"
            Push-Location $buildPath
            bibtex $mainBase
        }
        finally {
            Pop-Location
            $env:BIBINPUTS = $previousBibInputs
            $env:BSTINPUTS = $previousBstInputs
        }
    }
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $buildPath $MainFile
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $buildPath $MainFile
}

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    if (Get-Command perl -ErrorAction SilentlyContinue) {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$buildPath $MainFile
        if ($LASTEXITCODE -eq 0) {
            exit 0
        }
    }
}

Invoke-PdfLatexBuild
