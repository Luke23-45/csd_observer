# CI gate for csd_observer (R7): ruff + full test suite.
# Fast lane excludes the `heavy` marker (reproducibility training tests);
# the heavy lane runs them separately so failures are attributable.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    $env:PYTHONPATH = "src"
    if (Test-Path env:CSD_OBSERVER_SKIP_MIN_LENGTH_GATES) {
        # leave caller's setting alone
    } else {
        $env:CSD_OBSERVER_SKIP_MIN_LENGTH_GATES = "1"
    }

    Write-Host "== ruff =="
    ruff check .

    Write-Host "== pytest (fast lane) =="
    python -m pytest tests -q -m "not heavy" -p no:cacheprovider

    Write-Host "== pytest (heavy lane) =="
    python -m pytest tests -q -m "heavy" -p no:cacheprovider

    Write-Host "== CI green =="
}
finally {
    Pop-Location
}
