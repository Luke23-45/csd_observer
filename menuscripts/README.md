# Manuscript Workspace

This directory contains the modular LaTeX workspace for the manuscript draft.

## Template Status

The current scaffold uses a temporary `article` class so that the section files
and helper scripts are usable immediately. After selecting a target journal,
update `main.tex` to use the journal template and remove any placeholder settings
that the template replaces.

## Layout

- `main.tex`: manuscript entrypoint
- `build/`: generated PDF and LaTeX build artifacts
- `sections/`: one subdirectory per main section
- `appendix/`: appendix material
- `bib/references.bib`: bibliography
- `figures/`: manuscript-managed figures
- `data/`: generated tables, figures, and report artifacts already available
- `scripts/`: PowerShell helper scripts for build and cleanup

## Writing Rule

Write and revise one section at a time. Every sentence must either define the
problem, sharpen the gap, state a result, delimit a claim, explain the
significance, or orient the reader.

## Build

To compile the LaTeX manuscript PDF:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

## Clean

To delete the compiler artifacts and the build folder:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean.ps1
```
