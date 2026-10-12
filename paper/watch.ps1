# Rebuild main.pdf when LaTeX sources, bibliography, styles or figures change.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & latexmk -pdf -pvc -view=none main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
