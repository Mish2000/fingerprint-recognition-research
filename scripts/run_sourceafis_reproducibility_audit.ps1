[CmdletBinding()]
param(
    [string]$Python = "C:\Users\sirak\miniconda3\envs\fingerprint-recognition-research\python.exe",
    [string]$DataRoot = "C:\fingerprint-datasets",
    [string]$AuditRoot = "results\reproducibility_audits\sourceafis_reproducibility_audit_v1",
    [string]$SidecarJar = "apps\sourceafis-sidecar\target\sourceafis-sidecar-0.2.0.jar",
    [switch]$StrictJarHash,
    [switch]$SkipEditableInstall
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$previousLocation = Get-Location

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArguments
    )

    Write-Host ""
    Write-Host "=== $Phase ===" -ForegroundColor Cyan
    & $Python @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Phase failed with exit code $LASTEXITCODE. Later audit phases were not run."
    }
}

try {
    Set-Location $workspace

    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable does not exist: $Python"
    }
    if (-not (Test-Path -LiteralPath $SidecarJar -PathType Leaf)) {
        throw "SourceAFIS sidecar JAR does not exist: $SidecarJar"
    }
    if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
        throw "Dataset root does not exist: $DataRoot"
    }

    if (-not $SkipEditableInstall) {
        Invoke-CheckedPython -Phase "Editable installation" -PythonArguments @(
            "-m", "pip", "install", "-e", "."
        )
    }

    Invoke-CheckedPython -Phase "Freeze deterministic audit plan" -PythonArguments @(
        "-m", "fingerprint_benchmark.reproducibility_audit", "prepare",
        "--primary-results-root", "results",
        "--audit-root", $AuditRoot,
        "--seed", "sourceafis_reproducibility_audit_v1",
        "--low-positive-count", "100",
        "--positive-sample-count", "100"
    )

    $runArguments = @(
        "-m", "fingerprint_benchmark.reproducibility_audit", "run",
        "--audit-root", $AuditRoot,
        "--data-root", $DataRoot,
        "--sidecar-jar", $SidecarJar,
        "--skip-existing"
    )
    $compareArguments = @(
        "-m", "fingerprint_benchmark.reproducibility_audit", "compare",
        "--audit-root", $AuditRoot,
        "--score-abs-tolerance", "0"
    )
    if (-not $StrictJarHash) {
        $runArguments += "--allow-jar-hash-variation"
        $compareArguments += "--allow-jar-hash-variation"
    }

    Invoke-CheckedPython -Phase "Run isolated SourceAFIS audit bundles" -PythonArguments $runArguments
    Invoke-CheckedPython -Phase "Compare primary and rerun score payloads" -PythonArguments $compareArguments

    $reportPath = Join-Path $AuditRoot "comparison\audit_report.json"
    $summaryPath = Join-Path $AuditRoot "comparison\condition_summary.csv"
    Write-Host ""
    Write-Host "Audit completed successfully." -ForegroundColor Green
    Write-Host "Report: $reportPath"
    Get-Content -Raw -LiteralPath $reportPath
    Import-Csv -LiteralPath $summaryPath |
        Format-Table dataset, protocol, selected_pair_count, nonreproducible_pair_count, `
        score_payload_sha256_equal, implementation_hash_equal, implementation_accepted, passed
}
finally {
    Set-Location $previousLocation
}
