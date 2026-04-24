param(
  [switch]$SkipDeepModels,
  [switch]$SkipMining,
  [switch]$IncludeExtended
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
Set-Location $repoRoot

$logRoot = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\suite_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
  param(
    [string]$Step,
    [string]$Status,
    [double]$Seconds,
    [string]$Note
  )
  $results.Add([pscustomobject]@{
    step = $Step
    status = $Status
    seconds = [Math]::Round($Seconds, 3)
    note = $Note
  }) | Out-Null
}

function Get-LatestSummary {
  param([string]$OutputDir)
  if (-not (Test-Path -LiteralPath $OutputDir)) {
    return ""
  }
  $f = Get-ChildItem -Path $OutputDir -Filter "summary_*.json" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($null -eq $f) {
    return ""
  }
  return $f.FullName
}

function Get-ModelPathFromSummary {
  param(
    [string]$SummaryPath,
    [string]$Component
  )
  if (-not (Test-Path -LiteralPath $SummaryPath)) {
    return ""
  }
  try {
    $raw = Get-Content -Path $SummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
  }
  catch {
    return ""
  }
  $modelFiles = $null
  if ($raw.outputs -and $raw.outputs.model_files) {
    $modelFiles = $raw.outputs.model_files
  }
  elseif ($raw.model_files) {
    $modelFiles = $raw.model_files
  }
  if ($null -eq $modelFiles) {
    return ""
  }
  $compProp = $modelFiles.PSObject.Properties | Where-Object { $_.Name -eq $Component } | Select-Object -First 1
  if ($null -eq $compProp) {
    return ""
  }
  $entry = $compProp.Value
  foreach ($k in @("model_pt", "model_pkl", "model_file", "meta_json")) {
    $vProp = $entry.PSObject.Properties | Where-Object { $_.Name -eq $k } | Select-Object -First 1
    if ($vProp -and $vProp.Value) {
      return [string]$vProp.Value
    }
  }
  return ""
}

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Action
  )
  Write-Host "`n=== [SMOKE] $Name ==="
  $start = Get-Date
  try {
    & $Action
    $elapsed = ((Get-Date) - $start).TotalSeconds
    Add-Result -Step $Name -Status "passed" -Seconds $elapsed -Note ""
    Write-Host "[PASS] $Name (${elapsed}s)"
    return $true
  }
  catch {
    $elapsed = ((Get-Date) - $start).TotalSeconds
    $msg = $_.Exception.Message
    Add-Result -Step $Name -Status "failed" -Seconds $elapsed -Note $msg
    Write-Warning "[FAIL] $Name :: $msg"
    return $false
  }
}

function Add-Skipped {
  param(
    [string]$Name,
    [string]$Reason
  )
  Add-Result -Step $Name -Status "skipped" -Seconds 0 -Note $Reason
  Write-Host "[SKIP] $Name :: $Reason"
}

$scriptRoot = $PSScriptRoot

Invoke-Step -Name "run_strategy7_v2_01_train_tree_baseline" -Action {
  & (Join-Path $scriptRoot "run_strategy7_v2_01_train_tree_baseline.ps1")
} | Out-Null
$train01Out = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_01_train_tree"
$summaryTrain01 = Get-LatestSummary -OutputDir $train01Out

Invoke-Step -Name "run_strategy7_v2_02_train_tree_fe_dynamic" -Action {
  & (Join-Path $scriptRoot "run_strategy7_v2_02_train_tree_fe_dynamic.ps1")
} | Out-Null
$train02Out = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_02_train_tree_fe_dynamic"
$summaryTrain02 = Get-LatestSummary -OutputDir $train02Out

Invoke-Step -Name "run_strategy7_v2_03_train_custom_all_models" -Action {
  & (Join-Path $scriptRoot "run_strategy7_v2_03_train_custom_all_models.ps1")
} | Out-Null
$train03Out = Join-Path $repoRoot "Strategy7\outputs\smoke_v2\run_strategy7_03_train_custom_all"
$summaryTrain03 = Get-LatestSummary -OutputDir $train03Out

if ($summaryTrain02) {
  Invoke-Step -Name "run_strategy7_v2_04_load_from_summary_refit" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_04_load_from_summary_refit.ps1") -ModelSummaryJson $summaryTrain02
  } | Out-Null
}
else {
  Add-Skipped -Name "run_strategy7_v2_04_load_from_summary_refit" -Reason "missing summary from run_strategy7_v2_02"
}

if ($summaryTrain02) {
  Invoke-Step -Name "run_strategy7_v2_05_load_from_summary_strict" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_05_load_from_summary_strict.ps1") -ModelSummaryJson $summaryTrain02
  } | Out-Null
}
else {
  Add-Skipped -Name "run_strategy7_v2_05_load_from_summary_strict" -Reason "missing summary from run_strategy7_v2_02"
}

$modelsDir01 = Join-Path $train01Out "models"
if (Test-Path -LiteralPath $modelsDir01) {
  Invoke-Step -Name "run_strategy7_v2_06_load_from_models_dir_off" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_06_load_from_models_dir_off.ps1") -ModelsLoadDir $modelsDir01
  } | Out-Null
}
else {
  Add-Skipped -Name "run_strategy7_v2_06_load_from_models_dir_off" -Reason "missing models dir from run_strategy7_v2_01"
}

if ($summaryTrain03) {
  Invoke-Step -Name "run_strategy7_v2_07_load_custom_from_summary" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_07_load_custom_from_summary.ps1") -ModelSummaryJson $summaryTrain03
  } | Out-Null
}
else {
  Add-Skipped -Name "run_strategy7_v2_07_load_custom_from_summary" -Reason "missing summary from run_strategy7_v2_03"
}

if (-not $SkipDeepModels) {
  Invoke-Step -Name "run_strategy7_v2_08_train_factor_gcl_smoke" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_08_train_factor_gcl_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_strategy7_v2_09_train_dafat_smoke" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_09_train_dafat_smoke.ps1")
  } | Out-Null
}
else {
  Add-Skipped -Name "run_strategy7_v2_08_train_factor_gcl_smoke" -Reason "SkipDeepModels"
  Add-Skipped -Name "run_strategy7_v2_09_train_dafat_smoke" -Reason "SkipDeepModels"
}

Invoke-Step -Name "run_strategy7_v2_10_list_factors_export" -Action {
  & (Join-Path $scriptRoot "run_strategy7_v2_10_list_factors_export.ps1")
} | Out-Null

Invoke-Step -Name "run_strategy7_v2_11_factor_value_store_build_only" -Action {
  & (Join-Path $scriptRoot "run_strategy7_v2_11_factor_value_store_build_only.ps1")
} | Out-Null

if (-not $SkipMining) {
  Invoke-Step -Name "run_factor_mining_v2_01_fundamental_smoke" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_01_fundamental_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_factor_mining_v2_02_minute_parametric_smoke" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_02_minute_parametric_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_factor_mining_v2_03_minute_parametric_plus_smoke" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_03_minute_parametric_plus_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_factor_mining_v2_04_ml_ensemble_smoke" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_04_ml_ensemble_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_factor_mining_v2_05_gplearn_smoke" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_05_gplearn_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_factor_mining_v2_06_custom_smoke" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_06_custom_smoke.ps1")
  } | Out-Null

  Invoke-Step -Name "run_factor_mining_v2_07_list_factors_export" -Action {
    & (Join-Path $scriptRoot "run_factor_mining_v2_07_list_factors_export.ps1")
  } | Out-Null
}
else {
  Add-Skipped -Name "run_factor_mining_v2_01_fundamental_smoke" -Reason "SkipMining"
  Add-Skipped -Name "run_factor_mining_v2_02_minute_parametric_smoke" -Reason "SkipMining"
  Add-Skipped -Name "run_factor_mining_v2_03_minute_parametric_plus_smoke" -Reason "SkipMining"
  Add-Skipped -Name "run_factor_mining_v2_04_ml_ensemble_smoke" -Reason "SkipMining"
  Add-Skipped -Name "run_factor_mining_v2_05_gplearn_smoke" -Reason "SkipMining"
  Add-Skipped -Name "run_factor_mining_v2_06_custom_smoke" -Reason "SkipMining"
  Add-Skipped -Name "run_factor_mining_v2_07_list_factors_export" -Reason "SkipMining"
}

if ($IncludeExtended) {
  Invoke-Step -Name "run_strategy7_v2_12_train_weekly_volatility" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_12_train_weekly_volatility.ps1")
  } | Out-Null

  Invoke-Step -Name "run_strategy7_v2_13_train_30min_intraday_realistic" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_13_train_30min_intraday_realistic.ps1")
  } | Out-Null

  Invoke-Step -Name "run_strategy7_v2_14_train_price_only_mainboard" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_14_train_price_only_mainboard.ps1")
  } | Out-Null

  Invoke-Step -Name "run_strategy7_v2_15_train_custom_factor_plugin" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_15_train_custom_factor_plugin.ps1")
  } | Out-Null

  Invoke-Step -Name "run_strategy7_v2_16_train_factor_value_store_hydrate" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_16_train_factor_value_store_hydrate.ps1")
  } | Out-Null

  if ($summaryTrain02) {
    $stockPath = Get-ModelPathFromSummary -SummaryPath $summaryTrain02 -Component "stock_model"
    $timingPath = Get-ModelPathFromSummary -SummaryPath $summaryTrain02 -Component "timing_model"
    $portfolioPath = Get-ModelPathFromSummary -SummaryPath $summaryTrain02 -Component "portfolio_model"
    $executionPath = Get-ModelPathFromSummary -SummaryPath $summaryTrain02 -Component "execution_model"
    if ($stockPath -and $timingPath -and $portfolioPath -and $executionPath) {
      Invoke-Step -Name "run_strategy7_v2_17_load_explicit_paths_off" -Action {
        & (Join-Path $scriptRoot "run_strategy7_v2_17_load_explicit_paths_off.ps1") `
          -StockModelPath $stockPath `
          -TimingModelPath $timingPath `
          -PortfolioModelPath $portfolioPath `
          -ExecutionModelPath $executionPath
      } | Out-Null
    }
    else {
      Add-Skipped -Name "run_strategy7_v2_17_load_explicit_paths_off" -Reason "missing one or more model paths in summary from run_strategy7_v2_02"
    }
  }
  else {
    Add-Skipped -Name "run_strategy7_v2_17_load_explicit_paths_off" -Reason "missing summary from run_strategy7_v2_02"
  }

  Invoke-Step -Name "run_strategy7_v2_18_train_monthly_multitask_catalog_off" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_18_train_monthly_multitask_catalog_off.ps1")
  } | Out-Null

  Invoke-Step -Name "run_strategy7_v2_19_list_factors_30min_json_export" -Action {
    & (Join-Path $scriptRoot "run_strategy7_v2_19_list_factors_30min_json_export.ps1")
  } | Out-Null

  if (-not $SkipMining) {
    Invoke-Step -Name "run_factor_mining_v2_08_material_fe_value_store" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_08_material_fe_value_store.ps1")
    } | Out-Null

    Invoke-Step -Name "run_factor_mining_v2_09_custom_spec_json_smoke" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_09_custom_spec_json_smoke.ps1")
    } | Out-Null

    Invoke-Step -Name "run_factor_mining_v2_10_minute_parametric_30min_smoke" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_10_minute_parametric_30min_smoke.ps1")
    } | Out-Null

    Invoke-Step -Name "run_factor_mining_v2_11_price_only_mainboard_all" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_11_price_only_mainboard_all.ps1")
    } | Out-Null

    Invoke-Step -Name "run_factor_mining_v2_12_list_factors_markdown_export" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_12_list_factors_markdown_export.ps1")
    } | Out-Null

    Invoke-Step -Name "run_factor_mining_v2_13_disable_default_materials_with_factor_list" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_13_disable_default_materials_with_factor_list.ps1")
    } | Out-Null

    Invoke-Step -Name "run_factor_mining_v2_14_list_factors_with_custom_plugin" -Action {
      & (Join-Path $scriptRoot "run_factor_mining_v2_14_list_factors_with_custom_plugin.ps1")
    } | Out-Null
  }
  else {
    Add-Skipped -Name "run_factor_mining_v2_08_material_fe_value_store" -Reason "SkipMining"
    Add-Skipped -Name "run_factor_mining_v2_09_custom_spec_json_smoke" -Reason "SkipMining"
    Add-Skipped -Name "run_factor_mining_v2_10_minute_parametric_30min_smoke" -Reason "SkipMining"
    Add-Skipped -Name "run_factor_mining_v2_11_price_only_mainboard_all" -Reason "SkipMining"
    Add-Skipped -Name "run_factor_mining_v2_12_list_factors_markdown_export" -Reason "SkipMining"
    Add-Skipped -Name "run_factor_mining_v2_13_disable_default_materials_with_factor_list" -Reason "SkipMining"
    Add-Skipped -Name "run_factor_mining_v2_14_list_factors_with_custom_plugin" -Reason "SkipMining"
  }
}

$passed = @($results | Where-Object { $_.status -eq "passed" }).Count
$failed = @($results | Where-Object { $_.status -eq "failed" }).Count
$skipped = @($results | Where-Object { $_.status -eq "skipped" }).Count

$report = [pscustomobject]@{
  timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  repo_root = $repoRoot
  total = $results.Count
  passed = $passed
  failed = $failed
  skipped = $skipped
  results = $results
}

$jsonPath = Join-Path $logRoot "smoke_report_${ts}.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath -Encoding UTF8

$mdPath = Join-Path $logRoot "smoke_report_${ts}.md"
$md = @()
$md += "# Smoke Suite V2 Report"
$md += ""
$md += "- timestamp: $($report.timestamp)"
$md += "- total: $($report.total)"
$md += "- passed: $($report.passed)"
$md += "- failed: $($report.failed)"
$md += "- skipped: $($report.skipped)"
$md += ""
$md += "| step | status | seconds | note |"
$md += "|---|---:|---:|---|"
foreach ($r in $results) {
  $note = ($r.note -replace "\|", "\\|")
  $md += "| $($r.step) | $($r.status) | $($r.seconds) | $note |"
}
$md -join "`n" | Set-Content -Path $mdPath -Encoding UTF8

Write-Host "`n=== Smoke Suite V2 Finished ==="
Write-Host "passed=$passed failed=$failed skipped=$skipped"
Write-Host "json report: $jsonPath"
Write-Host "markdown report: $mdPath"

if ($failed -gt 0) {
  exit 1
}


