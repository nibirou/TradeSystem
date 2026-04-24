#!/usr/bin/env bash
set -euo pipefail

skip_deep_models=false
skip_mining=false
include_extended=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-deep-models)
      skip_deep_models=true
      shift
      ;;
    --skip-mining)
      skip_mining=true
      shift
      ;;
    --include-extended)
      include_extended=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_root}/../../.." && pwd)"
cd "${repo_root}"

log_root="${repo_root}/Strategy7/outputs/smoke_v2/suite_logs"
mkdir -p "${log_root}"
ts="$(date +%Y%m%d_%H%M%S)"

results_tsv="$(mktemp)"

add_result() {
  local step="$1"
  local status="$2"
  local seconds="$3"
  local note="$4"
  printf '%s\t%s\t%s\t%s\n' "$step" "$status" "$seconds" "$note" >> "${results_tsv}"
}

get_latest_summary() {
  local output_dir="$1"
  local latest
  latest="$(ls -1t "${output_dir}"/summary_*.json 2>/dev/null | head -n 1 || true)"
  echo "${latest}"
}

get_model_path_from_summary() {
  local summary_path="$1"
  local component="$2"
  python - "$summary_path" "$component" <<'PY'
import json
import os
import sys

summary_path = sys.argv[1]
component = sys.argv[2]
if not summary_path or not os.path.exists(summary_path):
    print("")
    raise SystemExit(0)
try:
    with open(summary_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
except Exception:
    try:
        with open(summary_path, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
    except Exception:
        print("")
        raise SystemExit(0)
model_files = (raw.get("outputs", {}) or {}).get("model_files") or raw.get("model_files") or {}
entry = model_files.get(component) or {}
for k in ("model_pt", "model_pkl", "model_file", "meta_json"):
    v = entry.get(k)
    if v:
        print(str(v))
        raise SystemExit(0)
print("")
PY
}

invoke_step() {
  local name="$1"
  shift
  echo
  echo "=== [SMOKE] ${name} ==="
  local start_ns end_ns elapsed
  start_ns="$(date +%s%N)"
  if "$@"; then
    end_ns="$(date +%s%N)"
    elapsed="$(python - <<PY
start_ns = int(${start_ns})
end_ns = int(${end_ns})
print(round((end_ns - start_ns) / 1e9, 3))
PY
)"
    add_result "$name" "passed" "$elapsed" ""
    echo "[PASS] ${name} (${elapsed}s)"
    return 0
  fi
  local rc="$?"
  end_ns="$(date +%s%N)"
  elapsed="$(python - <<PY
start_ns = int(${start_ns})
end_ns = int(${end_ns})
print(round((end_ns - start_ns) / 1e9, 3))
PY
)"
  add_result "$name" "failed" "$elapsed" "exit_code=${rc}"
  echo "[FAIL] ${name} :: exit_code=${rc}" >&2
  return 1
}

add_skipped() {
  local name="$1"
  local reason="$2"
  add_result "$name" "skipped" "0" "$reason"
  echo "[SKIP] ${name} :: ${reason}"
}

invoke_step "run_strategy7_v2_01_train_tree_baseline" bash "${script_root}/run_strategy7_v2_01_train_tree_baseline.sh" || true
train01_out="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_01_train_tree"
summary_train01="$(get_latest_summary "${train01_out}")"

invoke_step "run_strategy7_v2_02_train_tree_fe_dynamic" bash "${script_root}/run_strategy7_v2_02_train_tree_fe_dynamic.sh" || true
train02_out="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_02_train_tree_fe_dynamic"
summary_train02="$(get_latest_summary "${train02_out}")"

invoke_step "run_strategy7_v2_03_train_custom_all_models" bash "${script_root}/run_strategy7_v2_03_train_custom_all_models.sh" || true
train03_out="${repo_root}/Strategy7/outputs/smoke_v2/run_strategy7_03_train_custom_all"
summary_train03="$(get_latest_summary "${train03_out}")"

if [[ -n "${summary_train02}" ]]; then
  invoke_step "run_strategy7_v2_04_load_from_summary_refit" bash "${script_root}/run_strategy7_v2_04_load_from_summary_refit.sh" --model-summary-json "${summary_train02}" || true
else
  add_skipped "run_strategy7_v2_04_load_from_summary_refit" "missing summary from run_strategy7_v2_02"
fi

if [[ -n "${summary_train02}" ]]; then
  invoke_step "run_strategy7_v2_05_load_from_summary_strict" bash "${script_root}/run_strategy7_v2_05_load_from_summary_strict.sh" --model-summary-json "${summary_train02}" || true
else
  add_skipped "run_strategy7_v2_05_load_from_summary_strict" "missing summary from run_strategy7_v2_02"
fi

models_dir01="${train01_out}/models"
if [[ -d "${models_dir01}" ]]; then
  invoke_step "run_strategy7_v2_06_load_from_models_dir_off" bash "${script_root}/run_strategy7_v2_06_load_from_models_dir_off.sh" --models-load-dir "${models_dir01}" || true
else
  add_skipped "run_strategy7_v2_06_load_from_models_dir_off" "missing models dir from run_strategy7_v2_01"
fi

if [[ -n "${summary_train03}" ]]; then
  invoke_step "run_strategy7_v2_07_load_custom_from_summary" bash "${script_root}/run_strategy7_v2_07_load_custom_from_summary.sh" --model-summary-json "${summary_train03}" || true
else
  add_skipped "run_strategy7_v2_07_load_custom_from_summary" "missing summary from run_strategy7_v2_03"
fi

if [[ "${skip_deep_models}" == "false" ]]; then
  invoke_step "run_strategy7_v2_08_train_factor_gcl_smoke" bash "${script_root}/run_strategy7_v2_08_train_factor_gcl_smoke.sh" || true
  invoke_step "run_strategy7_v2_09_train_dafat_smoke" bash "${script_root}/run_strategy7_v2_09_train_dafat_smoke.sh" || true
else
  add_skipped "run_strategy7_v2_08_train_factor_gcl_smoke" "SkipDeepModels"
  add_skipped "run_strategy7_v2_09_train_dafat_smoke" "SkipDeepModels"
fi

invoke_step "run_strategy7_v2_10_list_factors_export" bash "${script_root}/run_strategy7_v2_10_list_factors_export.sh" || true
invoke_step "run_strategy7_v2_11_factor_value_store_build_only" bash "${script_root}/run_strategy7_v2_11_factor_value_store_build_only.sh" || true

if [[ "${skip_mining}" == "false" ]]; then
  invoke_step "run_factor_mining_v2_01_fundamental_smoke" bash "${script_root}/run_factor_mining_v2_01_fundamental_smoke.sh" || true
  invoke_step "run_factor_mining_v2_02_minute_parametric_smoke" bash "${script_root}/run_factor_mining_v2_02_minute_parametric_smoke.sh" || true
  invoke_step "run_factor_mining_v2_03_minute_parametric_plus_smoke" bash "${script_root}/run_factor_mining_v2_03_minute_parametric_plus_smoke.sh" || true
  invoke_step "run_factor_mining_v2_04_ml_ensemble_smoke" bash "${script_root}/run_factor_mining_v2_04_ml_ensemble_smoke.sh" || true
  invoke_step "run_factor_mining_v2_05_gplearn_smoke" bash "${script_root}/run_factor_mining_v2_05_gplearn_smoke.sh" || true
  invoke_step "run_factor_mining_v2_06_custom_smoke" bash "${script_root}/run_factor_mining_v2_06_custom_smoke.sh" || true
  invoke_step "run_factor_mining_v2_07_list_factors_export" bash "${script_root}/run_factor_mining_v2_07_list_factors_export.sh" || true
else
  add_skipped "run_factor_mining_v2_01_fundamental_smoke" "SkipMining"
  add_skipped "run_factor_mining_v2_02_minute_parametric_smoke" "SkipMining"
  add_skipped "run_factor_mining_v2_03_minute_parametric_plus_smoke" "SkipMining"
  add_skipped "run_factor_mining_v2_04_ml_ensemble_smoke" "SkipMining"
  add_skipped "run_factor_mining_v2_05_gplearn_smoke" "SkipMining"
  add_skipped "run_factor_mining_v2_06_custom_smoke" "SkipMining"
  add_skipped "run_factor_mining_v2_07_list_factors_export" "SkipMining"
fi

if [[ "${include_extended}" == "true" ]]; then
  invoke_step "run_strategy7_v2_12_train_weekly_volatility" bash "${script_root}/run_strategy7_v2_12_train_weekly_volatility.sh" || true
  invoke_step "run_strategy7_v2_13_train_30min_intraday_realistic" bash "${script_root}/run_strategy7_v2_13_train_30min_intraday_realistic.sh" || true
  invoke_step "run_strategy7_v2_14_train_price_only_mainboard" bash "${script_root}/run_strategy7_v2_14_train_price_only_mainboard.sh" || true
  invoke_step "run_strategy7_v2_15_train_custom_factor_plugin" bash "${script_root}/run_strategy7_v2_15_train_custom_factor_plugin.sh" || true
  invoke_step "run_strategy7_v2_16_train_factor_value_store_hydrate" bash "${script_root}/run_strategy7_v2_16_train_factor_value_store_hydrate.sh" || true

  if [[ -n "${summary_train02}" ]]; then
    stock_path="$(get_model_path_from_summary "${summary_train02}" "stock_model")"
    timing_path="$(get_model_path_from_summary "${summary_train02}" "timing_model")"
    portfolio_path="$(get_model_path_from_summary "${summary_train02}" "portfolio_model")"
    execution_path="$(get_model_path_from_summary "${summary_train02}" "execution_model")"
    if [[ -n "${stock_path}" && -n "${timing_path}" && -n "${portfolio_path}" && -n "${execution_path}" ]]; then
      invoke_step "run_strategy7_v2_17_load_explicit_paths_off" bash "${script_root}/run_strategy7_v2_17_load_explicit_paths_off.sh" \
        --stock-model-path "${stock_path}" \
        --timing-model-path "${timing_path}" \
        --portfolio-model-path "${portfolio_path}" \
        --execution-model-path "${execution_path}" || true
    else
      add_skipped "run_strategy7_v2_17_load_explicit_paths_off" "missing one or more model paths in summary from run_strategy7_v2_02"
    fi
  else
    add_skipped "run_strategy7_v2_17_load_explicit_paths_off" "missing summary from run_strategy7_v2_02"
  fi

  invoke_step "run_strategy7_v2_18_train_monthly_multitask_catalog_off" bash "${script_root}/run_strategy7_v2_18_train_monthly_multitask_catalog_off.sh" || true
  invoke_step "run_strategy7_v2_19_list_factors_30min_json_export" bash "${script_root}/run_strategy7_v2_19_list_factors_30min_json_export.sh" || true

  if [[ "${skip_mining}" == "false" ]]; then
    invoke_step "run_factor_mining_v2_08_material_fe_value_store" bash "${script_root}/run_factor_mining_v2_08_material_fe_value_store.sh" || true
    invoke_step "run_factor_mining_v2_09_custom_spec_json_smoke" bash "${script_root}/run_factor_mining_v2_09_custom_spec_json_smoke.sh" || true
    invoke_step "run_factor_mining_v2_10_minute_parametric_30min_smoke" bash "${script_root}/run_factor_mining_v2_10_minute_parametric_30min_smoke.sh" || true
    invoke_step "run_factor_mining_v2_11_price_only_mainboard_all" bash "${script_root}/run_factor_mining_v2_11_price_only_mainboard_all.sh" || true
    invoke_step "run_factor_mining_v2_12_list_factors_markdown_export" bash "${script_root}/run_factor_mining_v2_12_list_factors_markdown_export.sh" || true
    invoke_step "run_factor_mining_v2_13_disable_default_materials_with_factor_list" bash "${script_root}/run_factor_mining_v2_13_disable_default_materials_with_factor_list.sh" || true
    invoke_step "run_factor_mining_v2_14_list_factors_with_custom_plugin" bash "${script_root}/run_factor_mining_v2_14_list_factors_with_custom_plugin.sh" || true
  else
    add_skipped "run_factor_mining_v2_08_material_fe_value_store" "SkipMining"
    add_skipped "run_factor_mining_v2_09_custom_spec_json_smoke" "SkipMining"
    add_skipped "run_factor_mining_v2_10_minute_parametric_30min_smoke" "SkipMining"
    add_skipped "run_factor_mining_v2_11_price_only_mainboard_all" "SkipMining"
    add_skipped "run_factor_mining_v2_12_list_factors_markdown_export" "SkipMining"
    add_skipped "run_factor_mining_v2_13_disable_default_materials_with_factor_list" "SkipMining"
    add_skipped "run_factor_mining_v2_14_list_factors_with_custom_plugin" "SkipMining"
  fi
fi

json_path="${log_root}/smoke_report_${ts}.json"
md_path="${log_root}/smoke_report_${ts}.md"

python - "${results_tsv}" "${json_path}" "${md_path}" "${repo_root}" <<'PY'
import json
import sys
from datetime import datetime

results_tsv, json_path, md_path, repo_root = sys.argv[1:5]
rows = []
with open(results_tsv, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        step, status, seconds, note = (line.split("\t", 3) + ["", "", "", ""])[:4]
        try:
            sec_v = round(float(seconds), 3)
        except Exception:
            sec_v = 0.0
        rows.append({"step": step, "status": status, "seconds": sec_v, "note": note})

passed = sum(1 for r in rows if r["status"] == "passed")
failed = sum(1 for r in rows if r["status"] == "failed")
skipped = sum(1 for r in rows if r["status"] == "skipped")

report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "repo_root": repo_root,
    "total": len(rows),
    "passed": passed,
    "failed": failed,
    "skipped": skipped,
    "results": rows,
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

md = []
md.append("# Smoke Suite V2 Report")
md.append("")
md.append(f"- timestamp: {report['timestamp']}")
md.append(f"- total: {report['total']}")
md.append(f"- passed: {report['passed']}")
md.append(f"- failed: {report['failed']}")
md.append(f"- skipped: {report['skipped']}")
md.append("")
md.append("| step | status | seconds | note |")
md.append("|---|---:|---:|---|")
for r in rows:
    note = str(r["note"]).replace("|", "\\|")
    md.append(f"| {r['step']} | {r['status']} | {r['seconds']} | {note} |")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"passed={passed} failed={failed} skipped={skipped}")
PY

summary_line="$(python - "${json_path}" <<'PY'
import json
import sys
raw = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(f"passed={raw['passed']} failed={raw['failed']} skipped={raw['skipped']}")
PY
)"

echo
echo "=== Smoke Suite V2 Finished ==="
echo "${summary_line}"
echo "json report: ${json_path}"
echo "markdown report: ${md_path}"

failed_count="$(python - "${json_path}" <<'PY'
import json
import sys
raw = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(raw['failed'])
PY
)"

if [[ "${failed_count}" != "0" ]]; then
  exit 1
fi
