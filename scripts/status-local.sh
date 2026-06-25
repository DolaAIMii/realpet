#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

failures=0
fail() { printf "FAIL: %s\n" "$1"; failures=$((failures + 1)); }
pass() { printf "PASS: %s\n" "$1"; }

printf "== Workspace ==\nPath: %s\n" "$ROOT_DIR"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short --branch
  git diff --check && pass "git diff --check" || fail "git diff --check"
  git diff --cached --quiet && pass "no staged changes" || printf "WARN: staged changes exist\n"
else
  printf "WARN: not a git repository\n"
fi

printf "\n== Required Governance Files ==\n"
required_files=(
  "README.md"
  "PROJECT_INDEX.md"
  "governance/PROJECT_GOVERNANCE.md"
  "governance/WINDOW_START_PROMPTS.md"
  "branding/README.md"
  "branding/BRANDING_GOVERNANCE.md"
  "branding/BRANDING_STATUS.md"
  "branding/BRANDING_ASSET_WORKFLOW.md"
  "branding/MASCOT_HEAD_PROMPT_PACK.md"
  "product/PRODUCT_STATUS.md"
  "product/PRODUCT_PRD.md"
  "uiux/UIUX_STATUS.md"
  "uiux/README.md"
  "tech/TECH_STATUS.md"
  "tech/TECH_ARCHITECTURE.md"
  "modules/README.md"
  "quality/QA_STATUS.md"
  "quality/QUALITY_GOVERNANCE.md"
  "research/RESEARCH_STATUS.md"
  "research/RESEARCH_GOVERNANCE.md"
)
for file in "${required_files[@]}"; do
  [[ -f "$file" ]] && pass "$file" || fail "missing $file"
done

if [[ -d "backend/tests" ]]; then
  printf "\n== Backend Tests ==\n"
  if command -v python3 >/dev/null 2>&1; then
    python3 -m pytest backend/tests && pass "backend pytest" || fail "backend pytest"
  else
    printf "WARN: python3 unavailable; skipping backend tests\n"
  fi
fi

printf "\n== Result ==\n"
if [[ "$failures" -eq 0 ]]; then
  pass "local workflow status checks passed"
else
  printf "FAIL: %s check group(s) failed\n" "$failures"
fi
exit "$failures"
