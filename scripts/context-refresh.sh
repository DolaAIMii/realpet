#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

role="${1:-}"
if [[ -z "$role" || "$role" == "-h" || "$role" == "--help" ]]; then
  printf "Usage: ./scripts/context-refresh.sh <project|pd|tech|qa|research|tech-child|generic-child> [extra-file ...]\n"
  exit 0
fi

files=("README.md" "PROJECT_INDEX.md" "governance/PROJECT_GOVERNANCE.md" "governance/WINDOW_START_PROMPTS.md")

add_file() {
  local file="$1"
  local existing
  for existing in "${files[@]}"; do
    [[ "$existing" == "$file" ]] && return
  done
  files+=("$file")
}

case "$role" in
  project) add_file "governance/README.md" ;;
  pd) add_file "product/README.md"; add_file "product/PRODUCT_PRD.md"; add_file "product/PRODUCT_GOVERNANCE.md" ;;
  tech) add_file "tech/README.md"; add_file "tech/TECH_ARCHITECTURE.md"; add_file "tech/TECH_GOVERNANCE.md"; add_file "modules/README.md"; add_file "product/PRODUCT_PRD.md" ;;
  qa) add_file "quality/README.md"; add_file "quality/QUALITY_GOVERNANCE.md"; add_file "product/PRODUCT_PRD.md"; add_file "tech/TECH_ARCHITECTURE.md"; add_file "modules/README.md" ;;
  research) add_file "research/README.md"; add_file "research/RESEARCH_GOVERNANCE.md"; add_file "product/README.md"; add_file "tech/README.md" ;;
  tech-child) add_file "tech/README.md"; add_file "tech/TECH_GOVERNANCE.md"; add_file "modules/README.md" ;;
  generic-child) ;;
  *) printf "FAIL: unknown role: %s\n" "$role"; exit 1 ;;
esac

shift
for file in "$@"; do add_file "$file"; done

printf "== Context Refresh ==\nRole: %s\n\n== Files To Read ==\n" "$role"
missing=0
for file in "${files[@]}"; do
  if [[ -f "$file" ]]; then
    printf "OK   %s\n" "$file"
  else
    printf "MISS %s\n" "$file"
    missing=$((missing + 1))
  fi
done

cat <<'PROMPT'

== Refresh Prompt ==
【上下文刷新】

请先按本地项目规则重新读取上述文件，并输出：
- 已读取文件与版本
- 是否发现文件状态、版本或职责冲突
- 当前任务是否可以继续
- 如不能继续，需要升级给哪个主窗口

不要依赖旧聊天上下文，不要修改 PROJECT_INDEX.md，不要新增治理文件。
PROMPT

[[ "$missing" -eq 0 ]] || printf "\nWARN: %s file(s) missing; do not create automatically\n" "$missing"
