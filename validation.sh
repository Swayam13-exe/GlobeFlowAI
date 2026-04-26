#!/usr/bin/env bash
# validation.sh — validate GlobeFlowAI OpenEnv submission.
# Usage: ./validation.sh <hf_space_url> [repo_dir]
#
# Checks the same 3 things the automated round runs (HF Space ping, Docker
# build, `openenv validate`) PLUS GlobeFlowAI-specific things from the live
# QnA criteria: training plots committed as images, README linking every
# deliverable, all 4 tasks reachable, /tasks and /state alive.
#
# Note on payload shape: GlobeFlowAI's /reset uses {"task_name": "..."},
# not {"task_id": "..."} — don't "fix" it.

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
TASKS=("easy" "medium" "hard" "crisis")

# --- TTY-aware colors ------------------------------------------------------
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# --- Args ------------------------------------------------------------------
PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <hf_space_url> [repo_dir]"
  echo "Example: $0 https://swayam14-openenv-workforce.hf.space ."
  exit 1
fi

REPO_DIR="$(cd "$REPO_DIR" && pwd)"
PING_URL="${PING_URL%/}"

# --- Helpers ---------------------------------------------------------------
log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
warn() { log "${YELLOW}WARN  ${NC} -- $1"; }
stop() { printf "\n${RED}${BOLD}Stopped at %s.${NC}\n" "$1"; exit 1; }

# --- Locate Python ---------------------------------------------------------
# On Git Bash / MINGW64 (Windows), `python3` often doesn't exist — only `python`.
PY=""
if command -v python3 &>/dev/null; then
  PY=python3
elif command -v python &>/dev/null; then
  PY=python
elif command -v py &>/dev/null; then
  PY="py -3"
fi

printf "\n${BOLD}==== GlobeFlowAI Validator ====${NC}\n"
log "Repo:   $REPO_DIR"
log "Ping:   $PING_URL"
log "Python: ${PY:-NOT FOUND}"
printf "\n"

TOTAL_STEPS=10

# --- Step 1: Required files ------------------------------------------------
log "${BOLD}Step 1/${TOTAL_STEPS}: Required files present${NC}"
REQUIRED_FILES=("README.md" "Dockerfile" "openenv.yaml" "main.py" "requirements.txt")
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/$f" ]; then
    fail "missing: $f"; MISSING=1
  fi
done
if [ $MISSING -eq 0 ]; then
  pass "all required files present"
else
  stop "Step 1"
fi

# --- Step 2: openenv.yaml parses ------------------------------------------
log "${BOLD}Step 2/${TOTAL_STEPS}: openenv.yaml parseable${NC}"
if [ -z "$PY" ]; then
  fail "no Python interpreter found (looked for python3, python, py)"
  stop "Step 2"
fi
# cd into REPO_DIR so Python sees a plain relative path -- avoids
# MINGW64's /c/Users/... path being passed to a native Windows Python.exe.
# Capture stderr instead of suppressing it, so real errors surface.
if YAML_ERR=$(cd "$REPO_DIR" && $PY -c "import yaml; yaml.safe_load(open('openenv.yaml'))" 2>&1); then
  pass "openenv.yaml parses cleanly"
else
  fail "openenv.yaml check failed:"
  printf "%s\n" "$YAML_ERR"
  case "$YAML_ERR" in
    *"No module named 'yaml'"*|*"No module named yaml"*)
      printf "  ${YELLOW}Hint:${NC} pip install pyyaml\n"
      ;;
    *ScannerError*|*ParserError*)
      printf "  ${YELLOW}Hint:${NC} actual YAML syntax error -- see line/col above\n"
      ;;
  esac
  stop "Step 2"
fi

# --- Step 3: Training plot images committed -------------------------------
# Live QnA: "Training evidence committed to the repo as image files
# (.png / .jpg): At minimum a loss curve and a reward curve."
log "${BOLD}Step 3/${TOTAL_STEPS}: Training plot images${NC}"
LOSS_HIT=$(find "$REPO_DIR" -type f \( -iname "*loss*.png" -o -iname "*loss*.jpg" -o -iname "*loss*.jpeg" \) 2>/dev/null | head -1)
REW_HIT=$(find "$REPO_DIR" -type f \( -iname "*reward*.png" -o -iname "*reward*.jpg" -o -iname "*reward*.jpeg" \) 2>/dev/null | head -1)
if [ -n "$LOSS_HIT" ]; then pass "loss curve image: ${LOSS_HIT#$REPO_DIR/}"; else fail "no loss curve image found"; stop "Step 3"; fi
if [ -n "$REW_HIT" ];  then pass "reward curve image: ${REW_HIT#$REPO_DIR/}"; else fail "no reward curve image found"; stop "Step 3"; fi

# --- Step 4: README links every deliverable -------------------------------
# Live QnA: "A README that links every deliverable: HF Space, training
# notebook, and your writeup (blog / video / slides)..."
log "${BOLD}Step 4/${TOTAL_STEPS}: README links${NC}"
README="$REPO_DIR/README.md"
README_OK=1
grep -qi "huggingface.co/spaces" "$README" || { fail "README missing HF Space link"; README_OK=0; }
grep -qiE "(colab\.research\.google\.com|\.ipynb)" "$README" || { fail "README missing training notebook (Colab or .ipynb)"; README_OK=0; }
if grep -qiE "(huggingface\.co/blog|youtu|youtube\.com|/blog/|slides)" "$README"; then
  :
else
  fail "README missing writeup link (HF blog / YouTube / slides)"; README_OK=0
fi
if [ $README_OK -eq 1 ]; then
  pass "README references HF Space + notebook + writeup"
else
  stop "Step 4"
fi

# --- Step 5: HF Space publicly reachable ----------------------------------
log "${BOLD}Step 5/${TOTAL_STEPS}: HF Space publicly reachable${NC}"
ROOT_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL/" --max-time 30 || echo "000")
if [ "$ROOT_CODE" = "200" ] || [ "$ROOT_CODE" = "404" ]; then
  pass "Space responds (HTTP $ROOT_CODE)"
else
  fail "Space root unreachable (HTTP $ROOT_CODE) -- is it private or sleeping?"; stop "Step 5"
fi

# --- Step 6: /reset for all 4 tasks ---------------------------------------
log "${BOLD}Step 6/${TOTAL_STEPS}: /reset for all tasks${NC}"
ALL_RESET_OK=1
for task in "${TASKS[@]}"; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "{\"task_name\":\"$task\"}" \
    "$PING_URL/reset" --max-time 30 || echo "000")
  if [ "$CODE" = "200" ]; then
    pass "/reset $task -> 200"
  else
    fail "/reset $task -> HTTP $CODE"; ALL_RESET_OK=0
  fi
done
[ $ALL_RESET_OK -eq 1 ] || stop "Step 6"

# --- Step 7: /tasks lists all 4 -------------------------------------------
log "${BOLD}Step 7/${TOTAL_STEPS}: /tasks endpoint${NC}"
TASKS_BODY=$(curl -s "$PING_URL/tasks" --max-time 30 || echo "")
TASKS_OK=1
for task in "${TASKS[@]}"; do
  if echo "$TASKS_BODY" | grep -q "\"$task\""; then
    :
  else
    fail "/tasks missing: $task"; TASKS_OK=0
  fi
done
if [ $TASKS_OK -eq 1 ]; then
  pass "/tasks lists easy, medium, hard, crisis"
else
  stop "Step 7"
fi

# --- Step 8: /state after reset --------------------------------------------
log "${BOLD}Step 8/${TOTAL_STEPS}: /state after reset${NC}"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"task_name":"easy"}' "$PING_URL/reset" --max-time 30 > /dev/null
STATE_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL/state" --max-time 30 || echo "000")
if [ "$STATE_CODE" = "200" ]; then
  pass "/state -> 200"
else
  fail "/state -> HTTP $STATE_CODE"; stop "Step 8"
fi

# --- Step 9: Docker build --------------------------------------------------
log "${BOLD}Step 9/${TOTAL_STEPS}: Docker build${NC}"
if ! command -v docker &>/dev/null; then
  fail "docker not installed on this machine"; stop "Step 9"
fi
if ! timeout "$DOCKER_BUILD_TIMEOUT" docker build -q "$REPO_DIR" > /dev/null 2>&1; then
  fail "docker build failed -- run \`docker build $REPO_DIR\` manually to see the error"
  stop "Step 9"
fi
pass "docker build"

# --- Step 10: openenv validate --------------------------------------------
log "${BOLD}Step 10/${TOTAL_STEPS}: openenv validate${NC}"
if ! command -v openenv &>/dev/null; then
  fail "install: pip install openenv-core"; stop "Step 10"
fi
if ( cd "$REPO_DIR" && openenv validate ); then
  pass "openenv validate"
else
  fail "openenv validate"; stop "Step 10"
fi

printf "\n${GREEN}${BOLD}All ${TOTAL_STEPS}/${TOTAL_STEPS} checks passed. Ready to submit.${NC}\n\n"