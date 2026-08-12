#!/bin/bash
set -uo pipefail

# === CONFIGURATION ===
MODEL_DIR="${HOME}/llm"
LLM_SERVER="${HOME}/llama/llama-b9058/llama-server"
SERVER_PORT=8765
RUNS=10
SERVER_PID=""
CURL_TIMEOUT=300
NGl=0  # Offload layers to GPU

MODELS=(
  "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
  # "gemma-4-E4B-it-IQ4_NL.gguf"
)
CACHE_TYPES=("f16")
THREADS=(4)
BATCH_SIZES=("512:256")  # Format: "logical_batch:physical_ubatch"
FLASH_ATTN=(true false)   # true = -fa enabled, false = disabled

# === PROMPT GENERATION ===
generate_prompt() {
  local target_tokens=$1
  local base="Explain recursion in programming with detailed examples, "
  local result="$base"
  local target_chars=$((target_tokens * 4))
  while [[ ${#result} -lt $target_chars ]]; do
    result+="Recursion is a fundamental concept where a function calls itself. "
    result+="It requires a base case to terminate and a recursive case that progresses toward it. "
    result+="Common examples include factorial calculation, tree traversal, and divide-and-conquer algorithms. "
    result+="Potential issues include stack overflow if the base case is missing or unreachable. "
  done
  echo "${result:0:$target_chars}"
}

PROMPT_SHORT=$(generate_prompt 500)
PROMPT_MED=$(generate_prompt 1000)
PROMPT_LONG=$(generate_prompt 2000)
PROMPTS=(
  "short:500:${PROMPT_SHORT}"
  "medium:1000:${PROMPT_MED}"
  "long:2000:${PROMPT_LONG}"
)

# === VALIDATION ===
if [[ ! -x "$LLM_SERVER" ]]; then
  echo "Error: llama-server not found at $LLM_SERVER"
  exit 1
fi

# === FILENAME GENERATION LOGIC ===
format_model_names() {
  local arr=("$@")
  [[ ${#arr[@]} -eq 0 ]] && echo "unknown" && return
  if [[ ${#arr[@]} -eq 1 ]]; then
    echo "${arr[0]%.gguf}"
    return
  fi
  local common="${arr[0]}"
  for m in "${arr[@]}"; do
    while [[ ${#common} -gt 0 && "${m:0:${#common}}" != "$common" ]]; do
      common="${common%?}"
    done
  done
  local diffs=()
  for m in "${arr[@]}"; do
    diffs+=("${m#$common}")
  done
  local joined=$(IFS=+; echo "${diffs[*]}")
  joined="${joined%.gguf}"
  common="${common%.gguf}"
  common="${common%-}"
  echo "${common}-${joined}"
}

MODEL_TAG=$(format_model_names "${MODELS[@]}")
CACHE_TAG=$(IFS=+; echo "${CACHE_TYPES[*]}")
THREAD_TAG=$(IFS=-; echo "${THREADS[*]}")
BATCH_TAG="${BATCH_SIZES[0]//:/ub}"

FA_TAG=""
for fa_val in "${FLASH_ATTN[@]}"; do
  [[ -n "$FA_TAG" ]] && FA_TAG+="-"
  [[ "$fa_val" == "true" ]] && FA_TAG+="fa" || FA_TAG+="nofa"
done

BASE_RESULTS="bench_${MODEL_TAG}_${CACHE_TAG}_t${THREAD_TAG}_ngl${NGl}_${BATCH_TAG}_${FA_TAG}.txt"

# === RESUME / NEW RUN LOGIC ===
write_header() {
  cat > "$RESULTS" <<EOF
Gemma Manual Benchmark (Low Memory Mode, Reasoning OFF)
==========================================================
Date: $(date)
Config: Models=[${MODEL_TAG}] | KV=[${CACHE_TAG}] | Threads=[${THREAD_TAG}] | NGL=[${NGl}] | Batch=[${BATCH_TAG}] | FlashAttn=[${FA_TAG}]
CURL_TIMEOUT: ${CURL_TIMEOUT}s
EOF
}

if [[ -f "$BASE_RESULTS" ]]; then
  if grep -qF "# BENCHMARK_COMPLETE=true" "$BASE_RESULTS"; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVED="${BASE_RESULTS%.txt}_${TIMESTAMP}.txt"
    mv "$BASE_RESULTS" "$ARCHIVED"
    echo "[*] Previous completed benchmark archived as $ARCHIVED. Starting fresh run."
    RESULTS="$BASE_RESULTS"
    write_header
  else
    echo "[~] Resuming existing benchmark from $BASE_RESULTS"
    RESULTS="$BASE_RESULTS"
    echo -e "\n[~] Resuming benchmark at $(date)" | tee -a "$RESULTS"
  fi
else
  echo "[*] Starting new benchmark. Results will be saved to $BASE_RESULTS"
  RESULTS="$BASE_RESULTS"
  write_header
fi

# === CHECKPOINT FUNCTIONS ===
checkpoint_key() {
  local model=$1 ctk=$2 batch=$3 ubatch=$4 prompt=$5 threads=$6 fa=$7
  local fa_label
  [[ "$fa" == "true" ]] && fa_label="fa" || fa_label="nofa"
  echo "CHECKPOINT:${model}|${ctk}|b${batch}ub${ubatch}|t${threads}|${fa_label}|${prompt}"
}

is_done() {
  local key
  key=$(checkpoint_key "$@")
  [[ -f "$RESULTS" ]] && grep -qF "# ${key}=DONE" "$RESULTS" 2>/dev/null
}

mark_done() {
  local key
  key=$(checkpoint_key "$@")
  echo "# ${key}=DONE" >> "$RESULTS"
}

# === SERVER & MONITORING ===
monitor_ram() {
  local pid=$1 mem_file=$2
  > "$mem_file"
  while kill -0 "$pid" 2>/dev/null; do
    local mem
    mem=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo "0")
    echo "$mem" >> "$mem_file"
    sleep 1
  done
}

start_server() {
  local model=$1 ctk=$2 batch=$3 ubatch=$4 threads=$5 use_fa=$6
  local log_file="/tmp/llama_server_$$.log"

  local fa_args=()
  [[ "$use_fa" == "true" ]] && fa_args=("--flash-attn" "on") || fa_args=("--flash-attn" "off")

  $LLM_SERVER \
    -m "${MODEL_DIR}/${model}" \
    -t "$threads" \
    -ngl $NGl \
    --no-mmap \
    --reasoning off \
    -b "$batch" \
    -ub "$ubatch" \
    -ctk "$ctk" \
    -ctv "$ctk" \
    "${fa_args[@]}" \
    --port $SERVER_PORT \
    > "$log_file" 2>&1 &
  SERVER_PID=$!

  local attempts=0
  while [[ $attempts -lt 60 ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      wait "$SERVER_PID" 2>/dev/null
      local exit_code=$?
      echo "  |  |  |  |   [!] Server crashed during startup (exit: $exit_code)" | tee -a "$RESULTS"
      echo "  |  |  |  |[DEBUG] Log:" | tee -a "$RESULTS"
      sed 's/^/  |  |  |  |       /' "$log_file" | tee -a "$RESULTS"
      return 1
    fi
    local health
    health=$(curl -sf "http://127.0.0.1:${SERVER_PORT}/health" 2>/dev/null || true)
    if echo "$health" | grep -q '"status":"ok"'; then
      rm -f "$log_file"
      return 0
    fi
    sleep 1
    ((attempts++))
  done
  echo "  |  |  |  |   [!] Server failed to start (timeout 60s)" | tee -a "$RESULTS"
  echo "  |  |  |  |[DEBUG] Log:" | tee -a "$RESULTS"
  sed 's/^/  |  |  |  |       /' "$log_file" | tee -a "$RESULTS"
  stop_server
  return 1
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

# === QUERY & ANALYSIS ===
run_query_with_metrics() {
  local prompt="$1"
  local temp_file="/tmp/llama_resp_$$"
  local time_file="/tmp/llama_time_$$"
  {
    time curl -sf \
      --max-time "$CURL_TIMEOUT" \
      --connect-timeout 10 \
      "http://127.0.0.1:${SERVER_PORT}/completion" \
      -H "Content-Type: application/json" \
      -d "{
        \"prompt\": $(printf '%s' "$prompt" | jq -Rs .),
        \"n_predict\": 64,
        \"temperature\": 1,
        \"cache_prompt\": false,
        \"stream\": false
      }" \
      -o "$temp_file" \
      --write-out "time_total:%{time_total}
time_namelookup:%{time_namelookup}
time_connect:%{time_connect}
time_appconnect:%{time_appconnect}
time_pretransfer:%{time_pretransfer}
time_starttransfer:%{time_starttransfer}
" \
      2>/dev/null
  } > "$time_file" 2>&1
  local curl_exit=$?
  local ttft
  ttft=$(grep "time_starttransfer:" "$time_file" | cut -d: -f2 | tr -d ' ')
  local total_time
  total_time=$(grep "time_total:" "$time_file" | cut -d: -f2 | tr -d ' ')
  local response=""
  if [[ -f "$temp_file" ]]; then response=$(cat "$temp_file"); fi
  rm -f "$temp_file" "$time_file"
  echo "${curl_exit}|${ttft:-0}|${total_time:-0}|${response}"
}

analyze_response() {
  local curl_exit=$1 response=$2
  if [[ $curl_exit -ne 0 ]]; then
    [[ $curl_exit -eq 28 ]] && echo "ERROR:TIMEOUT" && return
    echo "ERROR:CURL_${curl_exit}" && return
  fi
  [[ -z "$response" ]] && { echo "ERROR:EMPTY_RESPONSE"; return; }
  echo "$response" | jq empty 2>/dev/null || { echo "ERROR:INVALID_JSON"; return; }
  if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
    echo "ERROR:SERVER:$(echo "$response" | jq -r '.error')"
    return
  fi
  local gen_speed
  gen_speed=$(echo "$response" | jq -r '.timings.predicted_per_second // empty')
  [[ -z "$gen_speed" || "$gen_speed" == "null" ]] && { echo "ERROR:NO_GENERATION_METRICS"; return; }
  echo "OK"
}

# === MAIN BENCHMARK LOOP ===
MEM_FILE="/tmp/llama_mem_$$.txt"

for model in "${MODELS[@]}"; do
  echo -e "\n[*] Model: $model" | tee -a "$RESULTS"
  for ctk in "${CACHE_TYPES[@]}"; do
    echo "  +-- KV: $ctk" | tee -a "$RESULTS"
    for batch_entry in "${BATCH_SIZES[@]}"; do
      IFS=':' read -r b_batch b_ubatch <<< "$batch_entry"
      echo "  |  +-- Batch: -b ${b_batch} -ub ${b_ubatch}" | tee -a "$RESULTS"

      for t in "${THREADS[@]}"; do
        echo "  |  |  +-- Threads: $t" | tee -a "$RESULTS"

        for fa in "${FLASH_ATTN[@]}"; do
          [[ "$fa" == "true" ]] && fa_label="enabled" || fa_label="disabled"
          echo "  |  |  |  +-- FlashAttn: $fa_label" | tee -a "$RESULTS"
          echo "  |  |  |  |   [~] Starting server..." | tee -a "$RESULTS"

          if ! start_server "$model" "$ctk" "$b_batch" "$b_ubatch" "$t" "$fa"; then
            echo "  |  |  |  +--" | tee -a "$RESULTS"
            continue
          fi

          echo "  |  |  |  |   [~] Server ready (PID: $SERVER_PID)" | tee -a "$RESULTS"
          monitor_ram $SERVER_PID "$MEM_FILE" &
          MON_PID=$!

          echo "  |  |  |  |   [~] Warmup..." | tee -a "$RESULTS"
          run_query_with_metrics "$PROMPT_SHORT" > /dev/null || true
          sleep 10

          for prompt_entry in "${PROMPTS[@]}"; do
            IFS=':' read -r p_name p_tokens p_text <<< "$prompt_entry"

            if is_done "$model" "$ctk" "$b_batch" "$b_ubatch" "$p_name" "$t" "$fa"; then
              echo "  |  |  |  |   [~] Prompt: $p_name (~$p_tokens tokens) [SKIP - done]" | tee -a "$RESULTS"
              echo "  |  |  |  |   +--" | tee -a "$RESULTS"
              continue
            fi

            echo "  |  |  |  |   [~] Prompt: $p_name (~$p_tokens tokens)" | tee -a "$RESULTS"
            PR_SUM=0; GEN_SUM=0; TTFT_SUM=0; MEM_SUM=0; MEM_PEAK=0; FAIL_COUNT=0
            declare -A ERROR_COUNTS=()

            for run in $(seq 1 $RUNS); do
              echo "  |  |  |  |   [>] Run $run/$RUNS [$p_name]... ($(date +%H:%M:%S))" | tee -a "$RESULTS"
              RESULT=$(run_query_with_metrics "$p_text")
              IFS='|' read -r curl_exit ttft total_time response <<< "$RESULT"
              status=$(analyze_response "$curl_exit" "$response")

              if [[ "$status" != "OK" ]]; then
                echo "  |  |  |  |     [!] $status" | tee -a "$RESULTS"
                ERROR_COUNTS["$status"]=$((${ERROR_COUNTS["$status"]:-0} + 1))
                FAIL_COUNT=$((FAIL_COUNT + 1))
                sleep 5
                continue
              fi

              PR_SPEED=$(echo "$response" | jq -r '.timings.prompt_per_second // 0')
              GEN_SPEED=$(echo "$response" | jq -r '.timings.predicted_per_second // 0')
              RUN_MEM_AVG=$(awk '$1>0{sum+=$1;count++}END{if(count>0)printf "%.0f",sum/count/1024;else print "0"}' "$MEM_FILE")
              RUN_MEM_PEAK=$(awk 'BEGIN{max=0}$1>max{max=$1}END{printf "%.0f",max/1024}' "$MEM_FILE")

              echo "  |  |  |  |     Prompt: ${PR_SPEED} t/s | Gen: ${GEN_SPEED} t/s | TTFT: ${ttft}s | RAM: ${RUN_MEM_AVG}/${RUN_MEM_PEAK} MiB" | tee -a "$RESULTS"

              PR_SUM=$(awk "BEGIN {print $PR_SUM + $PR_SPEED}")
              GEN_SUM=$(awk "BEGIN {print $GEN_SUM + $GEN_SPEED}")
              TTFT_SUM=$(awk "BEGIN {print $TTFT_SUM + $ttft}")
              MEM_SUM=$(awk "BEGIN {print $MEM_SUM + $RUN_MEM_AVG}")
              MEM_PEAK=$(awk "BEGIN {print ($RUN_MEM_PEAK > $MEM_PEAK) ? $RUN_MEM_PEAK : $MEM_PEAK}")
              sleep 5
            done

            SUCCESS=$((RUNS - FAIL_COUNT))
            echo "  |  |  |  |   [=] $p_name results ($SUCCESS/$RUNS ok):" | tee -a "$RESULTS"
            if [[ $SUCCESS -gt 0 ]]; then
              PR_AVG=$(awk "BEGIN {printf \"%.2f\", $PR_SUM / $SUCCESS}")
              GEN_AVG=$(awk "BEGIN {printf \"%.2f\", $GEN_SUM / $SUCCESS}")
              TTFT_AVG=$(awk "BEGIN {printf \"%.3f\", $TTFT_SUM / $SUCCESS}")
              MEM_AVG=$(awk "BEGIN {printf \"%.0f\", $MEM_SUM / $SUCCESS}")
              echo "  |  |  |  |       Prompt: ${PR_AVG} t/s | Gen: ${GEN_AVG} t/s | TTFT: ${TTFT_AVG}s | RAM: ${MEM_AVG}/${MEM_PEAK} MiB" | tee -a "$RESULTS"
            else
              echo "  |  |  |  |       All runs failed" | tee -a "$RESULTS"
            fi

            if [[ $FAIL_COUNT -gt 0 ]]; then
              echo "  |  |  |  |       Errors:" | tee -a "$RESULTS"
              for err in "${!ERROR_COUNTS[@]}"; do
                echo "  |  |  |  |         - $err: ${ERROR_COUNTS[$err]}" | tee -a "$RESULTS"
              done
            fi

            unset ERROR_COUNTS
            echo "  |  |  |  |   +--" | tee -a "$RESULTS"
            mark_done "$model" "$ctk" "$b_batch" "$b_ubatch" "$p_name" "$t" "$fa"
            sleep 5
          done

          kill $MON_PID 2>/dev/null || true
          wait $MON_PID 2>/dev/null || true
          echo "  |  |  |  |   [~] Stopping server..." | tee -a "$RESULTS"
          stop_server
          echo "  |  |  |  +--" | tee -a "$RESULTS"
          sleep 5
        done

        echo "  |  |  +--" | tee -a "$RESULTS"
        sleep 10
      done

      echo "  |  +--" | tee -a "$RESULTS"
    done
    echo "  +--" | tee -a "$RESULTS"
  done
done

# === COMPLETION MARKER & CLEANUP ===
echo "# BENCHMARK_COMPLETE=true" >> "$RESULTS"
rm -f "$MEM_FILE" /tmp/llama_server_$$.log /tmp/llama_resp_* /tmp/llama_time_*
echo -e "\n[ok] Benchmark complete. Results saved to $RESULTS"
echo "[ok] Total elapsed: $((SECONDS/3600))h $(( (SECONDS%3600)/60 ))m ${SECONDS%60}s" | tee -a "$RESULTS"
cat "$RESULTS"