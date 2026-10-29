#!/usr/bin/env bash
# W05 추가 측정 (약 5~10분): ① ctx_en 비용 재측정(앞뒤 차이 11.3% > 10%) ② CPU 병목 진단
#   bash /workspace/w05-3090/diag_3090.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd /workspace/ondevice-ai-team3-specdecode/spec-decode
OUT=results/lang_3090
PY=.venv/bin/python

echo "=== [A] 비용 재측정 (1차 결과는 costs_clean_run1.json 으로 보존)"
[ -f "$OUT/costs_clean_run1.json" ] || mv "$OUT/costs_clean.json" "$OUT/costs_clean_run1.json"
SD_DEVICE=cuda $PY lang.py costs --out "$OUT" --drafts models/Qwen3-0.6B models/Qwen3-1.7B 2>&1 | grep -v "Loading weights" | tee "$OUT/costs_run2.log"

echo "=== [B] CPU 병목 진단"
SD_DEVICE=cuda $PY "$HERE/diag_cpu.py" 2>&1 | grep -v "Loading weights" | tee "$OUT/diag_cpu.log"

echo "=== [C] 요약 다시 만들기 + 묶기"
$PY "$HERE/summarize_3090.py" results/lang "$OUT" > "$OUT/summary_3090.md"
lscpu > "$OUT/lscpu.txt"; cat /sys/fs/cgroup/cpu.max > "$OUT/cgroup_cpu_max.txt" 2>/dev/null || true
TAR="$HOME/result_3090_diag_$(date +%m%d_%H%M).tar.gz"
tar czf "$TAR" --exclude gen.jsonl -C results lang_3090
echo "결과 파일: $TAR"
echo "내 컴퓨터에서:  scp -P 21195 \"root@70.30.252.209:$TAR\" ."
