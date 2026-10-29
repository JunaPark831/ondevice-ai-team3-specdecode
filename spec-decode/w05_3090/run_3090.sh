#!/usr/bin/env bash
# W05 · vast.ai RTX 3090 에서 준하 님 W04(M5 Pro) 측정을 같은 코드·같은 문장·같은 설정으로 다시 잰다.
#
# 사용 (vast.ai 인스턴스 안, tmux 안에서):
#   bash run_3090.sh
# 이 파일과 같은 폴더에 lang_cuda.patch, gpu_env.py, summarize_3090.py 가 있어야 한다.
# 중간에 끊겨도 다시 실행하면 이어서 돈다 (모델 다운로드·online 측정 모두 이어 받기).
#
# 바꾸는 것: 하드웨어 하나 (Apple M5 Pro MPS → RTX 3090 CUDA)
# 그대로 둔 것: 모델(Qwen3-8B / 0.6B / 1.7B, bf16), FLORES 같은 12문장, greedy, batch 1,
#              γ(en↔ko 2·4, en→de 3·6), max_new 256, torch 2.14.1, transformers 5.18.0
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-/workspace}
REPO_URL=https://github.com/JunaPark831/ondevice-ai-team3-specdecode.git
BRANCH=w04-mps-measurements
T0=$(date +%s)
step() { echo; echo "=================== [$1] $2  ($(date '+%H:%M:%S'), 시작 후 $(( ($(date +%s) - T0) / 60 ))분)"; }

for f in lang_cuda.patch gpu_env.py summarize_3090.py; do
  [ -f "$HERE/$f" ] || { echo "❌ $HERE/$f 가 없습니다. 네 파일을 같은 폴더에 올렸는지 확인하세요."; exit 1; }
done

step 1 "GPU 확인"
nvidia-smi || { echo "❌ GPU가 안 보입니다. 다른 매물로 다시 빌리세요."; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ]; then
  echo "⚠️  드라이버 $DRV — torch 2.14.1 은 CUDA 13.0 을 쓰므로 드라이버 580 이상이 필요할 가능성이 큽니다."
  echo "    3단계에서 'CUDA 안 됨'으로 멈추면, 'Max CUDA 13.0 이상' 필터로 다른 매물을 빌리세요."
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "⚠️  GPU 에서 다른 프로세스가 돌고 있습니다. 측정이 오염될 수 있습니다:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
fi

step 2 "팀 저장소 받기 + lang.py GPU 수정 적용"
mkdir -p "$WORK" && cd "$WORK"
[ -d ondevice-ai-team3-specdecode ] || git clone -b "$BRANCH" "$REPO_URL"
cd ondevice-ai-team3-specdecode
git log -1 --format='기준 커밋: %h %ad %s' --date=short
if grep -q "^def device" spec-decode/lang.py; then
  echo "수정본 이미 적용됨"
else
  git apply "$HERE/lang_cuda.patch" && echo "lang.py 수정 적용 완료"
fi
cd spec-decode

step 3 "파이썬 환경 (팀과 같은 버전)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || pip install -q uv 2>/dev/null || pip install -q --break-system-packages uv 2>/dev/null \
  || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x .venv/bin/python ] || uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -q -r requirements.txt huggingface_hub
PY=.venv/bin/python
$PY - <<'EOF'
import torch, transformers
assert torch.cuda.is_available(), "❌ CUDA 안 됨 — 드라이버가 낮을 가능성. Max CUDA 13.0 이상 매물로 다시 빌리세요"
print("torch", torch.__version__, "CUDA", torch.version.cuda, "| transformers", transformers.__version__, "|", torch.cuda.get_device_name(0))
assert transformers.__version__ == "5.18.0", "transformers 버전이 팀과 다릅니다 (4.x 는 KV 캐시 crop 이 달라 속도가 왜곡됨)"
EOF

step 4 "모델 받기 (약 21GB) + 리비전 기록"
for m in Qwen3-8B Qwen3-0.6B Qwen3-1.7B; do
  .venv/bin/hf download "Qwen/$m" --local-dir "models/$m"
done
OUT=results/lang_3090
mkdir -p "$OUT"
$PY - "$OUT" <<'EOF'
import json, sys
from huggingface_hub import model_info
revs = {m: model_info(f"Qwen/{m}").sha for m in ("Qwen3-8B", "Qwen3-0.6B", "Qwen3-1.7B")}
open(f"{sys.argv[1]}/model_revisions.json", "w").write(json.dumps(revs, indent=2))
print(revs)
EOF

step 5 "기기 스펙 카드 + 실효 대역폭"
cp results/lang/gen.jsonl "$OUT/"   # costs/online 이 프롬프트를 여기서 읽는다 (M5 결과 폴더는 건드리지 않는다)
$PY "$HERE/gpu_env.py" "$OUT"

step 6 "sd.py 자기검증"
$PY sd.py

step 7 "독립 비용 c, r(k) 측정 (W04 costs_clean 과 같은 문맥)"
if [ -f "$OUT/costs_clean.json" ]; then
  echo "이미 있음 → 건너뜀 (다시 재려면 $OUT/costs_clean.json 을 지우세요)"
else
  SD_DEVICE=cuda $PY lang.py costs --out "$OUT" --drafts models/Qwen3-0.6B models/Qwen3-1.7B 2>&1 | tee "$OUT/costs.log"
fi

step 8 "실제 SD (W04 와 같은 12문장·같은 γ, draft 하나씩 단독)"
run() {  # 끊겼다 다시 돌리면 lang.py 가 끝난 블록을 건너뛴다
  local tag=$1; shift
  echo "--- $tag"
  SD_DEVICE=cuda $PY lang.py online --out "$OUT" --n-online 12 --tag "$tag" "$@" 2>&1 | tee -a "$OUT/online_$tag.log"
}
run 3090_iso06_cjk --drafts models/Qwen3-0.6B --dirs en-ko ko-en --cfg Qwen3-0.6B:2:0 Qwen3-0.6B:4:0
run 3090_iso06_de  --drafts models/Qwen3-0.6B --dirs en-de       --cfg Qwen3-0.6B:3:0 Qwen3-0.6B:6:0
run 3090_iso17_cjk --drafts models/Qwen3-1.7B --dirs en-ko ko-en --cfg Qwen3-1.7B:2:0 Qwen3-1.7B:4:0
run 3090_iso17_de  --drafts models/Qwen3-1.7B --dirs en-de       --cfg Qwen3-1.7B:3:0 Qwen3-1.7B:6:0
nvidia-smi -q -d PERFORMANCE,TEMPERATURE,CLOCK > "$OUT/gpu_state_after.txt" || true   # 발열로 클럭이 떨어졌는지

step 9 "M5 Pro vs 3090 요약표"
$PY "$HERE/summarize_3090.py" results/lang "$OUT" | tee "$OUT/summary_3090.md"

if [ "${CLOCK_CHECK:-0}" = 1 ]; then
  step 10 "(선택) GPU 클럭 고정이 되는지 — compute sweep 가능 여부. 끝나면 반드시 원복"
  trap 'nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT
  nvidia-smi -q -d SUPPORTED_CLOCKS | head -20 > "$OUT/clock_check.txt" 2>&1 || true
  if nvidia-smi -lgc 1000,1000 >> "$OUT/clock_check.txt" 2>&1; then echo "✅ 클럭 고정 가능" | tee -a "$OUT/clock_check.txt"
  else echo "❌ 클럭 고정 불가 (권한 없음)" | tee -a "$OUT/clock_check.txt"; fi
  nvidia-smi -rgc >> "$OUT/clock_check.txt" 2>&1 || true
fi

step 끝 "결과 묶기"
TAR="$HOME/result_3090_$(date +%m%d_%H%M).tar.gz"
tar czf "$TAR" --exclude gen.jsonl -C results lang_3090
echo "결과 파일: $TAR"
echo "내 컴퓨터 터미널에서 받기:  scp -P <포트> root@<주소>:$TAR ."
echo "받은 뒤 vast.ai 콘솔에서 Stop 이 아니라 Delete 를 눌러야 디스크 요금이 멈춥니다."
echo "총 $(( ($(date +%s) - T0) / 60 ))분"
