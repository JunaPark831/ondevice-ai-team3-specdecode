# spec-decode

Speculative decoding (Leviathan et al. 2023) 직접 구현 + 노트북(MPS) 속도 측정.

| 파일 | 내용 |
|---|---|
| `sd.py` | 알고리즘. greedy 검증 / rejection sampling, KV 캐시 롤백. `python sd.py` 로 자기검증 |
| `bench.py` | draft × γ × 태스크 sweep, vanilla 와 비교 → `results/<이름>/runs.csv`, `costs.json`, `env.json` |
| `plot.py` | 결과 → `summary.md`, `summary.png` (speedup, α, Leviathan 식 예측과 비교) |
| `models.txt` | 모델 다운로드 목록 (`aria2c -i models.txt -x 16 -s 16 -c`) |

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
aria2c -i models.txt -x 16 -s 16 -j 8 -c          # Qwen3-0.6B / 1.7B / 8B → models/
.venv/bin/python sd.py
.venv/bin/python bench.py --out results/main       # 기본: target 8B, draft 0.6B·1.7B, γ 1~12, 반복 2
.venv/bin/python bench.py --first-rep 1 --reps 1 --gammas 1 2 3 4 5 6 8 10 12 16 20 24 --out results/main_rep1
.venv/bin/python plot.py results/main results/main_rep1   # 두 회차 합침 → results/combined/
```

측정 중에는 같은 컴퓨터에서 다른 무거운 작업을 돌리지 않는다. `bench.py` 가 블록마다 vanilla 를 앞뒤로 재서 10% 넘게 흔들린 블록은 `plot.py` 가 자동으로 뺀다.

결과 해석은 [`docs/W02-research-log.html`](../docs/W02-research-log.html) (브라우저로 열기).

`results/main/runs_rep1_contaminated.csv` 는 측정 중 다른 프로세스 부하로 오염된 2회차 원본이다. 분석에서는 빼고 `results/main_rep1` 로 다시 쟀다.
