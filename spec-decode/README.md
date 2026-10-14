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

결과 해석은 [`docs/W02-research-log.md`](../docs/W02-research-log.md).

`results/main/runs_rep1_contaminated.csv` 는 측정 중 다른 프로세스 부하로 오염된 2회차 원본이다. 분석에서는 빼고 `results/main_rep1` 로 다시 쟀다.

## EXP-003 · 언어 행렬 (W03)

FLORES-200 같은 200문장을 en·ko·ja·zh 12방향 + en↔de 로 번역시키고, target 출력 위에서 draft 를 teacher forcing 해 토큰마다 "draft 1등 == target 토큰" 을 잰다. greedy SD 의 수락은 이것으로 전부 정해지므로 모든 γ 의 라운드를 그대로 재현할 수 있다 (`lang_plot.py` 의 `simulate`, online 실측과 라운드 단위로 대조).

| 파일 | 내용 |
|---|---|
| `lang.py` | `prep` 데이터 · `gen` target 번역(배치) · `score` teacher forcing 지표 · `online` 실제 SD 시간 측정 · `replay` 실측 출력 재채점 · `costs` forward 비용 |
| `lang_plot.py` | `results/lang/` → `summary.md`, `summary.png`, `fig_h1.png` (기사 단위 군집 부트스트랩, 사전 예측 판정) |
| `coverage.py` | 다른 계열 draft 후보(EXAONE·Kanana·HyperCLOVAX·Llama-3·Gemma-3)의 어휘 커버리지, 토크나이저 파일만 → `results/lang/coverage.md` |
| `data/flores.jsonl` | FLORES-200 devtest 200문장 (seed 0, CC BY-SA 4.0) · `data/wmt20.jsonl` WMT20 ja/zh 원문 각 100문장 (대조 실험) |

```bash
.venv/bin/python -I lang.py prep <flores200_dataset 폴더>           # FLORES-200: dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
.venv/bin/python lang.py gen --bs 64                                 # 2,800 번역, M5 Pro 에서 약 15분
.venv/bin/python lang.py score --bs 16 --drafts models/Qwen3-0.6B models/Qwen3-1.7B
.venv/bin/python lang.py online --drafts models/Qwen3-0.6B --n-online 12   # 실측 SD (다른 무거운 작업 없이)
.venv/bin/python lang.py replay && .venv/bin/python lang.py costs --drafts models/Qwen3-0.6B models/Qwen3-1.7B
.venv/bin/python lang_plot.py
```

`results/lang/contaminated_online_swap.csv` 는 다른 작업의 메모리 압박(swap)으로 vanilla 가 15 → 2 tok/s 까지 떨어진 첫 실측 원본이다. 분석에서 빼고 target + 0.6B 만 올려 다시 쟀다 (`online.csv`, 블록마다 swap-in 기록).

`sd.py` 는 2026-10-14 에 수락·거절 집계만 고쳤다 (EOS 뒤 토큰까지 세던 것). 생성과 시간 측정은 그대로라 W02 의 실측 speedup 은 유효하다. 영향을 받는 것은 EOS 로 끝난 번역 태스크(en2ko·ko2en)의 α 와 τ = (A+라운드)/라운드 로 계산한 비용모델 예측이다 (en_write·code 는 128토큰에서 잘려 EOS 가 없었다).
