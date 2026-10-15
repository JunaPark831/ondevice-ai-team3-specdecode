# spec-decode

Speculative decoding (Leviathan et al. 2023) 직접 구현 + 노트북(MPS) 속도 측정.

| 파일 | 내용 |
|---|---|
| `sd.py` | 알고리즘. greedy 검증 / rejection sampling, KV 캐시 롤백. `python sd.py` 로 자기검증 |
| `bench.py` | draft × γ × 태스크 sweep, vanilla 와 비교 → `results/<이름>/runs.csv`, `costs.json`, `env.json` |
| `plot.py` | 결과 → `summary.md`, `summary.png` (speedup, α, Leviathan 식 예측과 비교) |
| `models.txt` | 모델 다운로드 목록 (`aria2c -i models.txt -x 16 -s 16 -c`) |
| `e0_fp32.py` | E0 정합성: greedy SD 출력 == vanilla 인지 dtype 별로 확인 (fp32 에서 100% 면 W02 77% 는 bf16 수치 문제) |
| `e1_offline.py` | E1 오프라인 α: `gen`(target 출력) → `score`(teacher-forcing 위치별 지표) → `sim`(γ 별 라운드 재현, 하드웨어별 예측) |

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

## E0 · E1 (W03)

> `e1_offline.py` 는 `lang.py`(`gen`·`score`) + `lang_plot.py`(`simulate`)와 기능이 겹쳐 더 쓰지 않는다. W03 이후 오프라인 α 는 아래 EXP-003 파이프라인으로 잰다 (기록용으로 남겨 둠).

```bash
# E0 — CPU fp32 vs bf16. target 1.7B / draft 0.6B (fp32 8GB 정도면 충분)
.venv/bin/python e0_fp32.py --out results/e0

# E1 — α 는 하드웨어와 무관하게 한 번만 잰다 (greedy 는 teacher-forcing 이 온라인 SD 와 정확히 같은 일치열을 준다)
.venv/bin/python e1_offline.py gen   --prompts bench --tail 24 --out results/e1_w02    # target 출력 y (+ 검증용 tail)
.venv/bin/python e1_offline.py score --out results/e1_w02                              # 위치별 m_i, 1−TV, KL, H(p), top-k logit
.venv/bin/python e1_offline.py sim   --out results/e1_w02 --validate results/main results/main_rep1 \
                                     --costs results/main_rep1/costs.json              # γ 별 α·τ + W02 온라인과 비교
```

| `e1_offline.py` 산출물 | 내용 |
|---|---|
| `targets.jsonl` | 문장별 prompt 토큰, target greedy 출력 `y`, `tail`(y 뒤 추가 토큰, 검증용). `gen` 은 끊겨도 이어서 돈다 |
| `positions.csv` | 위치 × draft 한 줄: `match`(draft argmax == y), `acc_t1`·`acc_t0.7`(1−TV = sampling 기대 수락률), `kl`, `h_p`, `p_y`·`q_y`, `nchar`(그 토큰으로 늘어난 글자 수), `t_self`(teacher-forcing target argmax == y, 수치 확인용) |
| `topk/*.npz` | 위치별 target·draft top-k logit (E3 커버리지 마스킹을 forward 없이 다시 계산) |
| `sim.csv` | 문장 × draft × γ: 라운드·수락·거절, τ, α, 글자/라운드, `--costs` 를 주면 Leviathan·실측비용 speedup 예측 |
| `sentences.csv` | 문장 × draft: `alpha_pos`(γ 무관 위치 일치율), mean(1−TV), H(p), 글자/토큰 — H3 짝 비교·bootstrap 의 단위 |
| `summary.md` | 태스크별 α [95% CI], γ 별 표, 검증 결과 |

- `sim` 의 기본 정의(budget 모드)는 y 안에서만 draft 한다. `--validate` 는 `sd.generate` 와 똑같이(끝을 넘겨서도 γ 개) 재현해 온라인 `runs.csv` 와 라운드 단위로 비교한다. 이때 `gen --tail` 이 측정한 γ 최대값 이상이어야 한다.
- `--validate` 는 `e0_fp32.py` 의 `runs.csv` 도 받는다 (score 와 같은 dtype 행만 비교). fp32 에서는 정확히 일치해야 한다.
- 하드웨어가 바뀌면 `bench.py` 로 `costs.json` 만 새로 재서 `sim --costs` 에 넣으면 된다 (E5).
- Windows(CPU)에서는 `.venv\Scripts\python`, `--device cpu --dtype float32` (`--threads N` 으로 스레드 수 고정). `--target`/`--drafts` 에 `Qwen/Qwen3-1.7B` 같은 HF 이름을 넣어도 된다.
- 프롬프트 파일은 jsonl `{"id", "task", "prompt"}` (`prompt` = user 메시지 본문). FLORES+ 는 접근 승인 후 만든다.

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
