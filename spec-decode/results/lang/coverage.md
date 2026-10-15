# 어휘 커버리지 — 이종 어휘 draft 후보 (EXP-003 부록, 토크나이저 파일만)

- target 출력 = `results/lang/gen.jsonl` (Qwen3-8B greedy, 14방향 × 200문장, 특수 토큰 id ≥ 151643 제외) · 정답 문장 = FLORES-200 devtest 200문장 (`data/flores.jsonl`)
- 어휘 비교는 바이트열 기준 (byte-level BPE: bytes_to_unicode 역 · SentencePiece: ▁=공백, <0xNN>=그 바이트) · 특수 토큰 제외, 일반 추가 토큰 포함
- 토크나이저 파일만 `models/tok/<이름>/` 에 받음 (가중치 없음) · 받지 못한 후보: 없음 · 생성: `coverage.py`

## 1. 세분도 — FLORES 정답 문장의 바이트/토큰 · 글자/토큰 (tokenizer.json, 특수 토큰 없이 encode)

마지막 칸 = transformers 5.18.0 `AutoTokenizer` 가 tokenizer.json 과 다른 id 를 낸 문장 수 (5개 언어 × 200)

| 토크나이저 | en | ko | ja | zh | de | AutoTokenizer 불일치 |
|---|---|---|---|---|---|---|
| Qwen3-8B (target) | 4.78 · 4.78 | 3.54 · 1.47 | 4.27 · 1.45 | 4.34 · 1.54 | 3.63 · 3.57 | 0 (Qwen2Tokenizer) |
| Qwen2.5-0.5B | 4.78 · 4.78 | 3.54 · 1.47 | 4.27 · 1.45 | 4.34 · 1.54 | 3.63 · 3.57 | 0 (Qwen2Tokenizer) |
| Llama-3.2-1B | 4.85 · 4.85 | 3.93 · 1.63 | 4.04 · 1.38 | 3.46 · 1.23 | 3.63 · 3.57 | 0 (TokenizersBackend) |
| kanana-1.5-2.1b | 4.85 · 4.85 | 3.93 · 1.63 | 4.04 · 1.38 | 3.46 · 1.23 | 3.63 · 3.57 | 0 (TokenizersBackend) |
| HyperCLOVAX-SEED-0.5B | 4.85 · 4.84 | 4.84 · 2.01 | 2.70 · 0.92 | 2.36 · 0.84 | 3.62 · 3.56 | 171 (GPT2Tokenizer) |
| EXAONE-3.5-2.4B | 4.77 · 4.76 | 4.64 · 1.92 | 2.99 · 1.02 | 2.10 · 0.75 | 3.14 · 3.09 | 199 (GPT2Tokenizer) |
| EXAONE-4.0-1.2B | 4.77 · 4.76 | 4.64 · 1.92 | 2.97 · 1.01 | 1.98 · 0.70 | 3.14 · 3.09 | 0 (GPT2Tokenizer) |
| gemma-3-1b | 4.90 · 4.90 | 4.27 · 1.77 | 5.04 · 1.72 | 4.15 · 1.47 | 4.36 · 4.29 | 0 (GemmaTokenizer) |

## 2. TLI 커버리지 — Qwen3-8B 출력 토큰 중 같은 바이트열이 후보 어휘에 토큰 하나로 있는 비율 (%)

칸 = 전체 / 그 언어 문자가 걸친 토큰만 (ko 한글, ja 가나·한자, zh 한자, en·de 라틴). ko·ja 조각 = Qwen3 출력 중 UTF-8 문자 중간을 자르는 바이트 조각 토큰(예: `' 박'` → `' \xeb\xb0'` + `'\x95'`)만 본 커버리지

| 토크나이저 | en | ko | ja | zh | de | ko 조각 | ja 조각 |
|---|---|---|---|---|---|---|---|
| *Qwen3 출력 토큰 수* | 21,772 / 18,561 | 26,658 / 24,070 | 23,546 / 20,575 | 15,212 / 12,600 | 8,174 / 7,298 | 3,800 (14.3%) | 1,647 (7.0%) |
| Qwen3-8B (target) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 | 100 |
| Qwen2.5-0.5B | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 | 100 |
| Llama-3.2-1B | 100 / 100 | 97.8 / 97.7 | 89.5 / 88.0 | 72.6 / 66.9 | 99.0 / 98.9 | 99.7 | 99.9 |
| kanana-1.5-2.1b | 100 / 100 | 97.8 / 97.7 | 89.5 / 88.0 | 72.6 / 66.9 | 99.0 / 98.9 | 99.7 | 99.9 |
| HyperCLOVAX-SEED-0.5B | 100 / 100 | 99.6 / 99.8 | 59.0 / 53.2 | 45.0 / 33.6 | 99.0 / 98.9 | 99.9 | 92.2 |
| EXAONE-3.5-2.4B | 98.3 / 98.1 | 92.2 / 91.9 | 62.9 / 64.2 | 28.7 / 22.8 | 84.8 / 83.0 | 93.2 | 95.7 |
| EXAONE-4.0-1.2B | 98.3 / 98.1 | 92.2 / 91.9 | 62.9 / 64.2 | 28.7 / 22.8 | 84.8 / 83.0 | 93.2 | 95.7 |
| gemma-3-1b | 98.9 / 98.7 | 92.8 / 92.2 | 95.5 / 94.9 | 92.4 / 90.9 | 98.7 / 98.5 | 52.3 | 49.9 |

## 3. 어휘

∩ Qwen3 = 바이트열이 같은 토큰 수 (Qwen3 일반 토큰 151,643개 기준). 같은 어휘 = 바이트열 대칭차 < 100 인 토크나이저 (대칭차). 음절 통째 = FLORES ko 에 나온 한글 음절 중 토큰 경계가 음절 안쪽에 안 떨어지는 비율 (종류: 모든 출현이 통째인 음절 / 출현)

| 토크나이저 | HF repo | 유형 | 어휘 크기 | 바이트열 수 | ∩ Qwen3 | ∩ / Qwen3 | 같은 어휘 | 단독 토큰 한글 음절 (/11,172) | FLORES ko 음절 통째 (종류 / 출현) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (target) | `Qwen/Qwen3-8B` | byte-level BPE, NFC | 151,669 | 151,655 | 151,643 | 100.0% | Qwen2.5-0.5B (4) | 2,512 | 54.9% / 93.2% |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | byte-level BPE, NFC | 151,665 | 151,651 | 151,643 | 100.0% | Qwen3-8B (4) | 2,512 | 54.9% / 93.2% |
| Llama-3.2-1B | `unsloth/Llama-3.2-1B-Instruct` | byte-level BPE | 128,256 | 128,000 | 109,566 | 72.3% | kanana-1.5-2.1b (0) | 599 | 71.5% / 96.9% |
| kanana-1.5-2.1b | `kakaocorp/kanana-1.5-2.1b-instruct-2505` | byte-level BPE | 128,259 | 128,000 | 109,566 | 72.3% | Llama-3.2-1B (0) | 599 | 71.5% / 96.9% |
| HyperCLOVAX-SEED-0.5B | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` | byte-level BPE, NFC | 110,524 | 110,470 | 100,822 | 66.5% | · | 886 | 87.8% / 97.2% |
| EXAONE-3.5-2.4B | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | byte-level BPE, NFKC | 102,400 | 102,074 | 51,559 | 34.0% | EXAONE-4.0-1.2B (6) | 1,399 | 99.1% / 99.9% |
| EXAONE-4.0-1.2B | `LGAI-EXAONE/EXAONE-4.0-1.2B` | byte-level BPE | 102,400 | 102,080 | 51,559 | 34.0% | EXAONE-3.5-2.4B (6) | 1,399 | 99.1% / 99.9% |
| gemma-3-1b | `unsloth/gemma-3-1b-it` | SentencePiece BPE + byte fallback | 262,145 | 262,026 | 95,968 | 63.3% | · | 1,733 | 99.5% / 99.9% |

## 관찰 (위 표의 숫자 그대로, 순위는 이종 어휘 후보만)

- 어휘가 (일반 추가 토큰 몇 개 빼고) 같은 쌍: Qwen3-8B·Qwen2.5-0.5B (대칭차 4), Llama-3.2-1B·kanana-1.5-2.1b (대칭차 0), EXAONE-3.5-2.4B·EXAONE-4.0-1.2B (대칭차 6) → 토크나이저 8개(target 포함)의 어휘는 5종, 쌍끼리 커버리지 값이 같다
- ko 커버리지: HyperCLOVAX-SEED-0.5B 99.56 > Llama-3.2-1B = kanana-1.5-2.1b 97.81 > gemma-3-1b 92.80 > EXAONE-3.5-2.4B = EXAONE-4.0-1.2B 92.24. FLORES ko 음절 통째(출현): gemma-3-1b 99.94 > EXAONE-3.5-2.4B = EXAONE-4.0-1.2B 99.90 > HyperCLOVAX-SEED-0.5B 97.20 > Llama-3.2-1B = kanana-1.5-2.1b 96.87
- Qwen3 자신이 FLORES ko 음절 출현의 6.8% 를 바이트로 쪼개고, Qwen3 출력 토큰의 ko 14.3% · ja 7.0% 가 바이트 조각이다. ko 조각 토큰 커버리지는 byte-level BPE 후보 93.2~99.9%, SentencePiece 후보 gemma-3-1b 52.3% (조각 아닌 ko 토큰은 99.5%)
- zh 커버리지: gemma-3-1b 92.44 > Llama-3.2-1B = kanana-1.5-2.1b 72.62 > HyperCLOVAX-SEED-0.5B 45.00 > EXAONE-3.5-2.4B = EXAONE-4.0-1.2B 28.71. ja: gemma-3-1b 95.53 > Llama-3.2-1B = kanana-1.5-2.1b 89.53 > EXAONE-3.5-2.4B = EXAONE-4.0-1.2B 62.94 > HyperCLOVAX-SEED-0.5B 59.04. de: Llama-3.2-1B = kanana-1.5-2.1b 99.05 > HyperCLOVAX-SEED-0.5B 99.03 > gemma-3-1b 98.69 > EXAONE-3.5-2.4B = EXAONE-4.0-1.2B 84.76. en: Llama-3.2-1B = kanana-1.5-2.1b = HyperCLOVAX-SEED-0.5B 100 > gemma-3-1b 98.88 > EXAONE-3.5-2.4B = EXAONE-4.0-1.2B 98.26
- transformers 5.18.0 AutoTokenizer 가 tokenizer.json 과 다르게 자르는 후보: HyperCLOVAX-SEED-0.5B 171/1,000문장 (GPT2Tokenizer), EXAONE-3.5-2.4B 199/1,000문장 (GPT2Tokenizer) — 이 표의 세분도는 tokenizer.json 기준
