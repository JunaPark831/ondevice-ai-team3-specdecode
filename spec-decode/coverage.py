# EXP-003 부록 · 어휘 커버리지 — 다른 계열(이종 어휘) draft 후보를 위한 설명변수 (토크나이저 파일만, 모델 가중치 X)
#   H1 "수락률 하락은 draft-target 어휘 커버리지로 예측된다". 지금 draft(Qwen3-0.6B/1.7B)는 target(Qwen3-8B)과
#   토크나이저가 같아 커버리지가 자명하게 100% → 이종 어휘 SD(Timor et al. 2025, TLI/SLEM) 실험용 변수를 미리 만든다
#   (1) 세분도   : FLORES 정답 200문장의 UTF-8 바이트/토큰, 글자/토큰 (특수 토큰 없이 encode)
#   (2) 커버리지 : Qwen3-8B 출력 토큰(gen.jsonl, 특수 id ≥ 151643 제외) 중 같은 바이트열이 후보 어휘에 토큰 하나로 있는
#                  비율. TLI 는 draft 가 V_d ∩ V_t 안에서만 제안하므로 그 밖의 target 토큰은 draft 가 못 맞힌다.
#                  출력 언어별 전체 / 그 언어 문자(ko 한글, ja 가나·한자, zh 한자, en·de 라틴)가 걸친 토큰만 / 바이트 조각만
#   (3) 어휘     : 크기, 바이트 기준 |V_qwen ∩ V_cand|, FLORES ko 의 한글 음절이 바이트 조각 없이 토큰 하나 안에
#                  통째로 들어가는 비율 (단독 토큰이든 여러 글자 토큰의 일부든)
#   어휘는 전부 바이트열로 비교: byte-level BPE → GPT-2 bytes_to_unicode 의 역, SentencePiece → '▁'=공백, '<0xNN>'=그 바이트.
#   특수 토큰은 빼고, 일반 추가 토큰(EXAONE 공백·탭 묶음, Gemma 줄바꿈·HTML 태그 등)은 원문 그대로 넣는다.
#   세분도는 tokenizer.json 그대로 잘라서 잰다. 실험에서 쓸 transformers AutoTokenizer 가 같게 자르는지도 문장 단위로 센다
#
# 사용법:
#   .venv/bin/hf download <repo> tokenizer.json tokenizer_config.json --local-dir models/tok/<이름>   (CANDS 마다)
#   .venv/bin/python -I coverage.py   → results/lang/coverage.json, coverage.md
import json
import re
from collections import Counter, defaultdict
from itertools import accumulate, combinations
from pathlib import Path

import transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

transformers.logging.set_verbosity_error()

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/lang"
LANGS = ["en", "ko", "ja", "zh", "de"]
NATIVE = {"ko": {"hangul"}, "ja": {"kana", "han"}, "zh": {"han"}, "en": {"latin"}, "de": {"latin"}}  # lang_plot.py 와 같다
CANDS = {"Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
         "Llama-3.2-1B": "unsloth/Llama-3.2-1B-Instruct",  # Llama-3 토크나이저 (gated 아닌 미러)
         "kanana-1.5-2.1b": "kakaocorp/kanana-1.5-2.1b-instruct-2505",
         "HyperCLOVAX-SEED-0.5B": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
         "EXAONE-3.5-2.4B": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
         "EXAONE-4.0-1.2B": "LGAI-EXAONE/EXAONE-4.0-1.2B",
         "gemma-3-1b": "unsloth/gemma-3-1b-it"}  # Gemma-3 토크나이저 미러
QSPEC = 151643  # Qwen3 id ≥ 이 값 = 특수/추가 토큰
U2B = {v: k for k, v in bytes_to_unicode().items()}


def script(ch):  # lang_plot.py 와 같은 문자 범주
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FFFF:
        return "han"
    if ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    return "space" if ch.isspace() else "punct"


PRIO = ["hangul", "kana", "han", "latin", "digit", "punct", "space"]


def cats(bs):
    """토큰 바이트열 목록 → 토큰마다 (걸친 문자의 범주, 조각 여부). lang_plot.categorize 와 같은 규칙:
    조각 = 토큰 경계가 UTF-8 문자 중간을 자른다, 범주 = 걸친 문자들 중 PRIO 순 앞의 것 (조각은 그 문자의 범주)"""
    full = b"".join(bs)
    starts, need = [], 0  # 연속 바이트는 앞 리드 바이트가 기다리는 만큼만 붙인다 (떠돌이 0x80–0xBF 는 단독 단위)
    for k, x in enumerate(full):
        if need and x & 0xC0 == 0x80:
            need -= 1
            continue
        starts.append(k)
        need = 1 if x & 0xE0 == 0xC0 else 2 if x & 0xF0 == 0xE0 else 3 if x & 0xF8 == 0xF0 else 0
    kind = [""] * len(full)
    for s, e in zip(starts, starts[1:] + [len(full)]):
        kind[s:e] = [script(full[s:e].decode("utf-8", "replace")[0])] * (e - s)
    bounds = set(starts) | {len(full)}
    return [(min(set(kind[e - len(b): e]), key=PRIO.index), e - len(b) not in bounds or e not in bounds)
            for b, e in zip(bs, accumulate(map(len, bs)))]


def load(d):
    """tokenizer.json → (Tokenizer, {id: 바이트열} (특수 토큰 제외), SentencePiece 인가, 유형 설명, AutoTokenizer).
    유형은 pre_tokenizer/decoder/normalizer 로 판별: ByteLevel 이 있으면 byte-level BPE, '▁' 치환이 있으면 SentencePiece"""
    j = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    pipe = json.dumps([j.get(k) for k in ("normalizer", "pre_tokenizer", "decoder")], ensure_ascii=False)
    sp, fb = '"ByteLevel"' not in pipe, j["model"].get("byte_fallback")
    assert j["model"]["type"] == "BPE" and (not sp or "▁" in pipe), f"{d}: 바이트 매핑 규칙을 모르는 토크나이저"

    def tb(t):  # 어휘 문자열 → 바이트열
        if not sp:
            return bytes(U2B[c] for c in t)
        return bytes([int(t[3:5], 16)]) if fb and re.fullmatch(r"<0x[0-9A-F]{2}>", t) else t.replace("▁", " ").encode()

    added = {a["id"]: a for a in j["added_tokens"]}  # 추가 토큰은 BPE 전에 원문에서 바로 잘린다 → 어휘 문자열 대신 content
    ib = {i: tb(t) for t, i in j["model"]["vocab"].items() if i not in added}
    ib |= {i: (a["content"].replace("▁", " ") if sp else a["content"]).encode() for i, a in added.items() if not a["special"]}
    nm = (j.get("normalizer") or {}).get("type")
    desc = ("SentencePiece BPE" + (" + byte fallback" if fb else "") if sp else "byte-level BPE") + (f", {nm}" if nm in ("NFC", "NFKC") else "")
    return Tokenizer.from_file(str(d / "tokenizer.json")), ib, sp, desc, AutoTokenizer.from_pretrained(d)


toks = {"Qwen3-8B": load(HERE / "models/Qwen3-8B")}
toks |= {n: load(HERE / "models/tok" / n) for n in CANDS if (HERE / "models/tok" / n / "tokenizer.json").exists()}
skipped = [n for n in CANDS if n not in toks]
qb = toks["Qwen3-8B"][1]
QV = {qb[i] for i in range(QSPEC)}  # target 어휘 = 일반 토큰 151,643개

sents = [json.loads(l) for l in open(HERE / "data/flores.jsonl", encoding="utf-8")]
outs = defaultdict(list)  # 출력 언어 → [(Qwen3 출력 토큰 바이트열, 그 언어 문자 토큰인가, 바이트 조각인가)]
for r in map(json.loads, open(OUT / "gen.jsonl", encoding="utf-8")):
    bs = [qb[i] for i in r["out"] if i < QSPEC]
    assert b"".join(bs).decode("utf-8", "replace") == r["text"], r  # self-check: 바이트 매핑 == 저장된 tok.decode 결과
    outs[r["tgt"]] += [(b, c in NATIVE[r["tgt"]], f) for b, (c, f) in zip(bs, cats(bs))]


def share(V, o):
    return sum(b in V for b in o) / len(o) if o else None


R, VS = {}, {}
for n, (tok, ib, sp, desc, hf) in toks.items():
    V = VS[n] = set(ib.values())
    gran, occ, whole, hf_diff = {}, Counter(), Counter(), 0
    for lg in LANGS:
        nb = nc = nt = 0
        for x in sents:
            s = x[lg]
            ids = tok.encode(s, add_special_tokens=False).ids
            hf_diff += hf(s, add_special_tokens=False).input_ids != ids
            bs = [ib[i] for i in ids]
            norm = tok.normalizer.normalize_str(s) if tok.normalizer else s  # EXAONE-3.5 는 NFKC (전각 ，→ , 등)
            full = b"".join(bs)
            assert full == (norm.replace("▁", " ") if sp else norm).encode(), (n, lg, x["sid"])  # self-check: 매핑으로 이은 토큰 == 원문
            nb, nc, nt = nb + len(s.encode()), nc + len(s), nt + len(bs)
            if lg == "ko":  # 토큰 경계가 음절(3바이트) 안쪽에 떨어지면 조각 (byte-level 분할 또는 <0xNN> fallback)
                cut, a = set(accumulate(map(len, bs))), 0
                for ch in full.decode():
                    e = a + len(ch.encode())
                    if 0xAC00 <= ord(ch) <= 0xD7A3:
                        occ[ch] += 1
                        whole[ch] += not cut & set(range(a + 1, e))
                    a = e
        gran[lg] = dict(bytes_per_tok=nb / nt, chars_per_tok=nc / nt, n_tok=nt)
    cov = {lg: dict(all=share(V, [b for b, _, _ in o]), native=share(V, [b for b, k, _ in o if k]),
                    frag=share(V, [b for b, _, f in o if f]), nonfrag=share(V, [b for b, _, f in o if not f]),
                    n=len(o), n_native=sum(k for _, k, _ in o), n_frag=sum(f for _, _, f in o)) for lg, o in outs.items()}
    R[n] = dict(repo=CANDS.get(n, "Qwen/Qwen3-8B"), type=desc, vocab_size=tok.get_vocab_size(), n_bytes=len(V),
                inter_qwen=len(V & QV), hangul_vocab=sum(chr(o).encode() in V for o in range(0xAC00, 0xD7A4)),
                hangul_whole=dict(types=sum(whole[c] == occ[c] for c in occ) / len(occ), occ=sum(whole.values()) / sum(occ.values()),
                                  n_types=len(occ), n_occ=sum(occ.values())), gran=gran, cov=cov,
                hf_class=type(hf).__name__, hf_diff=hf_diff)  # AutoTokenizer 가 tokenizer.json 과 다르게 자른 문장 수 (/5×200)
    print(n, desc, {lg: round(c["all"], 4) for lg, c in cov.items()}, flush=True)
assert all(c["all"] == c["native"] == 1 for c in R["Qwen3-8B"]["cov"].values())  # self-check: target 자신은 100%

names = list(R)
symdiff = {(a, b): len(VS[a] ^ VS[b]) for a, b in combinations(names, 2)}
for n in names:  # 어휘가 (일반 추가 토큰 몇 개 빼고) 같은 토크나이저
    R[n]["same_vocab"] = {b if a == n else a: d for (a, b), d in symdiff.items() if n in (a, b) and d < 100}
(OUT / "coverage.json").write_text(json.dumps(dict(target="Qwen3-8B", skipped=skipped, tokenizers=R,
                                                   overlap={a: {b: len(VS[a] & VS[b]) for b in names} for a in names}),
                                              ensure_ascii=False, indent=1))


# ---------------- coverage.md ----------------
def pct(v, d=1):  # 반올림해서 100 이 되는 값은 100 으로 안 쓴다
    return "·" if v is None else "100" if v == 1 else f"{min(100 * v, 100 - 0.6 * 10 ** -d):.{d}f}"


def label(n):
    return f"{n} (target)" if n == "Qwen3-8B" else n


X = [n for n in names if R[n]["inter_qwen"] < QSPEC]  # 이종 어휘 후보 (Qwen 계열 제외)


def rank(f):  # 높은 순, 값이 정확히 같으면 = 로 묶음
    groups = defaultdict(list)
    for n in sorted(X, key=f, reverse=True):
        groups[f(n)].append(n)
    return " > ".join(f"{' = '.join(ns)} {pct(v, 2)}" for v, ns in groups.items())


C0 = R["Qwen3-8B"]["cov"]
L = ["# 어휘 커버리지 — 이종 어휘 draft 후보 (EXP-003 부록, 토크나이저 파일만)\n",
     "- target 출력 = `results/lang/gen.jsonl` (Qwen3-8B greedy, 14방향 × 200문장, 특수 토큰 id ≥ 151643 제외) · "
     "정답 문장 = FLORES-200 devtest 200문장 (`data/flores.jsonl`)",
     "- 어휘 비교는 바이트열 기준 (byte-level BPE: bytes_to_unicode 역 · SentencePiece: ▁=공백, <0xNN>=그 바이트) · "
     "특수 토큰 제외, 일반 추가 토큰 포함",
     f"- 토크나이저 파일만 `models/tok/<이름>/` 에 받음 (가중치 없음) · 받지 못한 후보: {', '.join(skipped) or '없음'} · 생성: `coverage.py`\n",
     "## 1. 세분도 — FLORES 정답 문장의 바이트/토큰 · 글자/토큰 (tokenizer.json, 특수 토큰 없이 encode)\n",
     f"마지막 칸 = transformers {transformers.__version__} `AutoTokenizer` 가 tokenizer.json 과 다른 id 를 낸 문장 수 (5개 언어 × 200)\n",
     "| 토크나이저 | " + " | ".join(LANGS) + " | AutoTokenizer 불일치 |\n|---|" + "---|" * (len(LANGS) + 1)]
L += [f"| {label(n)} | " + " | ".join(f"{r['gran'][lg]['bytes_per_tok']:.2f} · {r['gran'][lg]['chars_per_tok']:.2f}" for lg in LANGS)
      + f" | {r['hf_diff']} ({r['hf_class']}) |" for n, r in R.items()]
L += ["\n## 2. TLI 커버리지 — Qwen3-8B 출력 토큰 중 같은 바이트열이 후보 어휘에 토큰 하나로 있는 비율 (%)\n",
      "칸 = 전체 / 그 언어 문자가 걸친 토큰만 (ko 한글, ja 가나·한자, zh 한자, en·de 라틴). ko·ja 조각 = Qwen3 출력 중 "
      "UTF-8 문자 중간을 자르는 바이트 조각 토큰(예: `' 박'` → `' \\xeb\\xb0'` + `'\\x95'`)만 본 커버리지\n",
      "| 토크나이저 | " + " | ".join(LANGS) + " | ko 조각 | ja 조각 |\n|---|" + "---|" * (len(LANGS) + 2),
      "| *Qwen3 출력 토큰 수* | " + " | ".join(f"{C0[lg]['n']:,} / {C0[lg]['n_native']:,}" for lg in LANGS) + " | "
      + " | ".join(f"{C0[lg]['n_frag']:,} ({C0[lg]['n_frag'] / C0[lg]['n']:.1%})" for lg in ("ko", "ja")) + " |"]
L += [f"| {label(n)} | " + " | ".join(f"{pct(r['cov'][lg]['all'])} / {pct(r['cov'][lg]['native'])}" for lg in LANGS) + " | "
      + " | ".join(pct(r["cov"][lg]["frag"]) for lg in ("ko", "ja")) + " |" for n, r in R.items()]
L += ["\n## 3. 어휘\n",
      "∩ Qwen3 = 바이트열이 같은 토큰 수 (Qwen3 일반 토큰 151,643개 기준). 같은 어휘 = 바이트열 대칭차 < 100 인 토크나이저 (대칭차). "
      "음절 통째 = FLORES ko 에 나온 한글 음절 중 토큰 경계가 음절 안쪽에 안 떨어지는 비율 (종류: 모든 출현이 통째인 음절 / 출현)\n",
      "| 토크나이저 | HF repo | 유형 | 어휘 크기 | 바이트열 수 | ∩ Qwen3 | ∩ / Qwen3 | 같은 어휘 | 단독 토큰 한글 음절 (/11,172) | "
      "FLORES ko 음절 통째 (종류 / 출현) |\n|---|---|---|---|---|---|---|---|---|---|"]
for n, r in R.items():
    h = r["hangul_whole"]
    L.append(f"| {label(n)} | `{r['repo']}` | {r['type']} | {r['vocab_size']:,} | {r['n_bytes']:,} | {r['inter_qwen']:,} | "
             f"{r['inter_qwen'] / QSPEC:.1%} | {', '.join(f'{m} ({d})' for m, d in r['same_vocab'].items()) or '·'} | "
             f"{r['hangul_vocab']:,} | {h['types']:.1%} / {h['occ']:.1%} |")
pairs = [(a, b) for (a, b), d in symdiff.items() if d < 100]
n_vocab = sum(not any(m in R[n]["same_vocab"] for m in names[:i]) for i, n in enumerate(names))
frag_bl = [R[n]["cov"]["ko"]["frag"] for n in X if not toks[n][2]]
L += ["\n## 관찰 (위 표의 숫자 그대로, 순위는 이종 어휘 후보만)\n",
      f"- 어휘가 (일반 추가 토큰 몇 개 빼고) 같은 쌍: {', '.join(f'{a}·{b} (대칭차 {symdiff[(a, b)]})' for a, b in pairs)} "
      f"→ 토크나이저 {len(names)}개(target 포함)의 어휘는 {n_vocab}종"
      + (", 쌍끼리 커버리지 값이 같다" if all(R[a]["cov"] == R[b]["cov"] for a, b in pairs) else ""),
      f"- ko 커버리지: {rank(lambda n: R[n]['cov']['ko']['all'])}. "
      f"FLORES ko 음절 통째(출현): {rank(lambda n: R[n]['hangul_whole']['occ'])}",
      f"- Qwen3 자신이 FLORES ko 음절 출현의 {1 - R['Qwen3-8B']['hangul_whole']['occ']:.1%} 를 바이트로 쪼개고, Qwen3 출력 토큰의 "
      f"ko {C0['ko']['n_frag'] / C0['ko']['n']:.1%} · ja {C0['ja']['n_frag'] / C0['ja']['n']:.1%} 가 바이트 조각이다. "
      f"ko 조각 토큰 커버리지는 byte-level BPE 후보 {pct(min(frag_bl))}~{pct(max(frag_bl))}%, SentencePiece 후보 "
      + ", ".join(f"{n} {pct(R[n]['cov']['ko']['frag'])}% (조각 아닌 ko 토큰은 {pct(R[n]['cov']['ko']['nonfrag'])}%)" for n in X if toks[n][2]),
      f"- zh 커버리지: {rank(lambda n: R[n]['cov']['zh']['all'])}. ja: {rank(lambda n: R[n]['cov']['ja']['all'])}. "
      f"de: {rank(lambda n: R[n]['cov']['de']['all'])}. en: {rank(lambda n: R[n]['cov']['en']['all'])}",
      f"- transformers {transformers.__version__} AutoTokenizer 가 tokenizer.json 과 다르게 자르는 후보: "
      + (", ".join(f"{n} {R[n]['hf_diff']}/{len(sents) * len(LANGS):,}문장 ({R[n]['hf_class']})" for n in names if R[n]["hf_diff"]) or "없음")
      + " — 이 표의 세분도는 tokenizer.json 기준"]
(OUT / "coverage.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
