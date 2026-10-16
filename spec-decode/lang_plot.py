# EXP-003 분석: results/lang 의 gen.jsonl + score_<draft>.jsonl.gz (+ online*.csv, replay*.jsonl.gz) → summary.md, *.png
#
# 지표 (모두 "방향 안 합/합" 비율. 방향·draft·γ 끼리는 절대 합치지 않는다)
#   ā      = 내용 일치율: 출력 2번째 토큰 ~ EOS 직전에서 draft greedy 1등 == target 출력 인 비율.
#            EOS 위치는 거의 항상 맞는데 짧은 출력(en/zh)일수록 비중이 커서 언어 비교에서 뺀다.  ← 언어 비교의 주 지표
#   α(γ)   = m 으로 SD 라운드를 재현한 ΣA/Σ(A+R)  ← online α·W02 와 같은 정의 (sd.py 2026-10-14 수정판 집계 규칙)
#   τ(γ)   = Σ(출력-1)/Σ라운드 (실제 작업량), τ∞ = 방향의 m 을 이어 붙여 문장 끝 효과를 뺀 값
#   S(γ)   = Σ(출력-1)·T_t(1) / Σ라운드·(γ·T_d + T_t(γ+1))   ← W02 에서 실측과 ±0.05 로 맞은 비용 모델
# CI = 기사(FLORES URL) 단위 군집 부트스트랩. 모든 방향·draft 에 같은 추출을 써서 방향 간 차이는 paired.
# 1-TV 류는 greedy 궤적 위에서 잰 "정렬 지표"다 (샘플링 SD 의 α 가 아님 — 그건 online T=0.7 로 따로 잰다).
import csv
import difflib
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/lang"
LANG_NAME = {"en": "영어", "ko": "한국어", "ja": "일본어", "zh": "중국어", "de": "독일어"}
OUT_ORDER = ["ko", "en", "ja", "zh", "de"]  # 출력 언어 = 범례·색 순서 고정 (dataviz 팔레트 1~5)
CJK = ["en", "ko", "ja", "zh"]
DIRS = [(s, t) for s in CJK for t in CJK if s != t] + [("en", "de"), ("de", "en")]
DI = {d: i for i, d in enumerate(DIRS)}
NATIVE = {"ko": {"hangul"}, "ja": {"kana", "han"}, "zh": {"han"}, "en": {"latin"}, "de": {"latin"}}
GAMMAS = list(range(1, 25))
HB = [0, 0.05, 0.3, 0.7, 1.2, 2.0, 99]  # target 엔트로피 구간 (nats), 설계 검토에서 미리 정한 고정 경계
HBN = ["<.05", ".05–.3", ".3–.7", ".7–1.2", "1.2–2", "≥2"]
B = 2000
rng = np.random.default_rng(0)


def dname(d):
    return f"{d[0]}→{d[1]}"


# ======================= 데이터 =======================
gen = {(r["src"], r["tgt"], r["sid"]): r for r in map(json.loads, open(D / "gen.jsonl", encoding="utf-8"))}
sents = {x["sid"]: x for x in map(json.loads, open(HERE / "data" / "flores.jsonl", encoding="utf-8"))}
SIDS = sorted(sents)
SI = {s: i for i, s in enumerate(SIDS)}
SC = {}
for p_ in D.glob("score_*.jsonl.gz"):
    try:
        SC[p_.name[6:-9]] = {(r["src"], r["tgt"], r["sid"]): r for r in map(json.loads, gzip.open(p_, "rt"))}
    except (EOFError, OSError, json.JSONDecodeError):
        print("아직 쓰는 중이라 건너뜀:", p_.name)  # score 단계가 돌고 있는 동안
drafts = sorted(SC, key=lambda n: float(n.split("-")[1][:-1]) + 0.1 * ("2507" in n))
n0 = drafts[0]
print("drafts", drafts)


# ======================= 토큰 주석 (byte-level BPE → 바이트 → 문자) =======================
def script(ch):
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


def annotate(ids, tok, u2b):
    """토큰마다 (문자 종류, 조각 종류, 문자 구간). 조각 = 토큰 경계가 UTF-8 문자 중간을 자른다.
    조각 종류: cont = 앞 토큰이 시작한 문자를 마저 채움(앞이 잘림), lead = 다음 토큰으로 이어짐(뒤가 잘림), both.
    문자 종류는 토큰이 걸친 문자들 중 PRIO 앞의 것 ('Ġ한' → hangul, '4개' → hangul). 문자 구간 = 출력 문자열의 [a, b)."""
    bs = [None if i >= 151643 else bytes(u2b[c] for c in tok.convert_ids_to_tokens(i)) for i in ids]
    full = b"".join(b for b in bs if b)
    starts, need = [], 0  # 연속 바이트는 앞 리드 바이트가 기다리는 만큼만 붙인다 (떠돌이 0x80–0xBF 는 단독 단위)
    for k, x in enumerate(full):
        if need and x & 0xC0 == 0x80:
            need -= 1
            continue
        starts.append(k)
        need = 1 if x & 0xE0 == 0xC0 else 2 if x & 0xF0 == 0xE0 else 3 if x & 0xF8 == 0xF0 else 0
    kind, cidx = [""] * len(full), [0] * len(full)  # 바이트마다: 속한 문자의 종류, 문자 번호
    for ci, (s, e) in enumerate(zip(starts, starts[1:] + [len(full)])):
        kind[s:e] = [script(full[s:e].decode("utf-8", "replace")[0])] * (e - s)
        cidx[s:e] = [ci] * (e - s)
    bounds = set(starts) | {len(full)}
    out, a = [], 0
    for b in bs:
        if b is None:
            out.append(("special", "", (len(starts), len(starts)), "", "sym"))
            continue
        e = a + len(b)
        head, tail = a not in bounds, e not in bounds
        frag = "both" if head and tail else "cont" if head else "lead" if tail else ""
        # 어두 한글: 공백 + 한글 첫 바이트(0xEA–0xED)로 시작. 음절(3바이트)을 다 못 담으면 split (예: ' 박' → ' \xeb\xb0' + '\x95')
        wi = ("split" if len(b) < 4 else "whole") if len(b) >= 2 and b[0] == 0x20 and 0xEA <= b[1] <= 0xED else ""
        k_ = min(set(kind[a:e]), key=PRIO.index)
        ty = "sym" if k_ in ("digit", "punct", "space") else "word" if b[0] == 0x20 else "inword"
        out.append((k_, frag, (cidx[a], cidx[e - 1] + 1), wi, ty))
        a = e
    return out, full.decode("utf-8", "replace")


def load_tok():
    from transformers import AutoTokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    tok = AutoTokenizer.from_pretrained(HERE / "models/Qwen3-8B")
    return tok, {v: k for k, v in bytes_to_unicode().items()}


tok, u2b = load_tok()


def copied_chars(src, out):
    """출력 문자 중 원문과 4글자 이상 연속으로 같은 구간에 든 것 (이름·숫자·라틴 고유명사 복사)"""
    cp = np.zeros(len(out), bool)
    for blk in difflib.SequenceMatcher(None, src, out, autojunk=False).get_matching_blocks():
        if blk.size >= 4:
            cp[blk.b: blk.b + blk.size] = True
    return cp


ANN, FLAG = {}, {}
for k, r in gen.items():
    ann, text = annotate(r["out"], tok, u2b)
    src = sents[k[2]][k[0]]
    cp = copied_chars(src, text)
    copy = [bool(b > a and cp[a:b].mean() >= 0.5) for _, _, (a, b), _, _ in ann]
    ANN[k] = [(c, f, cy, wi, ty) for (c, f, _, wi, ty), cy in zip(ann, copy)][1:]  # score 와 같이 j ≥ 1
    letters = [script(ch) for ch in text if script(ch) in ("hangul", "kana", "han", "latin")]
    own = np.mean([x in NATIVE[k[1]] for x in letters]) if letters else 0.0
    sim = difflib.SequenceMatcher(None, src, text, autojunk=False).ratio()
    end = text.rstrip(" .!?\"'”’」』)").rstrip()
    reg = "formal" if end.endswith(("니다", "니까")) else "polite" if end.endswith("요") else "plain" if end.endswith("다") else "other"
    FLAG[k] = dict(own=own, sim=sim, bad=own < 0.5 or sim > 0.8,  # 미리 정한 제외 규칙: 엉뚱한 언어 / 원문 복사
                   leak=k[1] == "ko" and any(script(ch) in ("han", "kana") for ch in text),  # 한국어 출력에 한자·가나가 섞임
                   reg=reg if k[1] == "ko" else "")

# self-check: "뷁" 은 어휘에 없어 바이트 3조각 — 첫 바이트(lead), 가운데(앞뒤 다 잘림 = both), 끝(cont)
_a, _ = annotate(tok("가 뷁").input_ids, tok, u2b)
assert [x[1] for x in _a] == ["", "lead", "both", "cont"] and {x[0] for x in _a} == {"hangul"}, _a
assert annotate(tok(" 박사").input_ids, tok, u2b)[0][0][3] == "split"  # ' 박사' → [' \xeb\xb0', '\x95', '사']
assert annotate(tok(" 사람").input_ids, tok, u2b)[0][0][3] == "whole"  # ' 사람' 은 한 토큰

# 번역 품질 (greedy SD 출력 = target 출력이므로 SD 와 무관한 공변량): chrF vs FLORES 정답
from sacrebleu.metrics import CHRF

_chrf = CHRF()
CHRF_DIR = {d: _chrf.corpus_score([gen[(d[0], d[1], s)]["text"] for s in SIDS], [[sents[s][d[1]] for s in SIDS]]).score for d in DIRS}
BPT = {k: len(r["text"].encode()) / max(1, sum(i < 151643 for i in r["out"])) for k, r in gen.items()}


# ======================= SD 시뮬레이션 =======================
def simulate(m, g):
    """sd.py(2026-10-14 수정판) 의 greedy SD 를 m 으로 재현. m[0] = 출력 2번째 토큰(첫 토큰은 target prefill).
    라운드: m=1 이 이어지는 만큼(최대 γ) 수락. 전부 수락이면 보너스 위치는 draft 가 제안하지 않으므로 건너뛴다.
    수락이 EOS 까지 닿으면 그 라운드에서 끝 (거절 없음)."""
    i, A, R, rounds, n = 0, 0, 0, 0, len(m)
    while i < n:
        k = 0
        while k < g and i + k < n and m[i + k]:
            k += 1
        rounds += 1
        A += k
        if i + k >= n:
            break
        R += k < g
        i += k + 1
    return A, R, rounds


def simulate_log(m, g):
    """라운드별 (수락, 거절) — online 로그와 한 줄씩 대조용"""
    i, n, out = 0, len(m), []
    while i < n:
        k = 0
        while k < g and i + k < n and m[i + k]:
            k += 1
        if i + k >= n:
            out.append((k, False))
            break
        out.append((k, k < g))
        i += k + 1
    return out


assert simulate([1, 1, 0, 1, 1, 1, 1, 0], 2) == (5, 1, 3)  # [수락2+보너스] [수락2+보너스] [수락1, 거절]
assert simulate([1, 1, 1], 4) == (3, 0, 1)  # EOS 까지 다 수락 → 거절 없이 끝
assert simulate([0, 1], 1) == (1, 1, 2)

# ======================= 비용 =======================
# 검증 비용 곡선 r(k) = T_t(k)/T_t(1) 은 W02 측정(results/main_rep1/costs.json, k=1..25)을 쓴다 — 이번 실측 run 안의
# r(5)·r(9) 와 맞는지 확인한다. draft 비용비 c 는 우선순위: ① 실측 run 안의 시간 (draft 한 스텝 중앙값 / vanilla 토큰당 중앙값,
# 측정 조건 그대로) ② W02 ③ costs_clean.json. 따로 잰 비용 측정은 다른 작업 부하에 흔들려서(T_t(1) 67~517 ms) 보조로만 쓴다.
import statistics

_w02 = json.loads((HERE / "results/main_rep1/costs.json").read_text())
Tt = {int(k): v for k, v in _w02["target"].items()}
C_W02 = {n: v / Tt[1] for n, v in _w02["drafts"].items()}
C_INSITU, R_INSITU, T1_INSITU = {}, {}, None
for p_ in sorted(D.glob("online*.csv")):
    rows_ = list(csv.DictReader(open(p_)))
    van_ = [(float(r["total"]) - float(r["ttft"])) / (int(r["n_new"]) - 1) for r in rows_ if r["draft"].startswith("vanilla") and int(r["n_new"]) > 1]
    if not van_:
        continue
    t1 = statistics.median(van_)
    T1_INSITU = T1_INSITU or t1
    for n in {r["draft"] for r in rows_ if not r["draft"].startswith("vanilla")}:
        rs_ = [r for r in rows_ if r["draft"] == n and float(r.get("temp", 0)) == 0]
        if rs_:
            C_INSITU[n] = statistics.median(float(r["t_draft"]) / (int(r["rounds"]) * int(r["gamma"])) for r in rs_) / t1
            for g in sorted({int(r["gamma"]) for r in rs_}):
                R_INSITU[(n, g + 1)] = statistics.median(float(r["t_verify"]) / int(r["rounds"]) for r in rs_ if int(r["gamma"]) == g) / t1
C_CLEAN = {}
if (D / "costs_clean.json").exists():
    cc_ = json.loads((D / "costs_clean.json").read_text())["clean"]
    for n in next(iter(cc_.values()))["drafts"]:
        C_CLEAN[n] = float(np.median([v["drafts"][n] / v["target"]["1"] for v in cc_.values()]))
# 우선순위: 실측 run 안 > W02(조용할 때) > costs_clean. 따로 잰 측정은 부하가 있으면 target 만 느려져 c 가 낮게 나온다
# (이번: 0.6B 실측 run 0.216 vs 따로 잰 0.18)
C = {**C_CLEAN, **C_W02, **C_INSITU}
C_SRC = {n: "실측 run" if n in C_INSITU else "W02" if n in C_W02 else "costs_clean" for n in C}
C_SPREAD = {n: (min(x for x in (C_W02.get(n), C_CLEAN.get(n), C_INSITU.get(n)) if x), max(x for x in (C_W02.get(n), C_CLEAN.get(n), C_INSITU.get(n)) if x)) for n in C}
cost_src = ("r(k) = W02 곡선 (이번 실측 run 안: " + ", ".join(f"{n} r({k}) {v:.3f}" for (n, k), v in sorted(R_INSITU.items())) + f" vs W02 r(5) {Tt[5] / Tt[1]:.3f}, r(9) {Tt[9] / Tt[1]:.3f})"
            + " · c 출처: " + ", ".join(f"{n} {C[n]:.3f} ({C_SRC[n]}; W02 {C_W02.get(n, float('nan')):.3f}, clean {C_CLEAN.get(n, float('nan')):.3f})" for n in C))
R_K = {k: Tt[k] / Tt[1] for k in Tt}  # 검증 비용 곡선 r(k) = T_t(k)/T_t(1)


# ======================= 문장별 충분통계 =======================
# Z[(draft, 이름)] = [방향, 문장] 배열. 비율 지표는 (분자, 분모) 쌍으로 두고 부트스트랩에서 가중합의 비로 계산한다.
CATS = ["hangul", "kana", "han", "latin", "digit", "punct", "space", "frag_lead", "frag_cont", "frag_both", "special"]
Z = defaultdict(lambda: np.zeros((len(DIRS), len(SIDS))))
for n in drafts:
    for (s, t, sid), r in SC[n].items():
        if (s, t) not in DI:
            continue
        di, si = DI[(s, t)], SI[sid]
        m, H, py = r["m"], r["H"], r["py"]
        ann = ANN[(s, t, sid)]
        L = len(m)
        z = lambda name, v: Z[(n, name)].__setitem__((di, si), Z[(n, name)][di, si] + v)
        cont = range(L - 1)  # 내용 위치 = EOS 제외
        z("c_num", sum(m[j] for j in cont))
        z("c_den", L - 1)
        z("ntok", L)
        for g in GAMMAS:
            A, R_, rounds = simulate(m, g)
            z(f"A{g}", A)
            z(f"R{g}", R_)
            z(f"K{g}", rounds)
        for f in ("tv", "tv07", "tvq", "kl", "ce", "H", "Hq"):
            if f in r:
                z(f"sum_{f}", sum(r[f][j] for j in cont))
        z("t_ok_num", sum(r["t_ok"]))
        z("t_ok_den", L)
        z("top5_num", sum(r["rank"][j] < 5 for j in cont))
        z("surp", sum(-np.log(max(py[j], 1e-12)) for j in cont))  # target 이 자기 출력에 준 놀람 (nats/문장)
        for j in cont:
            c, fr, cy, wi, ty = ann[j]
            cat = f"frag_{fr}" if fr else c
            z(f"cat_num_{cat}", m[j])
            z(f"cat_den_{cat}", 1)
            hb = next(b for b in range(len(HB) - 1) if HB[b] <= H[j] < HB[b + 1])
            z(f"hb_num_{hb}", m[j])
            z(f"hb_den_{hb}", 1)
            z(f"ht_num_{hb}_{ty}", m[j])  # 엔트로피 구간 × 토큰 유형 (검증에서 지적: 한국어는 어중 토큰이 60%, 영어 8%)
            z(f"ht_den_{hb}_{ty}", 1)
            if not fr and c in NATIVE[t] and not cy:
                z("nat_num", m[j])
                z("nat_den", 1)
            if not cy:
                z("nc_num", m[j])
                z("nc_den", 1)
            if wi:
                z(f"wi_num_{wi}_{hb}", m[j])
                z(f"wi_den_{wi}_{hb}", 1)
        # 조각 합치기: lead/both 와 그 뒤를 잇는 cont 들을 한 단위로 (단위 일치 = 모두 일치), 단위 엔트로피 = 첫 토큰의 H
        j = 0
        while j < L - 1:
            k = j + 1
            if ann[j][1] in ("lead", "both"):
                while k < L - 1 and ann[k][1] in ("cont", "both"):
                    k += 1
            u = int(all(m[j:k]))
            z("u_num", u)
            z("u_den", 1)
            hb = next(b for b in range(len(HB) - 1) if HB[b] <= H[j] < HB[b + 1])
            z(f"uhb_num_{hb}", u)
            z(f"uhb_den_{hb}", 1)
            j = k
for k, f in FLAG.items():
    if k[:2] in DI:
        Z[("all", "bad")][DI[k[:2]], SI[k[2]]] = f["bad"]
        Z[("all", "leak")][DI[k[:2]], SI[k[2]]] = f["leak"]
        Z[("all", "chrf")][DI[k[:2]], SI[k[2]]] = _chrf.sentence_score(gen[k]["text"], [sents[k[2]][k[1]]]).score
        Z[("all", "bpt")][DI[k[:2]], SI[k[2]]] = BPT[k]

# ======================= 군집 부트스트랩 =======================
urls = sorted({sents[s]["url"] for s in SIDS})
cl = np.array([urls.index(sents[s]["url"]) for s in SIDS])
draws = rng.integers(0, len(urls), (B, len(urls)))
W = np.stack([np.bincount(d, minlength=len(urls))[cl] for d in draws]).astype(float)  # [B, 문장] 가중치
W_ONE = np.ones(len(SIDS))


def ratio(n, num, den, w=None, mask=None):
    """방향별 Σnum/Σden. w=None → 점추정 [방향], w=W → [B, 방향]. mask [방향, 문장] 이면 그 문장만."""
    a, b = Z[(n, num)], Z[(n, den)]
    if mask is not None:
        a, b = a * mask, b * mask
    if w is None:
        return a.sum(1) / b.sum(1)
    return (w @ a.T) / (w @ b.T)


def ci(bs, p=95):
    lo = (100 - p) / 2
    return np.percentile(bs, lo, 0), np.percentile(bs, 100 - lo, 0)


def fci(x, lo, hi, p=3, sign=False):
    f = f"{{:{'+' if sign else ''}.{p}f}}"
    return f"{f.format(x)} [{f.format(lo)}, {f.format(hi)}]"


A_PT = {n: ratio(n, "c_num", "c_den") for n in drafts}
A_BS = {n: ratio(n, "c_num", "c_den", W) for n in drafts}
GOOD = 1 - Z[("all", "bad")]

L = []
L.append("# 결과 요약 — EXP-003 언어 행렬 (FLORES-200)\n")
L.append(f"- target Qwen3-8B (bf16) · drafts {', '.join(drafts)} · greedy · FLORES-200 devtest {len(SIDS)}문장({len(urls)}개 기사) × {len(DIRS)}방향 · 영어 지시문")
L.append(f"- 값 = 방향 안 Σ/Σ 비율 [95% CI: 기사 단위 군집 부트스트랩 B={B}, 모든 방향에 같은 추출 → 방향 간 차이는 paired]")
L.append(f"- ā = 내용 일치율 (출력 2번째~EOS 직전, draft greedy 1등 == target 토큰) · α(γ) = SD 라운드 재현 ΣA/Σ(A+R)")
L.append(f"- 비용: {cost_src} · ms 환산은 W02 T_t(1) = {Tt[1] * 1e3:.1f} ms (부하 없을 때)" + (f", 이번 실측 중 vanilla 는 {T1_INSITU * 1e3:.1f} ms/토큰" if T1_INSITU else ""))
tok_all = Z[(n0, "t_ok_den")].sum()
tok_ok = Z[(n0, "t_ok_num")].sum()
nbad = Counter(k[:2] for k, f in FLAG.items() if f["bad"])
L.append(f"- 수치 확인: teacher forcing target 1등 == 배치 생성 토큰 {tok_ok / tok_all:.2%} · 제외 규칙(출력 문자의 목표 문자 비율 <0.5 또는 원문 유사도 >0.8) 해당 {sum(nbad.values())}개"
         + (" (" + ", ".join(f"{dname(d)} {c}" for d, c in nbad.most_common()) + ")" if nbad else "") + "\n")

# ======================= 1. 방향별 =======================
L.append("## 1. 방향별 내용 일치율 ā (EOS 제외) — 나쁜 순\n")
L.append("chrF 는 출력 문자 체계에 따라 눈금이 달라(한글 음절·한자 vs 라틴 글자) **같은 출력 언어끼리만** 비교한다.\n")
hdr = ["방향", "chrF", "출력 토큰", "바이트/토큰"] + [f"ā · {n}" for n in drafts] + [f"α(γ=4) · {n0}", f"제외 규칙 적용 ā · {n0}"]
L.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
order = list(np.argsort(A_PT[n0]))
a_good = ratio(n0, "c_num", "c_den", mask=GOOD)
for i in order:
    d = DIRS[i]
    lo_hi = {n: ci(A_BS[n][:, i]) for n in drafts}
    a4 = Z[(n0, "A4")][i].sum() / (Z[(n0, "A4")][i].sum() + Z[(n0, "R4")][i].sum())
    row = [dname(d), f"{CHRF_DIR[d]:.1f}", f"{Z[(n0, 'ntok')][i].mean() + 1:.0f}", f"{Z[('all', 'bpt')][i].mean():.2f}"]
    row += [fci(A_PT[n][i], *lo_hi[n]) for n in drafts] + [f"{a4:.3f}", f"{a_good[i]:.3f}"]
    L.append("| " + " | ".join(row) + " |")
for n in drafts:
    low = np.bincount(A_BS[n].argmin(1), minlength=len(DIRS)) / B
    top = np.argsort(-low)[:3]
    L.append(f"\n{n}: 14방향 중 최저일 부트스트랩 확률 — " + ", ".join(f"{dname(DIRS[i])} {low[i]:.2f}" for i in top if low[i] > 0))
for n in drafts:
    L.append(f"\n**ā 행렬 · {n}** (행 = 원문, 열 = 출력)\n")
    L.append("| 원문 \\ 출력 | " + " | ".join(OUT_ORDER) + " |\n|---|" + "---|" * len(OUT_ORDER))
    for s in OUT_ORDER:
        L.append(f"| {s} | " + " | ".join(f"{A_PT[n][DI[(s, t)]]:.3f}" if (s, t) in DI else "—" for t in OUT_ORDER) + " |")


# ======================= 2. 원문 vs 출력 언어 효과 (P1) =======================
def effects(cell, langs_s, langs_t):
    """α_st = μ + a_s + b_t (합=0 제약) 최소제곱. 대각선이 없어 두 요인이 직교하지 않는다 → 교차항을 따로 보고."""
    cells = [(s, t) for s in langs_s for t in langs_t if s != t]
    y = np.array([cell[DI[c]] for c in cells])

    def eff(levels, x):
        return np.array([1.0 if x == l else -1.0 if x == levels[-1] else 0.0 for l in levels[:-1]])

    X = np.array([np.concatenate([[1.0], eff(langs_s, s), eff(langs_t, t)]) for s, t in cells])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    na = len(langs_s) - 1
    a = np.append(beta[1:1 + na], -beta[1:1 + na].sum())
    b = np.append(beta[1 + na:], -beta[1 + na:].sum())
    res = y - X @ beta
    return a, b, res, y


L.append("\n## 2. 원문 언어 효과 vs 출력 언어 효과 (P1) — 12셀 (en·ko·ja·zh)\n")
L.append("모형 ā_st = μ + a_s(원문) + b_t(출력), 합=0 제약. V_out = 같은 원문에서 출력만 바꾼 셀 쌍 차이²의 평균, V_in = 같은 출력에서 원문만 바꾼 쌍. "
         "D = Σb̂² − Σâ² (부트스트랩 분산 보정). P1 판정: D 의 95% 하한 > 0.\n")
L.append("| draft | 원문 효과 â (en/ko/ja/zh) | 출력 효과 b̂ (en/ko/ja/zh) | 잔차 RMS | V_out / V_in | D [95% CI] | P1 |\n|---|---|---|---|---|---|---|")
P1 = {}
for n in drafts:
    a, b, res, y = effects(A_PT[n], CJK, CJK)
    bs = [effects(A_BS[n][r], CJK, CJK) for r in range(B)]
    va, vb = np.var([x[0] for x in bs], 0), np.var([x[1] for x in bs], 0)
    Dpt = (b ** 2 - vb).sum() - (a ** 2 - va).sum()
    Dbs = np.array([(x[1] ** 2 - vb).sum() - (x[0] ** 2 - va).sum() for x in bs])

    def vio(cell):
        vout = np.mean([(cell[DI[(s, t)]] - cell[DI[(s, u)]]) ** 2 for s in CJK for t in CJK for u in CJK if len({s, t, u}) == 3 and t < u])
        vin = np.mean([(cell[DI[(s, t)]] - cell[DI[(u, t)]]) ** 2 for t in CJK for s in CJK for u in CJK if len({s, t, u}) == 3 and s < u])
        return vout, vin

    vo, vi = vio(A_PT[n])
    lo, hi = ci(Dbs)
    P1[n] = lo > 0
    L.append(f"| {n} | {' / '.join(f'{x:+.3f}' for x in a)} | {' / '.join(f'{x:+.3f}' for x in b)} | {np.sqrt((res ** 2).mean()):.3f} | "
             f"{vo:.4f} / {vi:.4f} | {fci(Dpt, lo, hi, 4)} | {'지지' if lo > 0 else '판정 불가' if hi > 0 else '기각'} |")
# 민감도: 원문이 번역문(ko·ja·zh)인 9셀만 — FLORES 는 영어 원문을 번역해 만들었다 (C5)
L.append("\n민감도 (원문 = 번역문 ko·ja·zh 9셀): " + "; ".join(
    f"{n} 출력 효과 b̂ (en/ko/ja/zh) = " + " / ".join(f"{x:+.3f}" for x in effects(A_PT[n], ["ko", "ja", "zh"], CJK)[1]) for n in drafts))

# ======================= 3. paired 차이 (P2, P3) =======================
PAIRS = [(("en", "ko"), ("ko", "en")), (("en", "ko"), ("en", "ja")), (("en", "ko"), ("en", "zh")), (("en", "ko"), ("en", "de")),
         (("en", "ja"), ("en", "zh")), (("ko", "en"), ("ja", "en")), (("ko", "en"), ("zh", "en")), (("ko", "en"), ("de", "en")),
         (("ja", "ko"), ("ja", "en")), (("zh", "ko"), ("zh", "en")), (("ja", "ko"), ("zh", "ko")),
         (("ko", "zh"), ("en", "ko")), (("ko", "ja"), ("en", "ja"))]
L.append("\n## 3. 같은 문장끼리 차이 (paired ā, [95% CI])\n")
L.append("| 비교 | " + " | ".join(drafts) + f" | 복사 토큰 뺀 ā · {n0} |\n|---|" + "---|" * (len(drafts) + 1))
NC_PT, NC_BS = ratio(n0, "nc_num", "nc_den"), ratio(n0, "nc_num", "nc_den", W)
for x, y_ in PAIRS:
    i, j = DI[x], DI[y_]
    cells = [fci(A_PT[n][i] - A_PT[n][j], *ci(A_BS[n][:, i] - A_BS[n][:, j]), sign=True) for n in drafts]
    cells.append(fci(NC_PT[i] - NC_PT[j], *ci(NC_BS[:, i] - NC_BS[:, j]), sign=True))
    L.append(f"| {dname(x)} − {dname(y_)} | " + " | ".join(cells) + " |")


def tost(i, j, n, margin=0.03):
    d = A_PT[n][i] - A_PT[n][j]
    bs = A_BS[n][:, i] - A_BS[n][:, j]
    lo90, hi90 = ci(bs, 90)
    lo95, hi95 = ci(bs, 95)
    if -margin < lo90 and hi90 < margin:
        return "지지", d
    if hi95 < -margin or lo95 > margin:
        return "기각", d
    return "판정 불가", d


L.append("\n**사전 예측 판정**\n")
for n in drafts:
    r1 = tost(DI[("ko", "en")], DI[("ja", "en")], n)
    r2 = tost(DI[("ko", "en")], DI[("zh", "en")], n)
    p2 = "기각" if "기각" in (r1[0], r2[0]) else "지지" if r1[0] == r2[0] == "지지" else "판정 불가"
    lb = lambda a_, b_: np.percentile(A_BS[n][:, DI[a_]] - A_BS[n][:, DI[b_]], 5)
    lbs = [lb(("en", "zh"), ("en", "ja")), lb(("en", "zh"), ("en", "ko")), lb(("en", "ja"), ("en", "ko"))]
    p_order = np.mean((A_BS[n][:, DI[("en", "zh")]] > A_BS[n][:, DI[("en", "ja")]]) & (A_BS[n][:, DI[("en", "ja")]] > A_BS[n][:, DI[("en", "ko")]]))
    L.append(f"- {n} · P2 (|ko→en − ja→en|, |ko→en − zh→en| < 0.03, TOST): {r1[1]:+.3f} {r1[0]}, {r2[1]:+.3f} {r2[0]} → **{p2}**")
    L.append(f"- {n} · P3 (en→zh > en→ja > en→ko, 단측 95% 하한 모두 > 0): 하한 {', '.join(f'{x:+.3f}' for x in lbs)} → **{'지지' if min(lbs) > 0 else '판정 불가'}** (순서 성립 부트스트랩 확률 {p_order:.2f})")

# P6: draft 가 커지면 한국어 출력 격차가 줄어드나 — 확률 척도에선 α 가 1 에 가까워지면 저절로 줄므로 logit 척도에서 본다
def logit(x):
    return np.log(x / (1 - x))


if len(drafts) >= 2:
    ie, ik = [DI[d] for d in [("ja", "en"), ("zh", "en")]], [DI[d] for d in [("ja", "ko"), ("zh", "ko")]]

    def lg_gap(n, w=None):
        a = A_PT[n] if w is None else A_BS[n]
        if w is None:
            return logit(a[ie].mean()) - logit(a[ik].mean())
        return logit(a[:, ie].mean(1)) - logit(a[:, ik].mean(1))

    for n in drafts[1:]:
        did = lg_gap(n0) - lg_gap(n)
        bd = lg_gap(n0, W) - lg_gap(n, W)
        lo, hi = ci(bd)
        L.append(f"- P6 ({n0} → {n}): 영어 출력 − 한국어 출력 격차 (원문 ja·zh, logit) {lg_gap(n0):.3f} → {lg_gap(n):.3f}, 줄어든 양 {fci(did, lo, hi, sign=True)} → "
                 f"**{'지지' if lo > 0 else '기각' if hi < 0 else '판정 불가'}** (확률 척도: {A_PT[n0][ie].mean() - A_PT[n0][ik].mean():+.3f} → {A_PT[n][ie].mean() - A_PT[n][ik].mean():+.3f})")

# ======================= 4. 토큰 범주 (P4, C2 조각화) =======================
L.append(f"\n## 4. 토큰 범주별 일치율 ({n0}) — 점유율 / 일치율 (EOS 제외 내용 위치)\n")
L.append("frag_lead = 다음 토큰으로 이어지는 조각 (대부분 '공백+음절 첫 바이트': Qwen BPE 가 공백을 다음 음절의 첫 바이트와 합쳐서 흔한 음절도 쪼개진다) · "
         "frag_cont = 앞 토큰이 시작한 문자를 마저 채우는 조각 · frag_both = 앞뒤 다 잘림 · 조각 합친 ā = lead + 뒤따르는 cont 를 한 단위로(모두 맞아야 일치)\n")
show = ["hangul", "kana", "han", "latin", "digit", "punct", "frag_lead", "frag_cont", "frag_both"]
L.append("| 출력 | " + " | ".join(show) + " | ā | 조각 단위 합친 ā | 고유 문자만 ā (조각·복사 제외) |\n|---|" + "---|" * (len(show) + 3))
for t in OUT_ORDER:
    ds = [DI[d] for d in DIRS if d[1] == t]
    den_all = sum(Z[(n0, f"cat_den_{c}")][ds].sum() for c in CATS)
    cells = []
    for c in show:
        dn = Z[(n0, f"cat_den_{c}")][ds].sum()
        cells.append(f"{dn / den_all:.0%} / {Z[(n0, f'cat_num_{c}')][ds].sum() / dn:.2f}" if dn / den_all >= 0.005 else "·")
    a_all = Z[(n0, "c_num")][ds].sum() / Z[(n0, "c_den")][ds].sum()
    a_u = Z[(n0, "u_num")][ds].sum() / Z[(n0, "u_den")][ds].sum()
    a_n = Z[(n0, "nat_num")][ds].sum() / Z[(n0, "nat_den")][ds].sum()
    L.append(f"| {t} | " + " | ".join(cells) + f" | {a_all:.3f} | {a_u:.3f} | {a_n:.3f} |")


L.append(f"\n**어두 한글 토큰 ({n0} / {drafts[-1]}, 한국어 출력)** — 공백+첫 음절 토큰이 음절을 통째로 담나(whole) 첫 바이트만 담나(split). 같은 target 엔트로피 구간 안 일치율\n")
L.append("| 구간 | " + " | ".join(f"whole · {n} | split · {n}" for n in (n0, drafts[-1])) + " |\n|---|" + "---|---|" * 2)
ko_idx = [DI[d] for d in DIRS if d[1] == "ko"]
for hb in range(len(HB) - 1):
    cells = []
    for n in (n0, drafts[-1]):
        for wi in ("whole", "split"):
            dn = Z[(n, f"wi_den_{wi}_{hb}")][ko_idx].sum()
            cells.append(f"{Z[(n, f'wi_num_{wi}_{hb}')][ko_idx].sum() / dn:.2f} (n={dn:.0f})" if dn >= 30 else "·")
    L.append(f"| {HBN[hb]} | " + " | ".join(cells) + " |")
for n in drafts:
    num_s, den_s = Z[(n, "wi_num_split_0")][ko_idx].sum(0), Z[(n, "wi_den_split_0")][ko_idx].sum(0)
    num_w, den_w = Z[(n, "wi_num_whole_0")][ko_idx].sum(0), Z[(n, "wi_den_whole_0")][ko_idx].sum(0)
    d_pt = num_s.sum() / den_s.sum() - num_w.sum() / den_w.sum()
    d_bs = (W @ num_s) / (W @ den_s) - (W @ num_w) / (W @ den_w)
    L.append(f"- {n}: H<.05 구간 split − whole = {fci(d_pt, *ci(d_bs), sign=True)}")
nb_ = len(HB) - 1
for n in drafts:
    def split_stats(w=None):
        ns = np.stack([Z[(n, f"wi_num_split_{b}")][ko_idx].sum(0) for b in range(nb_)])  # [구간, 문장]
        ds = np.stack([Z[(n, f"wi_den_split_{b}")][ko_idx].sum(0) for b in range(nb_)])
        nw = np.stack([Z[(n, f"wi_num_whole_{b}")][ko_idx].sum(0) for b in range(nb_)])
        dw = np.stack([Z[(n, f"wi_den_whole_{b}")][ko_idx].sum(0) for b in range(nb_)])
        cden = Z[(n, "c_den")][ko_idx].sum(0)
        if w is None:
            ns, ds, nw, dw, cden = ns.sum(1), ds.sum(1), nw.sum(1), dw.sum(1), cden.sum()
        else:
            ns, ds, nw, dw, cden = w @ ns.T, w @ ds.T, w @ nw.T, w @ dw.T, w @ cden
        a_s, a_w = ns / np.maximum(ds, 1), nw / np.maximum(dw, 1)
        ws = ds / ds.sum(-1, keepdims=True)
        gap = (ws * (a_s - a_w)).sum(-1)  # 쪼갠 토큰의 엔트로피 분포로 맞춘 차이
        loss = (ds * (a_w - a_s)).sum(-1) / cden  # 쪼갠 자리가 통음절만큼 맞았다면 한국어 출력 ā 가 오를 양
        return gap, loss
    g_pt, l_pt = split_stats()
    g_bs, l_bs = split_stats(W)
    L.append(f"- {n}: 모든 구간(쪼갠 토큰 분포로 맞춤) split − whole = {fci(g_pt, *ci(g_bs), sign=True)} · 한국어 출력 ā 손해 {fci(l_pt, *ci(l_bs))}")
tot = {wi: sum(Z[(n0, f"wi_den_{wi}_{hb}")][ko_idx].sum() for hb in range(len(HB) - 1)) for wi in ("whole", "split")}
L.append(f"\n어두 한글 토큰 중 split 비율 {tot['split'] / (tot['split'] + tot['whole']):.0%} (문장당 {tot['split'] / len(ko_idx) / len(SIDS):.1f}개)")

# 존댓말: 원문 언어가 한국어 출력의 문체를 정하고, 문체가 일치율을 바꾼다 (합니다체 어미는 길고 예측 쉬운 토큰)
L.append(f"\n**한국어 출력 문체 ({n0})** — 마지막 어미로 분류: 합니다체(-니다) / 해요체(-요) / 한다체(-다). FLORES 한국어 정답은 합니다체 비율 "
         f"{np.mean([sents[s]['ko'].rstrip(' .!?\"').endswith(('니다', '니까')) for s in SIDS]):.0%}\n")
L.append("| 방향 | 합니다체 비율 | ā 합니다체 | ā 한다체 |\n|---|---|---|---|")
for d in [d for d in DIRS if d[1] == "ko"]:
    i = DI[d]
    regs = [FLAG[(d[0], d[1], s)]["reg"] for s in SIDS]
    a_r = lambda rr: (sum(Z[(n0, "c_num")][i, k_] for k_, r_ in enumerate(regs) if r_ == rr) / max(1, sum(Z[(n0, "c_den")][i, k_] for k_, r_ in enumerate(regs) if r_ == rr)))
    nf, npl = regs.count("formal"), regs.count("plain")
    L.append(f"| {dname(d)} | {nf / len(regs):.0%} | {a_r('formal'):.3f} (n={nf}) | {a_r('plain'):.3f} (n={npl}) |")

# 한자·가나 누출: 한국어 출력에 섞인 문장을 짝 비교의 양쪽에서 빼도 격차가 그대로인가
leak_sids = Z[("all", "leak")].max(0) > 0
keep = np.tile(~leak_sids, (len(DIRS), 1)).astype(float)
L.append(f"\n**누출 제외** — 한국어 출력 중 하나라도 한자·가나가 섞인 문장 {leak_sids.sum()}개를 모든 방향에서 빼고 다시 잰 짝 차이 ({n0})\n")
L.append("| 비교 | 전체 | 누출 문장 제외 |\n|---|---|---|")
A_K = ratio(n0, "c_num", "c_den", mask=keep)
for x, y_ in [(("en", "ko"), ("ko", "en")), (("ja", "ko"), ("ja", "en")), (("zh", "ko"), ("zh", "en")), (("en", "ko"), ("en", "de"))]:
    L.append(f"| {dname(x)} − {dname(y_)} | {A_PT[n0][DI[x]] - A_PT[n0][DI[y_]]:+.3f} | {A_K[DI[x]] - A_K[DI[y_]]:+.3f} |")

# target 놀람: 출력 언어가 바뀌면 target 자신의 확신이 얼마나 바뀌나 (토큰화와 무관한 문장 단위 양)
L.append(f"\n**target 의 놀람 Σ−log p(y) (nats/문장, 토큰 수와 무관)** — 원문이 같은 짝\n")
L.append("| 원문 | → en (문장당 / 토큰당) | → ko (문장당 / 토큰당) | ko/en 비 (문장당 / 토큰당) |\n|---|---|---|---|")
for s_ in ["ja", "zh"]:
    e_, k_ = DI[(s_, "en")], DI[(s_, "ko")]
    pe, pk = Z[(n0, "surp")][e_].mean(), Z[(n0, "surp")][k_].mean()
    te, tk = Z[(n0, "surp")][e_].sum() / Z[(n0, "c_den")][e_].sum(), Z[(n0, "surp")][k_].sum() / Z[(n0, "c_den")][k_].sum()
    L.append(f"| {s_} | {pe:.2f} / {te:.3f} | {pk:.2f} / {tk:.3f} | {pk / pe:.1f} / {tk / te:.1f} |")
L.append(f"| (참고) | ko→en {Z[(n0, 'surp')][DI[('ko', 'en')]].mean():.2f} | en→ko {Z[(n0, 'surp')][DI[('en', 'ko')]].mean():.2f} | |")


# ======================= 5. target 엔트로피 (P5, C3 출력 자유도) — Kitagawa 분해 =======================
TYPES = ["word", "inword", "sym"]


def kitagawa(n, ga, gb, w=None, unit=False, by_type=False):
    """두 방향 묶음의 ā 차이 = 구성(셀 분포가 달라서) + 셀 안(같은 셀에서도 달라서). 반환 (차이, 구성, 셀 안).
    셀 = target 엔트로피 구간, by_type 이면 엔트로피 구간 × 토큰 유형(어두/어중/기호)."""
    if by_type:
        cells = [f"ht_{{}}_{b}_{ty}" for b in range(len(HB) - 1) for ty in TYPES]
    else:
        cells = [("uhb" if unit else "hb") + f"_{{}}_{b}" for b in range(len(HB) - 1)]
    ia, ib = [DI[d] for d in ga], [DI[d] for d in gb]

    def grp(idx):
        num = np.stack([Z[(n, c.format("num"))][idx].sum(0) for c in cells])  # [셀, 문장]
        den = np.stack([Z[(n, c.format("den"))][idx].sum(0) for c in cells])
        if w is None:
            return num.sum(1), den.sum(1)
        return w @ num.T, w @ den.T  # [B, 구간]

    na_, da_ = grp(ia)
    nb_, db_ = grp(ib)
    wa, wb = da_ / da_.sum(-1, keepdims=True), db_ / db_.sum(-1, keepdims=True)
    aa, ab = np.where(da_ > 0, na_ / np.maximum(da_, 1), 0), np.where(db_ > 0, nb_ / np.maximum(db_, 1), 0)
    gap = (na_.sum(-1) / da_.sum(-1)) - (nb_.sum(-1) / db_.sum(-1))
    comp = ((wa - wb) * (aa + ab) / 2).sum(-1)
    within = ((aa - ab) * (wa + wb) / 2).sum(-1)
    return gap, comp, within


G_EN, G_KO = [("ja", "en"), ("zh", "en")], [("ja", "ko"), ("zh", "ko")]
L.append(f"\n## 5. 출력 자유도(target 엔트로피)로 설명되나 (P5) — 원문을 맞춘 비교: 영어 출력 {{ja,zh}}→en vs 한국어 출력 {{ja,zh}}→ko\n")
L.append(f"Kitagawa 분해: 차이 = 구성 효과(target 이 불확실한 위치의 비율이 달라서) + 구간 안 효과(같은 불확실도에서도 draft 가 덜 맞혀서). π = 구성/차이. "
         f"P5 판정: π 의 95% 하한 ≥ 0.5 → 지지. 구간 경계(nats) {HB[1:-1]}. 엔트로피는 출력 언어가 만드는 매개변수라 '원인'이 아니라 '격차가 어디에 있나'를 보여 준다.\n")
L.append("| draft | 단위 · 셀 | 차이 ā(en출력) − ā(ko출력) | 구성 효과 | 셀 안 효과 | π | P5 |\n|---|---|---|---|---|---|---|")
P5 = {}
for n in drafts:
    for name, kw in (("토큰 · 엔트로피 구간", {}), ("조각 합침 · 엔트로피 구간", dict(unit=True)), ("토큰 · 엔트로피 × 유형", dict(by_type=True))):
        g, c_, wi = kitagawa(n, G_EN, G_KO, **kw)
        gb_, cb_, wb_ = kitagawa(n, G_EN, G_KO, W, **kw)
        pi_b = cb_ / gb_
        lo, hi = ci(pi_b)
        verdict = "지지" if lo >= 0.5 else "기각" if hi < 0.5 else "판정 불가"
        if not kw:
            P5[n] = verdict
        L.append(f"| {n} | {name} | {fci(g, *ci(gb_), sign=True)} | {fci(c_, *ci(cb_), sign=True)} | {fci(wi, *ci(wb_), sign=True)} | "
                 f"{c_ / g:.2f} [{lo:.2f}, {hi:.2f}] | {verdict} |")
L.append("\n토큰 유형 = 어두(공백으로 시작) / 어중(공백 없이 이어지는 글자 — 한국어 조사·어미·음절 이음) / 기호. 한국어 내용 토큰은 어중이 많아서"
         " 엔트로피만 맞추면 '셀 안' 항에 토큰 유형 차이가 섞인다 (독립 검증에서 지적). 셀 안 항의 부호는 유형까지 맞춘 줄로 판단한다.")
# 탐색적: 같은 영어 출력인데 원문만 다른 ko→en vs zh→en — 엔트로피가 비슷한데 차이가 나면 draft 의 원문 이해(정렬) 문제
L.append("\n탐색적 (원문만 다름, 출력 = 영어): ko→en − zh→en")
for n in drafts:
    g, c_, wi = kitagawa(n, [("ko", "en")], [("zh", "en")])
    gb_, cb_, wb_ = kitagawa(n, [("ko", "en")], [("zh", "en")], W)
    L.append(f"- {n}: 차이 {fci(g, *ci(gb_), sign=True)} = 구성 {fci(c_, *ci(cb_), sign=True)} + 구간 안 {fci(wi, *ci(wb_), sign=True)}")

L.append(f"\n**구간별 점유율 / 일치율 ({n0}, 출력 언어별 모든 방향)**\n")
L.append("| 출력 | " + " | ".join(HBN) + " |\n|---|" + "---|" * len(HBN))
HBT = {}
for t in OUT_ORDER:
    ds = [DI[d] for d in DIRS if d[1] == t]
    dens = np.array([Z[(n0, f"hb_den_{b}")][ds].sum() for b in range(len(HBN))])
    nums = np.array([Z[(n0, f"hb_num_{b}")][ds].sum() for b in range(len(HBN))])
    HBT[t] = (dens / dens.sum(), np.where(dens > 0, nums / np.maximum(dens, 1), np.nan), dens)
    L.append(f"| {t} | " + " | ".join(f"{w_:.0%} / {a_:.2f}" if d_ >= 50 else f"{w_:.0%} / ·" for w_, a_, d_ in zip(*HBT[t])) + " |")


# ======================= 6. 토큰 수준 로지스틱: 방향 효과가 target 쪽 공변량으로 얼마나 설명되나 =======================
def logit_fit(X, y, iters=25):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - p)
        Hm = X.T @ (X * (p * (1 - p))[:, None]) + 1e-3 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


rows_y, rows_d, rows_x = [], [], []
for (s, t, sid), r in SC[n0].items():
    m, H, py = r["m"], r["H"], r["py"]
    for j in range(len(m) - 1):
        c, fr, cy, _, _ = ANN[(s, t, sid)][j]
        rows_y.append(m[j])
        rows_d.append(DI[(s, t)])
        hb = next(b for b in range(len(HB) - 1) if HB[b] <= H[j] < HB[b + 1])
        pb = min(int(py[j] * 10), 9)
        rows_x.append((hb, pb, CATS.index(f"frag_{fr}" if fr else c), int(cy)))
y_ = np.array(rows_y, float)
Dm = np.eye(len(DIRS))[rows_d]
xs = np.array(rows_x)
cov = np.concatenate([np.eye(len(HB) - 1)[xs[:, 0]][:, 1:], np.eye(10)[xs[:, 1]][:, 1:], np.eye(len(CATS))[xs[:, 2]][:, 1:], xs[:, 3:4]], 1)
b0 = logit_fit(Dm, y_)
b1 = logit_fit(np.concatenate([Dm, cov], 1), y_)[: len(DIRS)]
sd0, sd1 = b0.std(), (b1 - b1.mean()).std()
L.append(f"\n## 6. 방향 효과가 target 쪽 성질로 얼마나 설명되나 — 토큰 로지스틱 ({n0}, 탐색적)\n")
L.append(f"logit P(일치) = δ_방향 [+ target 엔트로피 구간 + target 확률 p(y) 구간 + 토큰 범주 + 복사 여부]. 공변량은 draft 와 무관한 것만.\n")
L.append(f"- 방향 효과 δ 의 표준편차(logit): 공변량 없이 {sd0:.3f} → 넣으면 {sd1:.3f} (설명된 비율 {1 - sd1 ** 2 / sd0 ** 2:.0%})")
L.append("- 방향 효과 (logit, 평균 0): " + ", ".join(f"{dname(DIRS[i])} {b0[i] - b0.mean():+.2f}→{b1[i] - b1.mean():+.2f}" for i in np.argsort(b0)))

# ======================= 7. 정렬 지표 (서술용) =======================
L.append(f"\n## 7. 분포 정렬 — 방향별 (greedy 궤적 위 평균, 서술용)\n")
L.append("1−TV(T) = 그 문맥에서 순수 temperature 샘플링 SD 의 토큰당 수락 확률. greedy 궤적에서 쟀으므로 샘플링 SD 의 α 그 자체는 아니다.\n")
cols = [("tv", "1−TV(T=1)", True), ("tv07", "1−TV(T=.7)", True), ("tvq", "1−TV(Qwen 권장)", True), ("kl", "KL(p‖q)", False), ("ce", "draft CE", False)]
L.append("| 방향 | target H | " + " | ".join(f"{h} · {n}" for n in drafts for _, h, _ in cols) + f" | top-5 적중 · {n0} |\n|---|---|" + "---|" * (len(cols) * len(drafts) + 1))
for i in order:
    den = Z[(n0, "c_den")][i].sum()
    cells = [f"{Z[(n0, 'sum_H')][i].sum() / den:.2f}"]
    for n in drafts:
        dn = Z[(n, "c_den")][i].sum()
        for f, _, inv in cols:
            v = Z[(n, f"sum_{f}")][i].sum() / dn
            cells.append(f"{1 - v:.3f}" if inv else f"{v:.2f}")
    cells.append(f"{Z[(n0, 'top5_num')][i].sum() / den:.3f}")
    L.append(f"| {dname(DIRS[i])} | " + " | ".join(cells) + " |")


# ======================= 8. 속도 예측, 길이, 손익분기 c (하드웨어 쪽) =======================
def tau_inf(n, d, g):
    """방향의 m 을 문장 순서대로 이어 붙여 시뮬레이션 → 문장 끝(잘린 마지막 라운드) 효과 없는 τ"""
    m = [x for sid in SIDS for x in SC[n][(d[0], d[1], sid)]["m"]]
    A, R_, K = simulate(m, g)
    return len(m) / K


def speed(n, i, g, c=None, w=None):
    c = C[n] if c is None else c
    num = Z[(n, "ntok")][i]
    den = Z[(n, f"K{g}")][i] * (g * c + R_K[g + 1])
    return (num.sum() / den.sum()) if w is None else (w @ num) / (w @ den)


L.append(f"\n## 8. 이 노트북에서의 예측 speedup (시뮬레이션 라운드 × 비용 모델) 과 문장당 시간\n")
L.append("S = Σ(출력−1)·T_t(1) / Σ라운드·(γ·T_d + T_t(γ+1)) (방향 작업량 기준). τ = 실제 라운드당 토큰, τ∞ = 문장 끝 효과를 뺀 값. "
         "ms/문장 = decode 만 (prefill 제외).\n")
gs_ok = [g for g in GAMMAS if g + 1 in R_K]
L.append("| 방향 | τ(4) / τ∞(4) | " + " | ".join(f"최고 γ / S / 역전 γ · {n}" for n in drafts if n in C) + f" | vanilla ms/문장 | SD ms/문장 (최고 γ, {n0}) |\n|---|---|" + "---|" * (len([n for n in drafts if n in C]) + 2))
BEST = {}
for i in order:
    d = DIRS[i]
    t4 = Z[(n0, "ntok")][i].sum() / Z[(n0, "K4")][i].sum()
    cells = [f"{t4:.2f} / {tau_inf(n0, d, 4):.2f}"]
    for n in drafts:
        if n not in C:
            continue
        sp = {g: speed(n, i, g) for g in gs_ok}
        best = max(sp, key=sp.get)
        rev = next((g for g in gs_ok if g > best and sp[g] < 1), None)
        BEST[(n, d)] = (best, sp[best], rev, sp)
        cells.append(f"{best} / {sp[best]:.2f}× / {rev if rev else f'>{gs_ok[-1]}'}")
    van = Z[(n0, "ntok")][i].mean() * Tt[1] * 1e3
    g_ = BEST[(n0, d)][0]
    sd_ms = (Z[(n0, f"K{g_}")][i] * (g_ * C[n0] + R_K[g_ + 1])).mean() * Tt[1] * 1e3
    cells += [f"{van:.0f}", f"{sd_ms:.0f}"]
    L.append(f"| {dname(d)} | " + " | ".join(cells) + " |")

# 손익분기 c: 이보다 draft 가 비싸면 어떤 γ 로도 SD 가 손해 (검증 비용 곡선은 실측 r(k))
L.append(f"\n**손익분기 draft 비용비 c_be** (= max_γ (τ(γ) − r(γ+1))/γ, 이보다 c 가 크면 어떤 γ 로도 S < 1). 비용 모델 외삽이며 실측 c 는 "
         + ", ".join(f"{n} {C[n]:.2f}" for n in drafts if n in C) + "\n")
L.append("| 방향 | " + " | ".join(f"c_be · {n}" for n in drafts) + " |\n|---|" + "---|" * len(drafts))
CBE, CBE_G = {}, []
for i in order:
    d = DIRS[i]
    vals = []
    for n in drafts:
        tau = {g: Z[(n, "ntok")][i].sum() / Z[(n, f"K{g}")][i].sum() for g in gs_ok}
        gb = max(gs_ok, key=lambda g: (tau[g] - R_K[g + 1]) / g)
        CBE[(n, d)] = (tau[gb] - R_K[gb + 1]) / gb
        CBE_G.append(gb)
        vals.append(f"{CBE[(n, d)]:.3f}")
    L.append(f"| {dname(d)} | " + " | ".join(vals) + " |")
L.append("\n" + ("모든 방향·draft 에서 γ=1 에서 최대 → c_be = τ(1) − r(2): draft 가 비싸지면 한 토큰짜리 추측이 마지막으로 남는다"
                  if set(CBE_G) == {1} else f"최대가 되는 γ: {sorted(Counter(CBE_G).items())}"))
CS = [0.05, 0.1, 0.21, 0.33, 0.45, 0.6]
L.append(f"\n**c 만 바꾼 외삽 ({n0} 의 α·τ 고정)** — 같은 γ 의 속도 비는 c 와 무관(식에서 바로 나옴, 검산용), 각자 최고 γ 에서의 비는 c 가 커질수록 늘지 않고 계단처럼 줄어든다 (최고 γ 가 바뀔 때만)\n")
L.append("| c | S(en출력)/S(ko출력) γ=4 | 각자 최고 γ 의 비 | ko 출력 최고 S | en 출력 최고 S |\n|---|---|---|---|---|")
for c in CS:
    def grp_S(t, g):
        idx = [DI[d] for d in DIRS if d[1] == t]
        num = sum(Z[(n0, "ntok")][i].sum() for i in idx)
        den = sum((Z[(n0, f"K{g}")][i] * (g * c + R_K[g + 1])).sum() for i in idx)
        return num / den
    se4, sk4 = grp_S("en", 4), grp_S("ko", 4)
    sbe, sbk = max(grp_S("en", g) for g in gs_ok), max(grp_S("ko", g) for g in gs_ok)
    L.append(f"| {c:.2f} | {se4 / sk4:.3f} | {sbe / sbk:.3f} | {sbk:.2f} | {sbe:.2f} |")

# ======================= 9. online 검증 =======================
ON_ROWS = []
for p in sorted(D.glob("online*.csv")):
    tag = p.stem[len("online"):]
    outs = {}
    po = D / f"online_out{tag}.jsonl"
    if po.exists():
        for o in map(json.loads, open(po, encoding="utf-8")):
            outs[(o["src"], o["tgt"], o["sid"], int(o.get("attempt", 0)), o["draft"], int(o["gamma"]), float(o.get("temp", 0)))] = o
    for r in csv.DictReader(open(p)):
        r["tag"] = tag
        key = (r["src"], r["tgt"], int(r["sid"]), int(r.get("attempt", 0)), r["draft"], int(r["gamma"]), float(r.get("temp", 0)))
        r["o"] = outs.get(key)
        ON_ROWS.append(r)
REPLAY = {}
for p in sorted(D.glob("replay*.jsonl.gz")):
    for r in map(json.loads, gzip.open(p, "rt")):
        REPLAY[(r["src"], r["tgt"], r["sid"], int(r.get("attempt", 0)), r["draft"], int(r["gamma"]), float(r["temp"]))] = r
ON = defaultdict(list)
if ON_ROWS:
    blk = defaultdict(dict)
    for r in ON_ROWS:
        blk[(r["tag"], r["src"], r["tgt"], int(r["sid"]), int(r.get("attempt", 0)))][(r["draft"], int(r["gamma"]), float(r.get("temp", 0)))] = r
    use, bad = {}, []
    for k, b in blk.items():  # 드리프트 검사: 앞뒤 vanilla 가 10% 넘게 다르면 버린다. 재측정한 블록이 있으면 통과한 쪽을 쓴다
        v0, v1 = float(b[("vanilla", 0, 0.0)]["decode_tps"]), float(b[("vanilla_after", 0, 0.0)]["decode_tps"])
        kk = k[:4]
        if abs(v1 / v0 - 1) > 0.10:
            bad.append(k)
        elif kk not in use or k[4] > use[kk][0]:
            use[kk] = (k[4], b, (v0 + v1) / 2)
    for (tag, s, t, sid), (att, b, vref) in use.items():
        van = b[("vanilla", 0, 0.0)]
        gen_y = gen[(s, t, sid)]["out"]
        for (n, g, temp), r in b.items():
            if n.startswith("vanilla") or n not in SC and n not in C:
                continue
            A, R_ = int(r["accepted"]), int(r["rejects"])
            o = r["o"] or {}
            out_tokens = o.get("out", [])
            rec = dict(tag=tag, dir=(s, t), sid=sid, n=n, g=g, temp=temp, a_on=A / (A + R_) if A + R_ else 1.0, A=A, R=R_,
                       sp=float(r["decode_tps"]) / vref, e2e=float(van["total"]) / float(r["total"]),
                       tau_on=(int(r["n_new"]) - 1) / int(r["rounds"]), same_gen=out_tokens == gen_y, log=o.get("log"))
            sc = SC.get(n, {}).get((s, t, sid))
            if sc is not None and temp == 0:
                As, Rs, Ks = simulate(sc["m"], g)
                rec.update(a_off=As / (As + Rs) if As + Rs else 1.0, pred_off=(len(sc["m"]) * Tt[1]) / (Ks * (g * C[n] * Tt[1] + Tt[g + 1])) if n in C else np.nan)
            if n in C and g + 1 in Tt:
                rec["pred_on"] = rec["tau_on"] * Tt[1] / (g * C[n] * Tt[1] + Tt[g + 1])
            rp = REPLAY.get((s, t, sid, att, n, g, temp))
            if rp is not None and temp == 0:
                sim = simulate_log(rp["m"][1:], g)
                rec["replay_ok"] = [tuple(x) for x in sim] == [(a_, bool(r_)) for _, a_, r_ in rp["log"]]
                rec["replay_tok"] = min(rp["t_ok"]) == 1
            if rp is not None and temp > 0 and rp.get("log"):
                tested = []
                for pos, a_, r_ in rp["log"]:
                    tested += list(range(pos, pos + a_ + int(r_)))
                rec["beta_sum"] = sum(rp["beta"][j] for j in tested if j < len(rp["beta"]))
                rec["n_tested"] = len(tested)
            ON[(n, g, temp)].append(rec)
    L.append(f"\n## 9. online 검증 — 실제 SD (sd.py 수정판, batch 1)\n")
    L.append(f"블록 {len(blk)}개 중 드리프트(앞뒤 vanilla 10% 초과) {len(bad)}개 → 재측정 통과분으로 대체. speedup = decode tok/s ÷ 블록 앞뒤 vanilla 평균, E2E = 전체 시간(prefill 포함) 비.\n")
    L.append("| 설정 | run | 배치 생성과 같은 궤적 | online α (문장 평균) | offline α (같은 문장 평균) | 차이 [90% CI] | 라운드 재현 일치 | 실측 S | 예측 S (online τ) | 예측 S (offline τ) | E2E |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for (n, g, temp), xs in sorted(ON.items()):
        if temp > 0:
            continue
        offs = [x for x in xs if "a_off" in x]
        aon = np.mean([x["a_on"] for x in offs]) if offs else float("nan")  # 차이 열과 같은 추정량(문장 평균)으로
        d = np.array([x["a_on"] - x["a_off"] for x in offs]) if offs else np.array([np.nan])
        bd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)] if offs else [np.nan]
        rp = [x["replay_ok"] for x in xs if "replay_ok" in x]
        L.append(f"| {n} γ={g} | {len(xs)} | {np.mean([x['same_gen'] for x in xs]):.0%} | {aon:.3f} | "
                 f"{np.mean([x['a_off'] for x in offs]) if offs else float('nan'):.3f} | {d.mean():+.3f} [{np.percentile(bd, 5):+.3f}, {np.percentile(bd, 95):+.3f}] | "
                 f"{(f'{np.mean(rp):.0%} ({len(rp)})') if rp else '·'} | {np.mean([x['sp'] for x in xs]):.2f} | {np.nanmean([x.get('pred_on', np.nan) for x in xs]):.2f} | "
                 f"{np.nanmean([x.get('pred_off', np.nan) for x in xs]):.2f} | {np.mean([x['e2e'] for x in xs]):.2f} |")
    for (n, g, temp), xs in sorted(ON.items()):
        offs = [x for x in xs if "a_off" in x]
        if temp > 0 or not offs:
            continue
        d = np.array([x["a_on"] - x["a_off"] for x in offs])
        bd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bd, 5), np.percentile(bd, 95)
        L.append(f"- P7 ({n} γ={g}): online α − offline α (같은 문장) {d.mean():+.3f}, 90% CI [{lo:+.3f}, {hi:+.3f}] → **{'지지' if -0.02 < lo and hi < 0.02 else '판정 불가'}** "
                 f"(문장별 |차이| 평균 {np.abs(d).mean():.3f}, 차이 0 인 문장 {np.mean(d == 0):.0%})")
    L.append("\n**독립 c 민감도** — 같은 run 에서 뽑은 c 대신 W02 의 c 를 넣은 예측 (offline τ):")
    for (n, g, temp), xs in sorted(ON.items()):
        if temp > 0 or n not in C_W02:
            continue
        def pw(x):
            sc_ = SC[n][(x["dir"][0], x["dir"][1], x["sid"])]
            As_, Rs_, Ks_ = simulate(sc_["m"], g)
            return (len(sc_["m"]) * Tt[1]) / (Ks_ * (g * C_W02[n] * Tt[1] + Tt[g + 1]))
        by = {t: (np.mean([x["sp"] for x in xs if x["dir"][1] == t]), np.mean([pw(x) for x in xs if x["dir"][1] == t])) for t in OUT_ORDER if any(x["dir"][1] == t for x in xs)}
        L.append(f"- {n} γ={g} (W02 c {C_W02[n]:.3f}): 전체 실측 {np.mean([x['sp'] for x in xs]):.3f} vs 예측 {np.mean([pw(x) for x in xs]):.3f} · "
                 + ", ".join(f"{t} {a:.2f} vs {b:.2f}" for t, (a, b) in by.items()))
    L.append("\n**출력 언어별 실측 speedup (online, 문장 평균) vs 같은 문장의 예측 (offline τ)**\n")
    L.append("| 설정 | " + " | ".join(OUT_ORDER) + " |\n|---|" + "---|" * len(OUT_ORDER))
    for (n, g, temp), xs in sorted(ON.items()):
        cells = []
        for t in OUT_ORDER:
            ys = [x for x in xs if x["dir"][1] == t]
            if not ys:
                cells.append("·")
                continue
            pred = np.array([x.get("pred_off", np.nan) for x in ys], dtype=float)
            pr = np.nanmean(pred) if np.isfinite(pred).any() else np.nan
            cells.append(f"{np.mean([x['sp'] for x in ys]):.2f}" + (f" (예측 {pr:.2f})" if not np.isnan(pr) else ""))
        L.append(f"| {n} γ={g}{f' T={temp}' if temp else ''} | " + " | ".join(cells) + " |")
    for (n, g, temp), xs in sorted(ON.items()):
        if temp == 0:
            continue
        L.append(f"\n**샘플링 SD (T={temp}, {n}, γ={g})** — online α = ΣA/Σ(A+R), 검사된 위치의 Σβ/Σ검사 = teacher forcing 으로 계산한 기대 수락률\n")
        L.append("| 출력 | online α | Σβ/Σ검사 | 실측 S | greedy α (같은 설정 γ=4) |\n|---|---|---|---|---|")
        for t in OUT_ORDER:
            ys = [x for x in xs if x["dir"][1] == t]
            if not ys:
                continue
            ag = [x for x in ON.get((n, g, 0.0), []) if x["dir"][1] == t]
            bsum = [x for x in ys if "beta_sum" in x]
            L.append(f"| {t} | {sum(x['A'] for x in ys) / sum(x['A'] + x['R'] for x in ys):.3f} | "
                     f"{(sum(x['beta_sum'] for x in bsum) / sum(x['n_tested'] for x in bsum)) if bsum else float('nan'):.3f} | {np.mean([x['sp'] for x in ys]):.2f} | "
                     f"{(sum(x['A'] for x in ag) / sum(x['A'] + x['R'] for x in ag)) if ag else float('nan'):.3f} |")

# ======================= 10. 대조 실험 =======================
def ctrl(dirpath, data):
    """대조 실험 폴더 → {(draft, 방향): (문장별 내용 일치 분자, 분모, 놀람, 군집 id)}"""
    if not (dirpath / "gen.jsonl").exists():
        return None
    src_rows = {x["sid"]: x for x in map(json.loads, open(data, encoding="utf-8"))}
    res = {}
    for f in sorted(dirpath.glob("score_*.jsonl.gz")):
        n = f.name[6:-9]
        for r in map(json.loads, gzip.open(f, "rt")):
            m, py = r["m"], r["py"]
            res.setdefault((n, (r["src"], r["tgt"])), []).append(
                (r["sid"], sum(m[:-1]), len(m) - 1, sum(-np.log(max(x, 1e-12)) for x in py[:-1]), src_rows[r["sid"]].get("url", r["sid"])))
    return res


def ctrl_ratio(rows, B_=2000, seed=1):
    """문장 묶음 → (Σ일치/Σ위치, 95% CI: 군집 부트스트랩)"""
    num, den = np.array([x[1] for x in rows], float), np.array([x[2] for x in rows], float)
    cl_ = sorted({x[4] for x in rows})
    ci_ = np.array([cl_.index(x[4]) for x in rows])
    rg = np.random.default_rng(seed)
    bs = []
    for _ in range(B_):
        w = np.bincount(rg.integers(0, len(cl_), len(cl_)), minlength=len(cl_))[ci_]
        bs.append((w * num).sum() / (w * den).sum())
    return num.sum() / den.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


WMT = ctrl(HERE / "results/lang_wmt20", HERE / "data/wmt20.jsonl")
if WMT:
    L.append("\n## 10. 대조 실험\n")
    L.append("**(a) 원문이 영어가 아닌 뉴스 (WMT20 ja-en / zh-en 시험셋 원문, 각 100문장)** — FLORES 는 모든 문장이 영어 원문의 번역이라, "
             "X→en 은 '원래 영어로 쓰인 내용으로 돌아가기'일 수 있다. 일본어·중국어 원작에서도 영어 출력 > 한국어 출력이면 영어 원문 효과는 배제\n")
    L.append("| draft | 원문 | ā → en | ā → ko | 차이 (ko − en) | target 놀람 → en / → ko (nats/문장) | FLORES 의 같은 차이 |\n|---|---|---|---|---|---|---|")
    for n in drafts:
        for s_ in ("ja", "zh"):
            e, k_ = WMT.get((n, (s_, "en"))), WMT.get((n, (s_, "ko")))
            if not e or not k_:
                continue
            ae, ak = ctrl_ratio(e), ctrl_ratio(k_)
            # 같은 문장끼리 짝 차이
            se, sk = {x[0]: x for x in e}, {x[0]: x for x in k_}
            both = sorted(set(se) & set(sk))
            diff = ak[0] - ae[0]
            cl_ = sorted({se[i][4] for i in both})
            cidx = np.array([cl_.index(se[i][4]) for i in both])
            rg = np.random.default_rng(2)
            bd = []
            for _ in range(2000):
                w = np.bincount(rg.integers(0, len(cl_), len(cl_)), minlength=len(cl_))[cidx]
                ne_, de_ = (w * np.array([se[i][1] for i in both])).sum(), (w * np.array([se[i][2] for i in both])).sum()
                nk_, dk_ = (w * np.array([sk[i][1] for i in both])).sum(), (w * np.array([sk[i][2] for i in both])).sum()
                bd.append(nk_ / dk_ - ne_ / de_)
            fl = A_PT[n][DI[(s_, "ko")]] - A_PT[n][DI[(s_, "en")]]
            L.append(f"| {n} | {s_} | {fci(*ae)} | {fci(*ak)} | {fci(diff, np.percentile(bd, 2.5), np.percentile(bd, 97.5), sign=True)} | "
                     f"{np.mean([x[3] for x in e]):.2f} / {np.mean([x[3] for x in k_]):.2f} | {fl:+.3f} |")

KOI = ctrl(HERE / "results/lang_koinstr", HERE / "data/flores.jsonl")
if KOI:
    L.append("\n**(b) 한국어 지시문** — 영어 지시문에서는 한국어를 쓰기 전 문맥에 한글이 하나도 없다. 같은 200문장을 한국어 지시문으로 다시 생성\n")
    L.append("| draft | 방향 | ā 영어 지시문 | ā 한국어 지시문 | 차이 |\n|---|---|---|---|---|")
    for n in drafts:
        dd = {}
        for d in [("en", "ko"), ("ko", "en")]:
            rows = KOI.get((n, d))
            if not rows:
                continue
            a_ko = ctrl_ratio(rows)[0]
            a_en = A_PT[n][DI[d]]
            dd[d] = a_ko - a_en
            L.append(f"| {n} | {dname(d)} | {a_en:.3f} | {a_ko:.3f} | {a_ko - a_en:+.3f} |")
        if len(dd) == 2:
            L.append(f"| {n} | 상호작용 (en→ko 변화 − ko→en 변화) | | | {dd[('en', 'ko')] - dd[('ko', 'en')]:+.3f} |")
    kg = [json.loads(l) for l in open(HERE / "results/lang_koinstr/gen.jsonl", encoding="utf-8")]
    fr = [r["text"].rstrip(" .!?\"'”’」』)").rstrip().endswith(("니다", "니까")) for r in kg if r["tgt"] == "ko"]
    L.append(f"\n한국어 지시문에서 en→ko 출력의 합니다체 비율 {np.mean(fr):.0%} (영어 지시문 {np.mean([FLAG[('en', 'ko', s_)]['reg'] == 'formal' for s_ in SIDS]):.0%})")

cov = D / "coverage.md"
if cov.exists():
    L.append("\n## 부록. 다른 계열 draft 후보의 어휘 커버리지 (모델 없이 토크나이저만) → `coverage.md`")

(D / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ======================= 그림 =======================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz 기본 팔레트 1~5, 순서 고정
COL = dict(zip(OUT_ORDER, SERIES))
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False})


def sname(n):
    return n.replace("Qwen3-", "").replace("-Instruct-2507", " (2507)")


def title(ax, s):
    ax.set_title(s, loc="left", color=INK, fontsize=11)


def fig_alpha(ax, ds_order=None):
    """방향별 ā — 점 = draft (작은 → 큰), 색 = 출력 언어"""
    idx = list(reversed(order)) if ds_order is None else ds_order
    mk = ["o", "s", "D"]
    for row, i in enumerate(idx):
        d = DIRS[i]
        xs_ = [A_PT[n][i] for n in drafts]
        ax.plot(xs_, [row] * len(xs_), color=AXIS, lw=1.5, zorder=1)
        for j, n in enumerate(drafts):
            lo, hi = ci(A_BS[n][:, i])
            ax.plot([lo, hi], [row, row], color=COL[d[1]], lw=1, alpha=0.5, zorder=2)
            ax.plot(A_PT[n][i], row, mk[j % 3], color=COL[d[1]], ms=7, mec=SURF, mew=1.5, zorder=3,
                    alpha=1.0 if j == 0 else 0.75)
    ax.set_yticks(range(len(idx)), [dname(DIRS[i]) for i in idx])
    ax.set_xlabel("일치율 = draft 1등 토큰이 target 토큰과 같은 비율 (EOS 제외)")
    for t in OUT_ORDER:
        ax.plot([], [], "o", color=COL[t], label=f"{LANG_NAME[t]} 출력")
    for j, n in enumerate(drafts):
        ax.plot([], [], mk[j % 3], color=MUTED, label=n)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(axis="y", visible=False)


# ---- 그림 1: W03 로그용 (H1 핵심 두 장) ----
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
title(axs[0], "(a) 같은 200문장, 14방향의 일치율 (가는 선 = 95% CI, 기사 단위)")
fig_alpha(axs[0])
ax = axs[1]
title(ax, "(b) 원문을 맞춘 {ja,zh}→en vs {ja,zh}→ko 격차 분해 (셀 = 엔트로피 × 토큰 유형)")
labels, comp, within, gaps, errs = [], [], [], [], []
for n in drafts:
    g, c_, wi = kitagawa(n, G_EN, G_KO, by_type=True)
    gb_, cb_, wb_ = kitagawa(n, G_EN, G_KO, W, by_type=True)
    labels.append(n)
    comp.append(c_)
    within.append(wi)
    gaps.append(g)
    errs.append(ci(gb_))
yy = np.arange(len(labels))
ax.barh(yy + 0.18, comp, height=0.34, color=SERIES[0], label="target 이 불확실한 위치가 더 많은 쪽 (구성 효과)")
ax.barh(yy - 0.18, within, height=0.34, color=SERIES[1], label="같은 불확실도·토큰 유형 안의 차이")
for k_, g in enumerate(gaps):
    ax.plot([g], [k_], "D", color=INK, ms=7, zorder=4)
    ax.annotate(f"전체 차이 {g:+.3f}", (g, k_), xytext=(6, 10), textcoords="offset points", fontsize=8, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(yy, labels)
ax.set_xlabel("일치율(영어 출력) - 일치율(한국어 출력)   · 음수 = 같은 불확실도에서는 한국어 출력이 더 잘 맞음")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", visible=False)
fig.suptitle("EXP-003 · 내용을 통제한 언어별 acceptance — FLORES-200, Qwen3-8B target, Apple M5 Pro", x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(D / "fig_h1.png", dpi=150)

# ---- 그림 2: 요약 4패널 ----
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
ax = axs[0, 0]
title(ax, f"(a) 일치율 행렬 · {n0} (행 = 원문, 열 = 출력)")
langs = ["en", "de", "zh", "ja", "ko"]
mat = np.array([[A_PT[n0][DI[(s, t)]] if (s, t) in DI else np.nan for t in langs] for s in langs])
im = ax.imshow(mat, cmap=BLUE, vmin=np.nanmin(mat) - 0.02, vmax=np.nanmax(mat) + 0.01)
ax.grid(False)
for a_ in range(len(langs)):
    for b_ in range(len(langs)):
        if not np.isnan(mat[a_, b_]):
            ax.text(b_, a_, f"{mat[a_, b_]:.2f}", ha="center", va="center", fontsize=10, color=SURF if mat[a_, b_] > np.nanmean(mat) + 0.01 else INK)
ax.set_xticks(range(len(langs)), [f"→{l}" for l in langs])
ax.set_yticks(range(len(langs)), [f"{l}→" for l in langs])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)

ax = axs[0, 1]
title(ax, f"(b) target 엔트로피 구간 안 일치율 · {n0} (토큰 50개 미만 구간 생략)")
for t in OUT_ORDER:
    w_, a_, d_ = HBT[t]
    ax.plot(np.arange(len(HBN))[d_ >= 50], a_[d_ >= 50], marker="o", color=COL[t], label=f"{LANG_NAME[t]} 출력 (낮은 구간 <.05 점유 {w_[0]:.0%})")
ax.set_xticks(range(len(HBN)), HBN)
ax.set_xlabel("target 엔트로피 H(p) 구간 (nats) — 오른쪽일수록 target 자신도 확신 없음")
ax.set_ylabel("구간 안 일치율")
ax.legend(loc="lower left", fontsize=8)

ax = axs[1, 0]
ON_SIDS = sorted({x["sid"] for xs in ON.values() for x in xs}) if ON else SIDS
sm = np.isin(np.array(SIDS), ON_SIDS)  # online 으로 잰 문장만 (점과 같은 문장으로 예측)
title(ax, f"(c) speedup — 선 = 예측(비용 모델), 점 = 실측, 같은 {sm.sum()}문장 · {n0}")
ax.axhline(1, color=INK2, lw=1, zorder=1)
ax.text(gs_ok[-1], 1.01, "vanilla", color=INK2, va="bottom", ha="right", fontsize=8)
for t in OUT_ORDER:
    idx = [DI[d] for d in DIRS if d[1] == t]
    y = []
    for g in gs_ok:  # 실측 점과 맞추려고 문장별 S 의 평균 (실측도 문장별 speedup 의 평균)
        per = [Z[(n0, "ntok")][i, sm] / (Z[(n0, f"K{g}")][i, sm] * (g * C[n0] + R_K[g + 1])) for i in idx]
        y.append(np.mean(np.concatenate(per)))
    ax.plot(gs_ok, y, color=COL[t], label=f"{LANG_NAME[t]} 출력", lw=1.8)
    for g in (4, 8):
        pts = [x["sp"] for x in ON.get((n0, g, 0.0), []) if x["dir"][1] == t]
        if pts:
            ax.plot([g], [np.mean(pts)], "o", color=COL[t], ms=8, mec=SURF, mew=2, zorder=4)
ax.set_xlabel("γ (draft 길이)")
ax.set_ylabel("decode speedup (× vanilla)")
ax.legend(loc="upper right", fontsize=8)

ax = axs[1, 1]
title(ax, "(d) 손익분기 c_be — draft 비용비가 이보다 크면 SD 가 손해 (외삽)")
idx = list(reversed(order))
for row, i in enumerate(idx):
    d = DIRS[i]
    v = CBE[(n0, d)]
    ax.plot([0, v], [row, row], color=COL[d[1]], lw=3, solid_capstyle="round")
    ax.text(v + 0.005, row, f"{v:.2f}", va="center", fontsize=8, color=INK2)
for j, n in enumerate(drafts):
    if n in C:
        ax.axvline(C[n], color=INK2, lw=[1, 1.8, 2.6][j % 3], label=f"{n} 실측 c = {C[n]:.2f}")
ax.legend(loc="lower right", fontsize=8)
ax.set_yticks(range(len(idx)), [dname(DIRS[i]) for i in idx])
ax.set_xlabel(f"c = T_draft(1) / T_target(1)  ({n0} 의 τ 기준)")
ax.set_xlim(0, max(CBE.values()) * 1.35)
ax.grid(axis="y", visible=False)
fig.suptitle(f"EXP-003 · FLORES-200 {len(SIDS)}문장 × {len(DIRS)}방향, Qwen3-8B target, Apple M5 Pro (MPS)", x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(D / "summary.png", dpi=150)
print("saved", D / "fig_h1.png", D / "summary.png")

# ---- 그림 3: H2 (속도) — 예측 vs 실측, 손익분기 c ----
if ON:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    title(ax, "(a) 같은 문장의 예측 speedup (offline 라운드 × 비용 모델) vs 실측")
    marks = {4: "o", 8: "s", 1: "^", 2: "D"}
    lo_, hi_ = 10.0, 0.0
    for (n, g, temp), xs in sorted(ON.items()):
        if temp > 0:
            continue
        for t in OUT_ORDER:
            ys = [x for x in xs if x["dir"][1] == t and not np.isnan(x.get("pred_off", np.nan))]
            if not ys:
                continue
            px, py_ = np.mean([x["pred_off"] for x in ys]), np.mean([x["sp"] for x in ys])
            lo_, hi_ = min(lo_, px, py_), max(hi_, px, py_)
            ax.plot(px, py_, marks.get(g, "o"), color=COL[t], ms=8, mec=SURF, mew=1.5, alpha=1.0 if n == n0 else 0.6)
    ax.plot([lo_ - 0.05, hi_ + 0.05], [lo_ - 0.05, hi_ + 0.05], color=INK2, lw=1)
    ax.text(hi_, hi_ + 0.01, "예측 = 실측", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("예측 speedup (× vanilla)")
    ax.set_ylabel("실측 decode speedup (× vanilla)")
    for t in OUT_ORDER:
        ax.plot([], [], "o", color=COL[t], label=f"{LANG_NAME[t]} 출력")
    for (n, g, temp) in sorted(ON):
        if temp == 0:
            ax.plot([], [], marks.get(g, "o"), color=MUTED, label=f"{n} γ={g}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax = axs[1]
    title(ax, "(b) 손익분기 draft 비용비 c_be — 이보다 draft 가 비싸면 SD 가 손해")
    idx = list(reversed(order))
    yy = np.arange(len(idx))
    ds_ = [x for x in drafts if (x, DIRS[idx[0]]) in CBE]
    hgt = 0.8 / len(ds_)
    for j, n in enumerate(ds_):
        vals = [CBE[(n, DIRS[i])] for i in idx]
        ax.barh(yy + 0.4 - hgt * (j + 0.5), vals, height=hgt * 0.9, color=[COL[DIRS[i][1]] for i in idx], alpha=[1.0, 0.6, 0.3][j % 3],
                label=f"막대 {j + 1}: {sname(n)} 의 τ")
    for j, n in enumerate(drafts):
        if n in C:
            ax.axvline(C[n], color=INK2, lw=[1, 1.8, 2.6][j % 3], label=f"{sname(n)} 실측 c = {C[n]:.2f}")
    ax.set_yticks(yy, [dname(DIRS[i]) for i in idx])
    ax.set_xlabel("c = T_draft(1) / T_target(1)")
    ax.set_xlim(0, 1.25)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(0.70, 0.5), fontsize=8)
    fig.suptitle("EXP-003 · 이 노트북(M5 Pro, MPS, bf16, batch 1)에서의 속도 — 비용 모델 검증과 외삽", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(D / "fig_h2.png", dpi=150)
    print("saved", D / "fig_h2.png")
