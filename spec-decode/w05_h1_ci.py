# W03(lang_plot.py)와 같은 방식으로 내용 일치율 ā의 방향 간 차이 CI를 계산한다.
# ā = 출력 2번째 토큰 ~ EOS 직전 위치에서 m(일치) 합 / 위치 수. CI = 기사(URL) 단위 군집 부트스트랩 B=2000, default_rng(0).
import gzip, json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SD = Path(sys.argv[1])            # spec-decode 폴더
D = SD / "results" / "lang"
sents = {x["sid"]: x for x in map(json.loads, open(SD / "data" / "flores.jsonl", encoding="utf-8"))}
SIDS = sorted(sents)
SI = {s: i for i, s in enumerate(SIDS)}
CJK = ["en", "ko", "ja", "zh"]
DIRS = [(s, t) for s in CJK for t in CJK if s != t] + [("en", "de"), ("de", "en")]
DI = {d: i for i, d in enumerate(DIRS)}

urls = sorted({sents[s]["url"] for s in SIDS})
cl = np.array([urls.index(sents[s]["url"]) for s in SIDS])
rng = np.random.default_rng(0)
B = 2000
draws = rng.integers(0, len(urls), (B, len(urls)))
W = np.stack([np.bincount(d, minlength=len(urls))[cl] for d in draws]).astype(float)

res = {}
for n in ["Qwen3-0.6B", "Qwen3-1.7B"]:
    num = np.zeros((len(DIRS), len(SIDS)))
    den = np.zeros((len(DIRS), len(SIDS)))
    for r in map(json.loads, gzip.open(D / f"score_{n}.jsonl.gz", "rt", encoding="utf-8")):
        d = (r["src"], r["tgt"])
        if d not in DI:
            continue
        m = r["m"]
        L = len(m)
        num[DI[d], SI[r["sid"]]] += sum(m[j] for j in range(L - 1))
        den[DI[d], SI[r["sid"]]] += L - 1
    pt = num.sum(1) / den.sum(1)
    bs = (W @ num.T) / (W @ den.T)
    res[n] = (pt, bs)


def diff(n, a, b):
    pt, bs = res[n]
    i, j = DI[a], DI[b]
    x = pt[i] - pt[j]
    lo, hi = np.percentile(bs[:, i] - bs[:, j], [2.5, 97.5])
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def nm(d):
    return f"{d[0]}→{d[1]}"


print("## 점추정 (W03 표와 대조)")
for n in res:
    print(n, " ".join(f"{nm(d)} {res[n][0][DI[d]]:.3f}" for d in DIRS))

checks = [("Qwen3-0.6B", ("en", "ko"), ("en", "de"), "W03: -0.110 [-0.126, -0.095]"),
          ("Qwen3-1.7B", ("en", "ko"), ("en", "ja"), "W03: -0.007 [-0.019, +0.005]"),
          ("Qwen3-0.6B", ("en", "ko"), ("ko", "en"), "W03: -0.075 [-0.092, -0.058]"),
          ("Qwen3-0.6B", ("en", "ko"), ("ko", "zh"), "W03: en→ko − ko→zh = -0.009 [-0.029, +0.013]")]
print("\n## 재현 확인")
for n, a, b, ref in checks:
    print(f"{n} {nm(a)} − {nm(b)} = {diff(n, a, b)}   ({ref})")

new = [(("en", "ja"), ("en", "de")), (("en", "zh"), ("en", "de")), (("en", "ko"), ("en", "ja")),
       (("ja", "ko"), ("en", "ko")), (("ja", "ko"), ("zh", "ko")),
       (("ko", "ja"), ("en", "ja")), (("ko", "ja"), ("zh", "ja")),
       (("en", "zh"), ("ja", "zh")), (("en", "zh"), ("ko", "zh")),
       (("de", "en"), ("zh", "en")), (("zh", "en"), ("ja", "en")), (("ja", "en"), ("ko", "en"))]
print("\n## 새로 계산")
for n in res:
    for a, b in new:
        print(f"{n} {nm(a)} − {nm(b)} = {diff(n, a, b)}")
