# W05 · 기기별 SD 실측 요약 (numpy 없이 표준 라이브러리만)
#   python3 summarize_3090.py <결과폴더> [<결과폴더2> ...]
#   결과폴더 = costs_clean.json 과 online_*.csv 가 있는 곳 (예: results/lang, results/lang_3090)
#
# 정의 (W04 w04_mps_summary.md 와 같게 맞춤)
#   c        = draft 1토큰 시간 / target 1토큰 시간            (costs_clean.json, 원문 언어의 문맥)
#   r(k)     = target k토큰 검증 시간 / target 1토큰 시간
#   speedup  = SD decode tok/s / 같은 블록 앞뒤 vanilla decode tok/s 평균   (TTFT 제외)
#   τ        = 문장별 (새 토큰 수 − 1) / 라운드 의 평균,  α = 문장별 수락 / (수락 + 거절) 의 평균
#   예측     = τ / (γ·c + r(γ+1))
#   CI       = 문장 단위 bootstrap 5,000회 95%
import csv
import json
import random
import statistics
import sys
from pathlib import Path


def load_costs(d):
    p = Path(d) / "costs_clean.json"
    if not p.exists():
        return None
    res = json.loads(p.read_text())["clean"]
    out = {}
    for ctx, v in res.items():
        t1 = v["target"]["1"]
        out[ctx] = dict(t1=t1, c={n: t / t1 for n, t in v["drafts"].items()},
                        r={int(k): t / t1 for k, t in v["target"].items()}, again=v.get("target_again_1", float("nan")) / t1)
    return out


def boot(xs, n=5000, seed=0):
    rnd = random.Random(seed)
    ms = sorted(statistics.fmean(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def summarize(d):
    d = Path(d)
    costs = load_costs(d)
    print(f"\n## {d}\n")
    if costs:
        print("| 문맥 | target 1토큰 (ms) | " + " | ".join(f"c · {n}" for n in costs["ctx_en"]["c"]) + " | r(3) | r(5) | r(7) | r(13) | r(17) | target 앞뒤 차이 |")
        print("|---|---:|" + "---:|" * (len(costs["ctx_en"]["c"]) + 6))
        for ctx, v in costs.items():
            print(f"| {ctx} | {v['t1'] * 1e3:.2f} | " + " | ".join(f"{x:.3f}" for x in v["c"].values()) +
                  " | " + " | ".join(f"{v['r'][k]:.3f}" for k in (3, 5, 7, 13, 17)) + f" | {abs(v['again'] - 1) * 100:.1f}% |")
    files = sorted(p for p in d.glob("online_*.csv") if "iso" in p.name)
    if not files:
        print("(online_*iso*.csv 없음)")
        return
    print("\n| 방향 | draft | γ | 실측 speedup [95% CI] | 비용 예측 | α | τ | SD=vanilla 출력 | 드리프트 최대 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in files:
        rows = list(csv.DictReader(open(f)))
        blocks = {}
        for r in rows:
            blocks.setdefault((r["src"], r["tgt"], r["sid"], r["attempt"]), []).append(r)
        # 드리프트로 다시 잰 블록(attempt 1)이 있으면 그 문장은 attempt 1 을 쓴다
        best = {}
        for (s, t, sid, a), rs in blocks.items():
            if (s, t, sid) not in best or int(a) > int(best[(s, t, sid)][0]):
                best[(s, t, sid)] = (a, rs)
        conds = {}
        for (s, t, sid), (a, rs) in best.items():
            v0 = float(next(r["decode_tps"] for r in rs if r["draft"] == "vanilla"))
            v1 = float(next(r["decode_tps"] for r in rs if r["draft"] == "vanilla_after"))
            for r in rs:
                if r["draft"].startswith("vanilla"):
                    continue
                k = (s, t, r["draft"], int(r["gamma"]))
                c = conds.setdefault(k, dict(sp=[], tau=[], alpha=[], same=0, n=0, drift=0.0))
                c["sp"].append(float(r["decode_tps"]) / ((v0 + v1) / 2))
                A, R, J = int(r["accepted"]), int(r["rounds"]), int(r["rejects"])
                c["tau"].append((int(r["n_new"]) - 1) / R)
                if A + J:
                    c["alpha"].append(A / (A + J))
                c["same"] += r["same"] == "True"; c["n"] += 1
                c["drift"] = max(c["drift"], abs(v1 / v0 - 1))
        for (s, t, n, g), c in sorted(conds.items()):
            tau = statistics.fmean(c["tau"])
            alpha = statistics.fmean(c["alpha"]) if c["alpha"] else float("nan")
            sp = statistics.fmean(c["sp"])
            lo, hi = boot(c["sp"])
            pred = "—"
            if costs:
                cc = costs["ctx_ko" if s == "ko" else "ctx_en"]
                if n in cc["c"] and g + 1 in cc["r"]:
                    pred = f"{tau / (g * cc['c'][n] + cc['r'][g + 1]):.3f}"
            print(f"| {s}→{t} | {n} | {g} | {sp:.3f} [{lo:.3f}, {hi:.3f}] | {pred} | {alpha:.3f} | {tau:.3f} | {c['same']}/{c['n']} | {c['drift'] * 100:.1f}% |")


if __name__ == "__main__":
    for d in sys.argv[1:] or ["results/lang"]:
        summarize(d)
