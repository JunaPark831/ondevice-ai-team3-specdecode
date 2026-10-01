# runs.csv + costs.json → summary.md (표) + summary.png (그림)
#   speedup = (SD decode tok/s) / (같은 프롬프트 vanilla decode tok/s, 앞뒤 평균)   ← TTFT 제외
#   α  = 수락 / (수락 + 거절 횟수)   (라운드마다 첫 거절에서 멈추므로 잘린 기하분포 MLE)
#   τ  = 라운드(=target forward)당 확정 토큰 수
#   이론1 Leviathan : (1 - α^(γ+1)) / ((1 - α)(γc + 1)),  c = T_draft(1) / T_target(1)
#   이론2 실측 비용 : τ · T_t(1) / (γ·T_d(1) + T_t(γ+1))   ← "검증 비용이 γ 와 무관" 가정을 뺀 것
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 사용법: python plot.py results/main [results/main_rep1 ...]
#   폴더 하나면 거기에, 여러 개면 results/combined 에 저장. 비용·환경은 γ 범위가 가장 넓은 폴더 기준
dirs = [Path(x) for x in sys.argv[1:]] or [Path("results/main")]
d = dirs[0] if len(dirs) == 1 else dirs[0].parent / "combined"
d.mkdir(exist_ok=True)
rows = [r for x in dirs for r in csv.DictReader(open(x / "runs.csv"))]
src = max(dirs, key=lambda x: len(json.load(open(x / "costs.json"))["target"]))
costs = json.load(open(src / "costs.json"))
env = json.load(open(src / "env.json"))
Tt = {int(k): v for k, v in costs["target"].items()}
Td = costs["drafts"]


def block(r):
    return r["rep"], r["task"], r["pid"]


# 드리프트 검사: 블록(프롬프트 1개의 sweep) 앞뒤 vanilla 속도가 10% 넘게 다르면 그 블록은 버린다
van = {block(r): r for r in rows if r["draft"] == "vanilla"}
after = {block(r): r for r in rows if r["draft"] == "vanilla_after"}
drift = {k: float(after[k]["decode_tps"]) / float(van[k]["decode_tps"]) - 1 for k in after}
bad = {k for k, x in drift.items() if abs(x) > 0.10}
n_blocks = len(van)
drift_note = (f"블록 {n_blocks}개 사용 (드리프트 검사 {len(drift)}개 중 {len(bad)}개 제외: "
              + ", ".join(f"rep{k[0]} {k[1]}#{k[2]} {drift[k]:+.0%}" for k in sorted(bad)) + ")") if drift else f"블록 {n_blocks}개 (드리프트 검사 없음)"
n_blocks -= len(bad)
print(drift_note)

G = defaultdict(list)  # (draft, γ, task) → runs.  task="all" 은 전체
for r in rows:
    if r["draft"].startswith("vanilla") or block(r) in bad:
        continue
    k = block(r)
    v = float(van[k]["decode_tps"])
    if k in after:  # 앞뒤 vanilla 평균을 기준으로 (블록 안의 완만한 드리프트를 반으로 줄임)
        v = (v + float(after[k]["decode_tps"])) / 2
    r["speedup"] = float(r["decode_tps"]) / v
    for t in (r["task"], "all"):
        G[(r["draft"], int(r["gamma"]), t)].append(r)

drafts = list(Td)
gammas = sorted({g for _, g, _ in G})
tasks = [t for t in env["tasks"]]


def leviathan(a, g, c):
    return (g + 1) / (g * c + 1) if a >= 1 else (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))


def refined(tau, g, dn):
    return tau * Tt[1] / (g * Td[dn] + Tt[g + 1])


def agg(rs):
    """예측은 run 마다 (그 run 의 α, τ 로) 계산한 뒤 평균한다.
    α·τ 를 먼저 합산(pooled)하면 라운드가 많은(=τ 낮은) 프롬프트에 가중치가 쏠려 큰 γ 에서 예측이 낮게 나온다."""
    sp = [r["speedup"] for r in rs]
    A = sum(int(r["accepted"]) for r in rs)
    R = sum(int(r["rejects"]) for r in rs)
    n = sum(int(r["rounds"]) for r in rs)
    lev, ref = [], []
    for r in rs:
        a, rj, k, g, dn = int(r["accepted"]), int(r["rejects"]), int(r["rounds"]), int(r["gamma"]), r["draft"]
        lev.append(leviathan(a / (a + rj) if a + rj else 1.0, g, Td[dn] / Tt[1]))
        ref.append(refined((a + k) / k, g, dn))
    return dict(sp=statistics.mean(sp), sd=statistics.stdev(sp) if len(sp) > 1 else 0.0,
                alpha=A / (A + R) if A + R else 1.0, tau=(A + n) / n,
                lev=statistics.mean(lev), ref=statistics.mean(ref),
                same=sum(r["same"] == "True" for r in rs) / len(rs),
                draft_share=sum(float(r["t_draft"]) for r in rs) / sum(float(r["t_draft"]) + float(r["t_verify"]) for r in rs))


S = {k: agg(v) for k, v in G.items()}


def reversal(dn, task="all", key=None):
    """최고점 γ, 그 뒤 처음으로 1 아래로 떨어지는 γ"""
    key = key or (lambda g: S[(dn, g, task)]["sp"])
    sp = {g: key(g) for g in gammas}
    best = max(gammas, key=sp.get)
    if sp[best] < 1:
        return best, sp[best], "전 구간 역전"
    after = [g for g in gammas if g > best and sp[g] < 1]
    return best, sp[best], (f"γ={after[0]}" if after else f"γ≤{gammas[-1]} 에서 없음")


# ---------------- summary.md ----------------
L = []
L.append(f"# 결과 요약 — {d.name}\n")
L.append(f"- {env['chip']} {env['mem_gb']}GB · {env['device']} · {env['dtype']} · torch {env['torch']} · transformers {env['transformers']} · {env['date']}")
L.append(f"- target `{env['target']}` · drafts {', '.join('`'+x+'`' for x in env['drafts'])} · greedy · max_new {env['max_new']} · 프롬프트 {len(tasks)}태스크×{env['n_prompts']}")
L.append(f"- 데이터: {', '.join(str(x) for x in dirs)} · {drift_note}\n")
L.append("## forward 비용 (ctx %d 토큰, median)\n" % costs["ctx_len"])
L.append("| | ms | c = T_d(1)/T_t(1) |\n|---|---|---|")
L.append(f"| target 1토큰 | {Tt[1]*1e3:.1f} | |")
for dn in drafts:
    L.append(f"| {dn} 1토큰 | {Td[dn]*1e3:.1f} | **{Td[dn]/Tt[1]:.3f}** |")
L.append("\n| target k토큰 | " + " | ".join(str(k) for k in Tt) + " |\n|---|" + "---|" * len(Tt))
L.append("| T_t(k)/T_t(1) | " + " | ".join(f"{Tt[k]/Tt[1]:.2f}" for k in Tt) + " |\n")

L.append("## 성능 역전 지점 (전체 태스크)\n")
L.append("| draft | 최고 γ | 최고 speedup | 역전(실측) | 역전(Leviathan 예측) | 역전(실측 비용 모델) |\n|---|---|---|---|---|---|")
for dn in drafts:
    b, s, rv = reversal(dn)
    _, _, rl = reversal(dn, key=lambda g: S[(dn, g, 'all')]["lev"])
    _, _, rr = reversal(dn, key=lambda g: S[(dn, g, 'all')]["ref"])
    L.append(f"| {dn} | {b} | {s:.2f}× | **{rv}** | {rl} | {rr} |")

for dn in drafts:
    L.append(f"\n## {dn} — γ sweep (전체 태스크, mean ± std over 프롬프트×반복)\n")
    L.append("| γ | speedup | α | τ | Leviathan 예측 | 실측비용 예측 | draft 시간 비중 | vanilla와 출력 동일 |\n|---|---|---|---|---|---|---|---|")
    for g in gammas:
        s = S[(dn, g, "all")]
        L.append(f"| {g} | {s['sp']:.2f} ± {s['sd']:.2f} | {s['alpha']:.3f} | {s['tau']:.2f} | "
                 f"{s['lev']:.2f} | {s['ref']:.2f} | {s['draft_share']:.0%} | {s['same']:.0%} |")

L.append("\n## 태스크별 (draft별 α, 최고 γ, 역전 지점)\n")
L.append("| draft | task | α (γ=4) | 최고 γ | 최고 speedup | 역전 |\n|---|---|---|---|---|---|")
for dn in drafts:
    for t in tasks:
        b, s, rv = reversal(dn, t)
        L.append(f"| {dn} | {t} | {S[(dn, 4 if 4 in gammas else gammas[0], t)]['alpha']:.3f} | {b} | {s:.2f}× | {rv} |")
(d / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ---------------- summary.png ----------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz 기본 팔레트 1~4 순서 고정
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False})


def base(ax, title, ylabel="speedup (× vanilla)"):
    ax.axhline(1, color=INK2, lw=1, ls="--", zorder=1)
    ax.text(gammas[0], 1, "vanilla ", color=INK2, va="bottom", ha="left", fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("γ (draft 길이)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(gammas)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.plot([x], [y], "o", color=color, ms=6)


fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))

ax = axs[0, 0]
base(ax, "(a) draft 크기별 speedup — 전체 태스크 (막대 = 프롬프트 간 std)")
for i, dn in enumerate(drafts):
    y = [S[(dn, g, "all")]["sp"] for g in gammas]
    e = [S[(dn, g, "all")]["sd"] for g in gammas]
    ax.errorbar(gammas, y, yerr=e, color=SERIES[i], marker="os"[i % 2], capsize=2, elinewidth=1, ecolor=SERIES[i] + "80",
                label=f"{dn} (c={Td[dn]/Tt[1]:.2f})")
ax.legend(loc="upper right")

dn0 = drafts[0]
ax = axs[0, 1]
base(ax, f"(b) 태스크별 speedup — draft {dn0}")
for i, t in enumerate(tasks):
    y = [S[(dn0, g, t)]["sp"] for g in gammas]
    ax.plot(gammas, y, color=SERIES[i], marker="o", label=f"{t} (α={S[(dn0, 4 if 4 in gammas else gammas[0], t)]['alpha']:.2f})")
    end_label(ax, gammas[-1], y[-1], t, SERIES[i])
ax.legend(loc="upper right")

ax = axs[1, 0]
base(ax, f"(c) 이론 vs 실측 — draft {dn0}")
c0 = Td[dn0] / Tt[1]
ax.plot(gammas, [S[(dn0, g, "all")]["sp"] for g in gammas], color=SERIES[0], marker="o", label="실측")
ax.plot(gammas, [S[(dn0, g, "all")]["lev"] for g in gammas], color=SERIES[1], marker="s", ls=":", label=f"Leviathan 식 (실측 α, c={c0:.2f})")
ax.plot(gammas, [S[(dn0, g, "all")]["ref"] for g in gammas], color=SERIES[2], marker="^", ls="--", label="실측 비용 모델 (τ, T_t(γ+1))")
ax.legend(loc="upper right")

ax = axs[1, 1]
ks = sorted(Tt)
ax.plot(ks, [Tt[k] * 1e3 for k in ks], color=INK2, marker="o", label="target: k토큰 한 번에 검증")
for i, dn in enumerate(drafts):
    ax.axhline(Td[dn] * 1e3, color=SERIES[i], lw=2, ls="--", label=f"{dn}: 1토큰 draft")
ax.set_ylim(0, Tt[ks[-1]] * 1e3 * 1.25)
ax.set_title("(d) forward 1번 비용 — 검증은 k≤16 까지 거의 그대로", loc="left", color=INK, fontsize=11)
ax.set_xlabel("k (= γ + 1)")
ax.set_ylabel("ms")
ax.set_xticks(ks[::2] if len(ks) > 14 else ks)
ax.legend(loc="center right")

fig.suptitle(f"Speculative decoding on {env['chip']} ({env['device'].upper()}) — target {env['target'].split('/')[-1]}",
             x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(d / "summary.png", dpi=150)
print("saved", d / "summary.png")
