# Q2-a (W04): Windows 노트북 하드웨어 조건별 SD 비용 곡선 — llama.cpp 의 llama-bench 로 c 와 r(k) 를 잰다.
#   c    = T_d(1) / T_t(1)        draft 1토큰 시간 ÷ target 1토큰 시간 (둘 다 tg, 문맥 --depth 토큰 뒤)
#   r(k) = T_t(k) / T_t(1)        target 이 k토큰을 한 번에 처리(검증, pp k)하는 시간 ÷ 1토큰 생성 시간(tg)
#   llama.cpp 는 1토큰 생성(tg)과 묶음 처리(pp)가 다른 커널을 쓰므로 pp1 ≠ tg 1토큰일 수 있다 (iGPU 에서 특히).
#   식의 분모 γ·T_d(1) + T_t(γ+1) 을 vanilla T_t(1) 로 나눈 것이 γc + r(γ+1) 이라 기준은 tg 다. r(1) 도 그대로 남긴다.
#   speedup 예측은 S(γ) = τ(γ) / (γ·c + r(γ+1)) 이고 τ 는 lang.py score 결과(하드웨어 무관)에서 온다 (Q2-b).
#
# 조건(블록) 하나 = target(pp k 목록 + tg) → draft 들(pp1 + tg) → target tg 다시. 앞뒤 target tg 가
# --drift 넘게 다르면 그 블록을 버리고 다시 잰다 (bench.py 의 vanilla 앞뒤 검사와 같은 생각).
#
#   python -I q2_costs.py run --dry-run                 # 실행할 명령만 출력
#   python -I q2_costs.py run                           # 전체 조건 (끊겨도 다시 돌리면 끝난 조건은 건너뜀)
#   python -I q2_costs.py run --conds cpu-t8 igpu       # 일부 조건만
#   python -I q2_costs.py summary                       # raw.jsonl → costs.csv, summary.md 다시 만들기
import argparse
import csv
import ctypes
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = HERE.parent.parent / "tools" / "llama.cpp-b11430"
GGUF = HERE / "models" / "gguf"

# 조건 이름 → (llama.cpp 빌드, 스레드, -ngl, target GGUF)
CONDS = {
    "cpu-t2": ("cpu", 2, 0, "Qwen3-8B-Q8_0"),
    "cpu-t4": ("cpu", 4, 0, "Qwen3-8B-Q8_0"),
    "cpu-t8": ("cpu", 8, 0, "Qwen3-8B-Q8_0"),
    "cpu-t16": ("cpu", 16, 0, "Qwen3-8B-Q8_0"),
    "igpu": ("vulkan", 8, 99, "Qwen3-8B-Q8_0"),
    "igpu-q4": ("vulkan", 8, 99, "Qwen3-8B-Q4_K_M"),
}
DRAFTS = ["Qwen3-0.6B-Q8_0", "Qwen3-1.7B-Q8_0"]
# W02 γ 격자(1~6, 8, 10, 12, 16)의 검증 길이 γ+1, 그리고 k=1 (pp1 과 tg 1토큰 비교용)
KS = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17]
BEST_PERF_OVERLAY = "ded574b5-45a0-4f42-8737-46345c09c238"  # Windows 11 전원 모드 '최고 성능'
HIGH_PERF_SCHEMES = {"8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", "e9a42b02-d5df-448d-aa00-03f14749eb61"}  # 고성능, 최고의 성능


class GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

    def __str__(self):
        b = bytes(self.d4)
        return f"{self.d1:08x}-{self.d2:04x}-{self.d3:04x}-{b[:2].hex()}-{b[2:].hex()}"


def power_state():
    """AC 연결 여부, 전원 구성표, Windows 11 전원 모드. 실패하면 None."""
    st = {"ac": None, "scheme": None, "overlay": None}
    if platform.system() != "Windows":
        return st

    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("bat", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = SPS()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        st["ac"] = {0: False, 1: True}.get(s.ac)
        st["battery_saver"] = bool(s.saver)
    p = ctypes.POINTER(GUID)()
    if ctypes.windll.powrprof.PowerGetActiveScheme(None, ctypes.byref(p)) == 0:
        st["scheme"] = str(p.contents)
        ctypes.windll.kernel32.LocalFree(p)
    g = GUID()
    try:
        if ctypes.windll.powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(g)) == 0:
            st["overlay"] = str(g)
    except AttributeError:
        pass
    return st


def avail_ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                    ("ext", ctypes.c_ulonglong)]
    m = MS()
    m.len = ctypes.sizeof(MS)
    if platform.system() == "Windows" and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return round(m.avail / 2**30, 2)
    return None


def bench_cmd(args, cond, model, ps, ns):
    build, threads, ngl, _ = CONDS[cond]
    exe = Path(args.llama_dir) / build / "llama-bench.exe"
    cmd = [str(exe), "-m", str(GGUF / f"{model}.gguf"), "-t", str(threads), "-ngl", str(ngl),
           "-d", str(args.depth), "-r", str(args.reps), "-o", "jsonl"]
    cmd += ["-p", ",".join(map(str, ps)) if ps else "0", "-n", ",".join(map(str, ns)) if ns else "0"]
    return cmd


def run_one(args, cond, role, model, ps, ns, block, attempt, out):
    cmd = bench_cmd(args, cond, model, ps, ns)
    if args.dry_run:
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        return []
    log = out / "logs" / f"{cond}_b{block}_a{attempt}_{role}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as err:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8")
    if p.returncode != 0:
        sys.exit(f"llama-bench 실패 ({cond} {role}), 로그: {log}")
    recs = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{")]
    ram = avail_ram_gb()
    for r in recs:
        r.update(cond=cond, role=role, model=model, block=block, attempt=attempt, avail_ram_gb=ram)
    with open(out / "raw.jsonl", "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {role:<14} {model:<18} {time.time() - t0:6.0f}s  RAM 여유 {ram} GB")
    return recs


def tg_ms(recs):
    r = [x for x in recs if x["n_gen"] > 0]
    return r[0]["avg_ns"] / r[0]["n_gen"] / 1e6 if r else None


def load_raw(out):
    f = out / "raw.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def accepted_blocks(raw):
    """조건별로 drift 검사를 통과한 마지막 블록의 기록. 통과한 블록이 없으면 마지막 블록(표시는 drift)."""
    by = {}
    for r in raw:
        by.setdefault(r["cond"], {}).setdefault((r["block"], r["attempt"]), []).append(r)
    res = {}
    for cond, blocks in by.items():
        ok = [k for k, rs in blocks.items() if any(x["role"] == "check" and x.get("drift_ok") for x in rs)]
        done = [k for k, rs in blocks.items() if any(x["role"] == "check" for x in rs)]
        if ok:
            res[cond] = (blocks[max(ok)], True)
        elif done:
            res[cond] = (blocks[max(done)], False)
    return res


def run(args):
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    conds = args.conds or list(CONDS)
    for c in conds:
        if c not in CONDS:
            sys.exit(f"모르는 조건 {c}. 가능: {', '.join(CONDS)}")
    for m in {CONDS[c][3] for c in conds} | set(args.drafts):
        if not (GGUF / f"{m}.gguf").exists():
            sys.exit(f"GGUF 없음: {GGUF / f'{m}.gguf'}")

    pw = power_state()
    print(f"전원: AC={pw['ac']} 구성표={pw['scheme']} 전원 모드={pw['overlay']}")
    if pw["ac"] is False and not args.allow_battery and not args.dry_run:
        sys.exit("배터리로 돌고 있다. 전원을 연결하거나 --allow-battery.")
    if pw["overlay"] != BEST_PERF_OVERLAY and pw["scheme"] not in HIGH_PERF_SCHEMES:
        print("  경고: 전원 모드가 '최고 성능'/고성능 이 아니다. 클럭이 흔들릴 수 있다.")

    if not args.dry_run:
        env = {"time": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
               "cpu": platform.processor(), "power": pw, "avail_ram_gb": avail_ram_gb(), "llama_dir": str(args.llama_dir),
               "depth": args.depth, "reps": args.reps, "ks": args.ks, "n_gen": args.n_gen, "drift": args.drift,
               "conds": {c: CONDS[c] for c in conds}, "drafts": args.drafts}
        envs = json.loads((out / "env.json").read_text(encoding="utf-8")) if (out / "env.json").exists() else []
        (out / "env.json").write_text(json.dumps(envs + [env], indent=1, ensure_ascii=False), encoding="utf-8")

    raw = load_raw(out)
    done = {c for c, (_, ok) in accepted_blocks(raw).items() if ok}
    block = max([r["block"] for r in raw], default=-1) + 1
    for cond in conds:
        if cond in done and not args.redo:
            print(f"[{cond}] 이미 있음, 건너뜀 (--redo 로 다시)")
            continue
        target = CONDS[cond][3]
        for attempt in range(args.retry + 1):
            print(f"[{cond}] 블록 {block} 시도 {attempt}  {datetime.now():%H:%M:%S}")
            before = run_one(args, cond, "target", target, args.ks, [args.n_gen], block, attempt, out)
            for d in args.drafts:
                run_one(args, cond, "draft", d, [1], [args.n_gen], block, attempt, out)
            after = run_one(args, cond, "target_again", target, [], [args.n_gen], block, attempt, out)
            if args.dry_run:
                break
            a, b = tg_ms(before), tg_ms(after)
            drift = abs(b / a - 1)
            ok = drift <= args.drift
            with open(out / "raw.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"cond": cond, "role": "check", "block": block, "attempt": attempt, "model": target,
                                    "tg_before_ms": a, "tg_after_ms": b, "drift": drift, "drift_ok": ok,
                                    "n_gen": 0, "n_prompt": 0}) + "\n")
            print(f"  target tg 앞 {a:.1f} ms / 뒤 {b:.1f} ms → 차이 {drift:.1%} {'통과' if ok else '초과, 다시 잼'}")
            if ok:
                break
        block += 1
        if not args.dry_run and args.cooldown and cond != conds[-1]:
            time.sleep(args.cooldown)
    if not args.dry_run:
        summary(args)


def summary(args):
    out = Path(args.out)
    acc = accepted_blocks(load_raw(out))
    rows = []
    for cond in [c for c in CONDS if c in acc]:
        recs, ok = acc[cond]
        tgt = [r for r in recs if r["role"] == "target"]
        pp = {r["n_prompt"]: r["avg_ns"] / 1e6 for r in tgt if r["n_prompt"] > 0}
        t1 = tg_ms(tgt)
        chk = next(r for r in recs if r["role"] == "check")
        row = {"cond": cond, "target": CONDS[cond][3], "drift_ok": ok, "drift": round(chk["drift"], 4),
               "T_t1_tg_ms": round(t1, 2), "T_t1_pp_ms": round(pp.get(1, float("nan")), 2)}
        for d in sorted({r["model"] for r in recs if r["role"] == "draft"}):
            dr = [r for r in recs if r["role"] == "draft" and r["model"] == d]
            name = d.split("-")[1]  # 0.6B / 1.7B
            row[f"T_d1_ms[{name}]"] = round(tg_ms(dr), 2)
            row[f"c[{name}]"] = round(tg_ms(dr) / t1, 4)
            dpp = next((r["avg_ns"] / 1e6 for r in dr if r["n_prompt"] == 1), None)
            if dpp and 1 in pp:
                row[f"c_pp[{name}]"] = round(dpp / pp[1], 4)  # 확인용: pp1 끼리의 비
        for k in sorted(pp):
            row[f"r({k})"] = round(pp[k] / t1, 4)
        rows.append(row)
    if not rows:
        print("결과 없음")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "costs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    show = ["cond", "target", "T_t1_tg_ms"] + [k for k in keys if k.startswith("c[")] + \
           [k for k in keys if k.startswith("r(")] + ["drift"]
    md = ["# Q2-a 비용 곡선 (llama-bench)", "",
          f"문맥 {args.depth}토큰 뒤, 반복 {args.reps}회 평균. c = draft tg 1토큰 ÷ target tg 1토큰, r(k) = target pp k ÷ target tg 1토큰."
          "drift = 블록 앞뒤 target tg 차이 (표시 * = 검사 초과 블록).", "",
          "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    for r in rows:
        cells = [str(r.get(k, "")) for k in show]
        if not r["drift_ok"]:
            cells[0] += " *"
        md.append("| " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--conds", nargs="*", help=f"기본: 전부 ({' '.join(CONDS)})")
    ap.add_argument("--drafts", nargs="*", default=DRAFTS)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)  # target 이 한 번에 처리할 토큰 수 k
    ap.add_argument("--n-gen", type=int, default=32)          # tg 로 1토큰 시간을 잴 때 생성 토큰 수
    ap.add_argument("--depth", type=int, default=128)         # 문맥 길이 (FLORES 프롬프트 + 번역 중간쯤)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--drift", type=float, default=0.10)
    ap.add_argument("--retry", type=int, default=1)           # drift 초과 시 다시 재는 횟수
    ap.add_argument("--cooldown", type=int, default=30)       # 조건 사이 쉬는 초 (발열)
    ap.add_argument("--llama-dir", default=str(LLAMA))
    ap.add_argument("--out", default=str(HERE / "results" / "q2_costs"))
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"run": run, "summary": summary}[args.cmd](args)
