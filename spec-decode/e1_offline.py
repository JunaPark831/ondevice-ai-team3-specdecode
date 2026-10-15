# E1 오프라인 α 파이프라인 — α 는 하드웨어와 무관하게 한 번만 재고, c 는 하드웨어별로 따로 재서 식으로 합친다
#   gen   : target greedy 출력 y 생성 (sd.generate vanilla 경로 = W02 vanilla 와 같음) → targets.jsonl
#   score : prompt+y 를 target·draft 에 teacher-forcing 으로 forward 1번씩 → 위치별 지표 positions.csv, top-k logit topk/
#   sim   : 위치별 일치 m_i 로 모든 γ 의 SD 라운드를 그대로 재현 → sim.csv, summary.md   (모델 불필요)
#
# greedy 에서는 teacher-forcing 이 정확하다: 수락된 draft 토큰 = target 토큰이므로, draft 가 i 번째 토큰을 맞히는지는
#   "y[:i] 를 보고 draft argmax == y[i]" 와 같다 (m_i). 라운드 = m 을 앞에서부터 γ 개씩 읽다가 첫 0 에서 멈추는 것.
# sampling(temperature>0) 의 기대 수락률은 위치별 Σ min(p, q) = 1 − TV(p, q) (Leviathan 의 α 정의). 같은 forward 에서 기록한다.
#   단, y 는 greedy 궤적이므로 sampling 궤적 위의 α 와는 다를 수 있다 (근사).
#
#   python e1_offline.py gen   --prompts bench --out results/e1_w02 --tail 24      # W02 12프롬프트 (검증용)
#   python e1_offline.py score --out results/e1_w02
#   python e1_offline.py sim   --out results/e1_w02 --validate results/main results/main_rep1 --costs results/main_rep1/costs.json
#   python e1_offline.py gen   --prompts prompts/flores_devtest200.jsonl --out results/e1_flores200   # make_prompts.py 로 만든 파일
import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench import PROMPTS, short
from sd import generate, sync

DEFAULT_DEV = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
GAMMAS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def load_prompts(src, n):
    """bench → W02 프롬프트 (id = task#pid). 그 외 → jsonl 파일 {id, task, prompt[, ...]}"""
    if src == "bench":
        rows = [dict(id=f"{t}#{i}", task=t, prompt=p) for t, ps in PROMPTS.items() for i, p in enumerate(ps)]
    else:
        rows = read_jsonl(src)
        assert rows and all({"id", "task", "prompt"} <= r.keys() for r in rows), f"{src}: id, task, prompt 필드 필요"
    return rows[:n] if n else rows


def tokenizer_and_eos(name):
    tok = AutoTokenizer.from_pretrained(name)
    return tok, {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}


def load_model(name, dev, dtype):
    return AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(dev).eval()


def write_env(out, args, stage):
    env = dict(stage=stage, date=datetime.now().isoformat(timespec="seconds"), chip=platform.processor(),
               os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
               transformers=transformers.__version__, **{k: v for k, v in vars(args).items() if k not in ("out", "func")})
    (out / f"env_{stage}.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))


# ---------------- gen ----------------
def cmd_gen(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_env(out, args, "gen")
    tok, eos = tokenizer_and_eos(args.target)
    dev = torch.device(args.device)
    rows = load_prompts(args.prompts, args.n)
    path = out / "targets.jsonl"
    done = {r["id"] for r in read_jsonl(path)}  # 밤샘 실행이 끊겨도 이어서
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)}개 중 {len(done)}개 완료, {len(todo)}개 남음 → {path}", flush=True)
    if not todo:
        return
    target = load_model(args.target, dev, args.dtype)
    t_start = time.time()
    n_tok = 0
    for i, r in enumerate(todo):
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text).input_ids
        t0 = time.perf_counter()
        y, st = generate(target, ids, args.max_new, eos=eos)
        # tail: y 가 끝난 뒤(eos 이후 포함) target greedy 토큰을 더 만든다. 온라인 SD 는 마지막 라운드에서 끝을 넘겨
        #   draft 하므로, W02 runs.csv 와 라운드 단위로 정확히 맞춰 보려면 필요하다 (--validate). 분석용이면 0.
        tail = generate(target, ids + y, args.tail, eos=())[0] if args.tail else []
        sync(dev)
        rec = dict(r, prompt_ids=ids, y=y, tail=tail, eos_end=bool(y) and y[-1] in eos,
                   text=tok.decode(y, skip_special_tokens=True), gen_s=round(time.perf_counter() - t0, 3),
                   target=short(args.target), dtype=args.dtype, device=args.device)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_tok += len(y) + len(tail)
        el = time.time() - t_start
        print(f"[{i + 1}/{len(todo)}] {r['id']}  {len(y)} tok  {n_tok / el:.1f} tok/s  "
              f"남은 시간 ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)


# ---------------- score ----------------
@torch.no_grad()
def tf_logits(model, ids, n):
    """ids 전체를 캐시 없이 forward → 마지막 n 개 토큰 각각을 예측한 위치의 logits [n, V] (float32)"""
    x = torch.tensor([ids], device=model.device)
    return model(input_ids=x, use_cache=False, logits_to_keep=n + 1).logits[0, :n].float()


def char_steps(tok, y):
    """토큰 i 가 확정될 때 늘어나는 글자 수 (한글 1글자가 토큰 여러 개로 쪼개지면 앞 토큰은 0, 합은 전체 글자 수)"""
    lens = [len(tok.decode(y[: i + 1], skip_special_tokens=True).rstrip("�")) for i in range(len(y))]
    return [b - a for a, b in zip([0] + lens[:-1], lens)]


def cmd_score(args):
    out = Path(args.out)
    write_env(out, args, "score")
    samples = read_jsonl(out / "targets.jsonl")
    assert samples, f"{out}/targets.jsonl 없음 — gen 먼저"
    tok, _ = tokenizer_and_eos(args.target)
    dev = torch.device(args.device)
    temps = args.temps
    fields = ["id", "task", "draft", "pos", "tok", "in_y", "nchar", "match", "t_self", "top1_agree",
              "p_y", "q_y", "h_p", "h_q", "kl"] + [f"acc_t{t:g}" for t in temps]

    pos_path, done_path = out / "positions.csv", out / "scored.txt"
    done = set(done_path.read_text(encoding="utf-8").split("\n")) - {""} if done_path.exists() else set()
    if pos_path.exists():  # 중간에 끊긴 샘플의 일부 행은 지우고 다시
        keep = [r for r in csv.DictReader(open(pos_path, encoding="utf-8")) if r["id"] in done]
        with open(pos_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(keep)
    else:
        with open(pos_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writeheader()
    todo = [s for s in samples if s["id"] not in done]
    print(f"{len(samples)}개 중 {len(done)}개 완료, {len(todo)}개 남음", flush=True)
    if not todo:
        return

    target = load_model(args.target, dev, args.dtype)
    drafts = {short(d): load_model(d, dev, args.dtype) for d in args.drafts}
    topk_dir = out / "topk"
    if args.topk:
        topk_dir.mkdir(exist_ok=True)
    warned = set()
    t_start = time.time()
    for i, s in enumerate(todo):
        full = s["y"] + s["tail"]
        n, ny = len(full), len(s["y"])
        if n == 0:
            continue
        ids = s["prompt_ids"] + full
        y = torch.tensor(full, device=dev)
        nchar = char_steps(tok, s["y"]) + [0] * len(s["tail"])

        lt = tf_logits(target, ids, n)
        t_arg = lt.argmax(-1)
        lp = torch.log_softmax(lt, -1)
        p = lp.exp()
        h_p = -(p * lp).sum(-1)
        npz = {}
        if args.topk:
            v, ix = lt.topk(args.topk, -1)
            npz.update(t_ids=ix.int().cpu().numpy(), t_val=v.half().cpu().numpy())

        rows = []
        for dn, dm in drafts.items():
            ld = tf_logits(dm, ids, n)
            V = min(lt.shape[-1], ld.shape[-1])  # embedding 패딩 크기가 다를 수 있음 (실제 어휘 밖이라 무시)
            if ld.shape[-1] != lt.shape[-1] and dn not in warned:
                print(f"  주의: vocab 크기 target {lt.shape[-1]} vs {dn} {ld.shape[-1]} → 앞 {V} 개만 비교")
                warned.add(dn)
            ltv, ldv = lt[:, :V], ld[:, :V]
            d_arg = ld.argmax(-1)
            lq = torch.log_softmax(ld, -1)
            q = lq.exp()
            h_q = -(q * lq).sum(-1)
            kl = (p[:, :V] * (lp[:, :V] - lq[:, :V])).sum(-1)
            acc = {}
            for t in temps:
                pt, qt = torch.softmax(ltv / t, -1), torch.softmax(ldv / t, -1)
                acc[t] = 1 - 0.5 * (pt - qt).abs().sum(-1)  # = Σ min(p, q)
            cols = dict(match=(d_arg == y), t_self=(t_arg == y), top1_agree=(d_arg == t_arg),
                        p_y=p.gather(1, y[:, None])[:, 0], q_y=q.gather(1, y[:, None])[:, 0], h_p=h_p, h_q=h_q, kl=kl,
                        **{f"acc_t{t:g}": acc[t] for t in temps})
            cols = {k: v.cpu().tolist() for k, v in cols.items()}
            for j in range(n):
                rows.append(dict(id=s["id"], task=s["task"], draft=dn, pos=j, tok=full[j], in_y=int(j < ny), nchar=nchar[j],
                                 **{k: (int(v[j]) if isinstance(v[j], bool) else round(v[j], 5)) for k, v in cols.items()}))
            if args.topk:
                v, ix = ld.topk(args.topk, -1)
                npz.update({f"d_ids_{dn}": ix.int().cpu().numpy(), f"d_val_{dn}": v.half().cpu().numpy()})

        with open(pos_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writerows(rows)
        if args.topk:
            np.savez_compressed(topk_dir / f"{s['id'].replace('/', '_').replace('#', '_')}.npz", y=np.array(full), in_y=ny, **npz)
        with open(done_path, "a", encoding="utf-8") as f:
            f.write(s["id"] + "\n")
        el = time.time() - t_start
        print(f"[{i + 1}/{len(todo)}] {s['id']}  {n} pos  " + "  ".join(
            f"{dn} m={statistics.mean(r['match'] for r in rows if r['draft'] == dn and r['in_y'] and r['pos'] > 0) if ny > 1 else float('nan'):.2f}"
            for dn in drafts) + f"  남은 시간 ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)


# ---------------- sim ----------------
def simulate(m, gamma, n=None, max_new=None, eos_at=None):
    """m[j] = draft 가 j 번째 출력 토큰을 맞혔는지 (j=0 은 prefill 에서 target 이 바로 내므로 쓰지 않음)
    budget 모드 (max_new=None): y 길이 n 안에서만 draft 한다 (남은 토큰보다 많이 draft 하지 않음). 분석용 정의.
    online 모드 (max_new 지정): sd.generate 와 똑같이 — 끝을 넘겨서도 γ 개 draft, eos 가 확정되면 멈춤. W02 검증용.
    반환: rounds, drafted, accepted, rejects, ok(online 에서 m 이 모자라지 않았는지)"""
    R = D = A = J = 0
    pos, ok = 1, True
    if max_new is None:
        while pos < n:
            d = min(gamma, n - 1 - pos)
            a = 0
            while a < d and m[pos + a]:
                a += 1
            R, D, A, J = R + 1, D + d, A + a, J + (a < d)
            pos += a + 1
        return R, D, A, J, ok
    if eos_at == 0:
        return R, D, A, J, ok
    while pos < max_new:
        a = 0
        while a < gamma and pos + a < len(m) and m[pos + a]:
            a += 1
        if a < gamma and pos + a >= len(m):
            ok = False  # tail 부족
        R, D, A, J = R + 1, D + gamma, A + a, J + (a < gamma)
        if eos_at is not None and pos <= eos_at <= pos + a:
            break
        pos += a + 1
    return R, D, A, J, ok


def boot_ci(xs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, float)
    ms = rng.choice(xs, (n, len(xs))).mean(1)
    return np.percentile(ms, 2.5), np.percentile(ms, 97.5)


def cmd_sim(args):
    out = Path(args.out)
    samples = {s["id"]: s for s in read_jsonl(out / "targets.jsonl")}
    P = defaultdict(list)  # (id, draft) → 위치 행 (pos 순서)
    for r in csv.DictReader(open(out / "positions.csv", encoding="utf-8")):
        P[(r["id"], r["draft"])].append(r)
    drafts = sorted({d for _, d in P})
    acc_cols = [k for k in next(iter(P.values()))[0] if k.startswith("acc_t")]
    costs = json.load(open(args.costs)) if args.costs else None

    sim_rows, per_sent = [], []
    for (sid, dn), rs in P.items():
        rs.sort(key=lambda r: int(r["pos"]))
        ys = [r for r in rs if r["in_y"] == "1"]
        m = [r["match"] == "1" for r in ys]
        n, task = len(ys), samples[sid]["task"]
        dec = ys[1:]  # 첫 토큰은 prefill 에서 target 이 냄 → decode 구간만
        if not dec:
            continue
        per_sent.append(dict(id=sid, task=task, draft=dn, n_tok=n, n_char=sum(int(r["nchar"]) for r in ys),
                             alpha_pos=statistics.mean(m[1:]), t_self=statistics.mean(r["t_self"] == "1" for r in dec),
                             h_p=statistics.mean(float(r["h_p"]) for r in dec), kl=statistics.mean(float(r["kl"]) for r in dec),
                             **{k: statistics.mean(float(r[k]) for r in dec) for k in acc_cols}))
        chars_dec = sum(int(r["nchar"]) for r in dec)
        for g in args.gammas:
            R, D, A, J, _ = simulate(m, g, n=n)
            row = dict(id=sid, task=task, draft=dn, gamma=g, rounds=R, drafted=D, accepted=A, rejects=J,
                       tau=(A + R) / R, alpha=A / (A + J) if A + J else 1.0, chars_per_round=chars_dec / R)
            if costs and dn in costs["drafts"]:
                Tt = {int(k): v for k, v in costs["target"].items()}
                Td = costs["drafts"][dn]
                c = Td / Tt[1]
                a_ = row["alpha"]
                row["pred_leviathan"] = (g + 1) / (g * c + 1) if a_ >= 1 else (1 - a_ ** (g + 1)) / ((1 - a_) * (g * c + 1))
                if g + 1 in Tt:
                    row["pred_cost"] = row["tau"] * Tt[1] / (g * Td + Tt[g + 1])
            sim_rows.append(row)

    with open(out / "sim.csv", "w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for r in sim_rows for k in r))
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(sim_rows)
    with open(out / "sentences.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(per_sent[0]))
        w.writeheader()
        w.writerows(per_sent)

    # ---- summary.md ----
    L = [f"# E1 오프라인 α — {out.name}\n"]
    env = read_jsonl_env(out)
    if env:
        L.append(f"- target `{env.get('target')}` · drafts {env.get('drafts')} · {env.get('device')} · {env.get('dtype')} · {env.get('date')}")
    L.append(f"- 문장 {len(samples)}개 · α_pos = 문장별 위치 일치율(greedy, γ 무관)의 평균 [bootstrap 95% CI] · "
             f"acc_tT = 문장별 mean(1−TV) (sampling 기대 수락률, temperature T) · t_self = teacher-forcing target argmax 가 y 와 같은 비율\n")
    tasks = sorted({s["task"] for s in per_sent})
    L.append("| draft | task | 문장 | 글자/토큰 | α_pos [95% CI] | " + " | ".join(acc_cols) + " | H(p) | t_self |")
    L.append("|---|---|---|---|---|" + "---|" * (len(acc_cols) + 2))
    for dn in drafts:
        for t in tasks + ["all"]:
            ss = [s for s in per_sent if s["draft"] == dn and (t == "all" or s["task"] == t)]
            if not ss:
                continue
            a = [s["alpha_pos"] for s in ss]
            lo, hi = boot_ci(a) if len(a) > 1 else (a[0], a[0])
            L.append(f"| {dn} | {t} | {len(ss)} | {sum(s['n_char'] for s in ss) / sum(s['n_tok'] for s in ss):.2f} | "
                     f"{statistics.mean(a):.3f} [{lo:.3f}, {hi:.3f}] | "
                     + " | ".join(f"{statistics.mean(s[k] for s in ss):.3f}" for k in acc_cols)
                     + f" | {statistics.mean(s['h_p'] for s in ss):.2f} | {statistics.mean(s['t_self'] for s in ss):.3f} |")

    L.append("\n## γ 별 (전체, 문장 평균) — α = 수락/(수락+거절), τ = 라운드당 확정 토큰, 글자/라운드 = target forward 1번에 전진하는 글자 수\n")
    has_pred = any("pred_leviathan" in r for r in sim_rows)
    L.append("| draft | γ | α | τ | 글자/라운드 |" + (" Leviathan 예측 | 실측비용 예측 |" if has_pred else ""))
    L.append("|---|---|---|---|---|" + ("---|---|" if has_pred else ""))
    for dn in drafts:
        for g in args.gammas:
            rs = [r for r in sim_rows if r["draft"] == dn and r["gamma"] == g]
            line = (f"| {dn} | {g} | {statistics.mean(r['alpha'] for r in rs):.3f} | {statistics.mean(r['tau'] for r in rs):.2f} | "
                    f"{statistics.mean(r['chars_per_round'] for r in rs):.2f} |")
            if has_pred:
                pc = [r["pred_cost"] for r in rs if "pred_cost" in r]
                line += f" {statistics.mean(r['pred_leviathan'] for r in rs):.2f} | " + (f"{statistics.mean(pc):.2f} |" if pc else "- |")
            L.append(line)

    if args.validate:
        L += validate(args, samples, P)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("saved", out / "summary.md", out / "sim.csv", out / "sentences.csv")


def read_jsonl_env(out):
    p = out / "env_score.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def validate(args, samples, P):
    """온라인 측정(bench.py · e0_fp32.py 의 runs.csv)과 비교: 같은 (task, pid, draft, γ) 의 수락/거절/라운드 수"""
    L = ["\n## 검증 — 온라인 측정(runs.csv)과 비교 (online 모드 시뮬레이션)\n"]
    dtype = read_jsonl_env(Path(args.out)).get("dtype")  # E0 runs.csv 처럼 dtype 이 섞여 있으면 score 와 같은 것만
    runs = [r for d in args.validate for r in csv.DictReader(open(Path(d) / "runs.csv", encoding="utf-8"))
            if not r["draft"].startswith("vanilla") and r.get("dtype", dtype) == dtype]
    cmp, short_tail = [], 0
    for r in runs:
        sid = f"{r['task']}#{r['pid']}"
        if (sid, r["draft"]) not in P:
            continue
        s = samples[sid]
        rs = sorted(P[(sid, r["draft"])], key=lambda x: int(x["pos"]))
        m = [x["match"] == "1" for x in rs]
        eos_at = len(s["y"]) - 1 if s["eos_end"] else None
        g = int(r["gamma"])
        R, D, A, J, ok = simulate(m, g, max_new=args.max_new, eos_at=eos_at)
        short_tail += not ok
        cmp.append(dict(draft=r["draft"], gamma=g, same=r["same"] == "True", ok=ok,
                        on=(int(r["accepted"]), int(r["rejects"]), int(r["rounds"])), off=(A, J, R)))
    if not cmp:
        return L + ["비교할 run 없음 (id = task#pid 로 맞춰야 함: gen --prompts bench)"]
    L.append(f"- run {len(cmp)}개 비교" + (f" · tail 부족 {short_tail}개 (gen --tail 을 γ 최대값 이상으로)" if short_tail else ""))
    L.append("\n| draft | 조건 | run | 수락·거절·라운드 정확히 일치 | α 온라인 (pooled) | α 오프라인 (pooled) | 차이 |\n|---|---|---|---|---|---|---|")
    for dn in sorted({c["draft"] for c in cmp}):
        for name, sel in (("same=True", lambda c: c["same"]), ("전체", lambda c: True)):
            cs = [c for c in cmp if c["draft"] == dn and sel(c)]
            if not cs:
                continue
            ex = sum(c["on"] == c["off"] for c in cs)
            a_on = sum(c["on"][0] for c in cs) / sum(c["on"][0] + c["on"][1] for c in cs)
            a_off = sum(c["off"][0] for c in cs) / sum(c["off"][0] + c["off"][1] for c in cs)
            L.append(f"| {dn} | {name} | {len(cs)} | {ex}/{len(cs)} | {a_on:.3f} | {a_off:.3f} | {a_off - a_on:+.3f} |")
    L.append("\n통과 기준(W03): same=True run 은 정확히 일치, 전체 α 차이 ≤ 0.02. "
             "bf16 이면 teacher-forcing(한 번에 forward)과 온라인(한 토큰씩)의 수치가 달라 동점 위치에서 어긋날 수 있다.")
    return L


def main():
    sys.stdout.reconfigure(errors="replace")  # Windows 콘솔(cp949)에서 γ·— 등 출력 오류 방지
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    g = sub.add_parser("gen", help="target greedy 출력 생성")
    g.add_argument("--prompts", default="bench", help="bench 또는 jsonl 경로 {id, task, prompt}")
    g.add_argument("--n", type=int, default=0, help="앞에서 n 개만 (0 = 전부)")
    g.add_argument("--target", default="models/Qwen3-8B")
    g.add_argument("--max-new", type=int, default=128)
    g.add_argument("--tail", type=int, default=0, help="y 뒤에 더 만들 토큰 수 (W02 검증 시 24)")
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("score", help="teacher-forcing 으로 위치별 지표 기록")
    s.add_argument("--target", default="models/Qwen3-8B")
    s.add_argument("--drafts", nargs="+", default=["models/Qwen3-0.6B", "models/Qwen3-1.7B"])
    s.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.7], help="1−TV 를 기록할 temperature")
    s.add_argument("--topk", type=int, default=32, help="위치별 top-k logit 저장 (E3 마스킹 재계산용, 0 = 저장 안 함)")
    s.set_defaults(func=cmd_score)

    m = sub.add_parser("sim", help="γ 별 라운드 시뮬레이션 + 요약 (모델 불필요)")
    m.add_argument("--gammas", type=int, nargs="+", default=GAMMAS)
    m.add_argument("--costs", help="bench.py 의 costs.json → 하드웨어별 speedup 예측")
    m.add_argument("--validate", nargs="*", help="온라인 측정 폴더 (bench.py 또는 e0_fp32.py 의 runs.csv) 와 라운드 단위 비교")
    m.add_argument("--max-new", type=int, default=128, help="--validate 때 온라인 측정의 max_new")
    m.set_defaults(func=cmd_sim)

    for p in (g, s):
        p.add_argument("--device", default=DEFAULT_DEV)
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        p.add_argument("--threads", type=int, default=0)
    for p in (g, s, m):
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    if getattr(args, "threads", 0):
        torch.set_num_threads(args.threads)
    random.seed(0)
    torch.manual_seed(0)
    args.func(args)


if __name__ == "__main__":
    main()
