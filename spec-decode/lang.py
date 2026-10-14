# EXP-003 · 내용을 통제한 언어별 acceptance (FLORES-200 병렬 문장)
#   prep  : FLORES-200 devtest 에서 문장 N개를 골라 data/flores.jsonl 로 (5개 언어가 같은 문장)
#   gen   : target 이 모든 방향을 greedy 로 번역 (배치) → <out>/gen.jsonl
#   score : target·draft 를 [프롬프트 + target 출력] 에 한 번씩 forward (teacher forcing) → 토큰별 지표
#           <out>/score_<draft>.jsonl.gz.  greedy SD 의 수락 여부는 m_j = [draft argmax == target 출력 y_j] 로
#           전부 정해진다 (수락된 draft = target 토큰이므로 draft 의 문맥이 늘 target 궤적과 같다) → lang_plot.py 가
#           m 으로 모든 γ 의 라운드를 시뮬레이션한다
#   online: 일부 문장에서 실제 SD(sd.py) 를 돌려 시뮬레이션 α 와 속도 예측을 검증 → <out>/online.csv
#   replay: online SD 출력을 다시 teacher forcing → 라운드 단위로 시뮬레이터와 대조 → <out>/replay.jsonl.gz
#   대조 실험: prep_wmt → gen --data data/wmt20.jsonl --dirs ja-en ja-ko zh-en zh-ko --out results/lang_wmt20 (원문이 영어가 아닌 뉴스)
#             gen --instr ko --dirs en-ko ko-en --out results/lang_koinstr (한국어 지시문)
#
# 사용법:
#   .venv/bin/python -I lang.py prep <flores200_dataset 폴더>
#   .venv/bin/python lang.py gen --out results/lang
#   .venv/bin/python lang.py score --drafts models/Qwen3-0.6B models/Qwen3-1.7B models/Qwen3-4B
#   .venv/bin/python lang.py online
import argparse
import csv
import gzip
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = {"en": ("eng_Latn", "English"), "ko": ("kor_Hang", "Korean"), "ja": ("jpn_Jpan", "Japanese"),
         "zh": ("zho_Hans", "Chinese"), "de": ("deu_Latn", "German")}
CJK = ["en", "ko", "ja", "zh"]
DIRECTIONS = [(s, t) for s in CJK for t in CJK if s != t] + [("en", "de"), ("de", "en")]  # 12 + 유럽어 대조 2


def prep(flores_dir, n=200, seed=0):
    def read(code):
        lines = (Path(flores_dir) / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()
        # kor_Hang 일부 줄이 CSV 따옴표로 감싸져 있다: "그는 ""...""고" → 그는 "..."고
        return [s[1:-1].replace('""', '"') if s.startswith('"') and s.endswith('"') and '""' in s else s for s in lines]

    cols = {k: read(code) for k, (code, _) in LANGS.items()}
    # 같은 기사에서 나온 문장끼리는 독립이 아니다 → URL 을 군집 id 로 (부트스트랩을 기사 단위로)
    url = [l.split("\t")[0] for l in (Path(flores_dir) / "metadata_devtest.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    ids = sorted(random.Random(seed).sample(range(len(cols["en"])), n))
    out = HERE / "data" / "flores.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"sid": i, "url": url[i], **{k: cols[k][i] for k in LANGS}}, ensure_ascii=False) + "\n")
    print("saved", out, len(ids))


KO_NAME = {"en": "영어", "ko": "한국어", "ja": "일본어", "zh": "중국어", "de": "독일어"}


def prep_wmt(n=100, seed=0):
    """WMT20 ja-en, zh-en 시험셋의 원문(일본어·중국어 원작 뉴스) n문장씩 → data/wmt20.jsonl. sacrebleu 가 내려받는다."""
    rows = []
    for k, (lang, base) in enumerate((("ja", 10000), ("zh", 20000))):
        echo = lambda f: subprocess.run([str(HERE / ".venv/bin/sacrebleu"), "-t", "wmt20", "-l", f"{lang}-en", "--echo", f],
                                        capture_output=True, text=True, check=True).stdout.splitlines()
        src, doc = echo("src"), echo("docid")
        for i in sorted(random.Random(seed + k).sample(range(len(src)), n)):
            rows.append({"sid": base + i, "url": doc[i], lang: src[i]})
    with open(HERE / "data" / "wmt20.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print("saved data/wmt20.jsonl", len(rows))


def prompt(s, t, text, instr="en"):
    if instr == "ko":  # 대조 실험: 한국어 지시문 (출력 전에 한글이 문맥에 있게)
        return f"다음 {KO_NAME[s]} 텍스트를 {KO_NAME[t]}로 번역하세요. 번역문만 출력하세요.\n\n{text}"
    return f"Translate the following {LANGS[s][1]} text into {LANGS[t][1]}. Output only the translation.\n\n{text}"


def encode(tok, p):
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text).input_ids


def eos_ids(tok):
    return [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


def gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("mps")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.target)
    eos = eos_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(dev).eval()
    sents = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    dirs = [tuple(d.split("-")) for d in args.dirs] if args.dirs else DIRECTIONS

    done = set()
    path = out / "gen.jsonl"
    if path.exists():  # 중단 후 이어 돌리기
        done = {(r["src"], r["tgt"], r["sid"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    jobs = [(s, t, x) for s, t in dirs for x in sents if s in x and (s, t, x["sid"]) not in done]
    jobs.sort(key=lambda j: len(j[2][j[0]]))  # 길이가 비슷한 것끼리 배치 → 패딩 낭비 감소
    print(f"{len(jobs)} jobs (skip {len(done)})", flush=True)

    t0, n_tok = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for b in range(0, len(jobs), args.bs):
            batch = jobs[b: b + args.bs]
            ids = [encode(tok, prompt(s, t, x[s], args.instr)) for s, t, x in batch]
            L = max(map(len, ids))
            pad = tok.pad_token_id
            x = torch.tensor([[pad] * (L - len(i)) + i for i in ids], device=dev)  # 왼쪽 패딩
            m = torch.tensor([[0] * (L - len(i)) + [1] * len(i) for i in ids], device=dev)
            with torch.no_grad():
                y = model.generate(input_ids=x, attention_mask=m, max_new_tokens=args.max_new, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=pad)
            for (s, t, sx), p_ids, row in zip(batch, ids, y[:, L:].tolist()):
                o = next((row[: k + 1] for k, v in enumerate(row) if v in eos), row)  # EOS 포함까지
                n_tok += len(o)
                f.write(json.dumps(dict(src=s, tgt=t, sid=sx["sid"], instr=args.instr, prompt_len=len(p_ids), out=o, eos=o[-1] in eos,
                                        text=tok.decode(o, skip_special_tokens=True)), ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[{b + len(batch)}/{len(jobs)}] {el / 60:.1f} min · {n_tok / el:.0f} tok/s · "
                  f"남은 ~{el / (b + len(batch)) * (len(jobs) - b - len(batch)) / 60:.0f} min", flush=True)


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()


def rows_with_prompts(out, tok, data=HERE / "data" / "flores.jsonl"):
    sents = {x["sid"]: x for x in map(json.loads, open(data, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(out / "gen.jsonl", encoding="utf-8")]
    for r in rows:
        r["prompt"] = encode(tok, prompt(r["src"], r["tgt"], sents[r["sid"]][r["src"]], r.get("instr", "en")))
        assert len(r["prompt"]) == r["prompt_len"], (r["src"], r["tgt"], r["sid"])
    return rows, sents


def out_logits(model, batch, pad):
    """[프롬프트 + 출력] 을 오른쪽 패딩 배치로 forward 1번 → 출력 토큰 위치를 예측하는 logits (fp32) 리스트.
    출력 j 번째 토큰(절대 위치 P+j)은 위치 P+j-1 의 hidden 이 예측한다. lm_head 는 그 구간에만 건다."""
    import torch
    seqs = [r["prompt"] + r["out"] for r in batch]
    L = max(map(len, seqs))
    x = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="mps")
    m = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="mps")
    with torch.no_grad():
        h = model.model(input_ids=x, attention_mask=m).last_hidden_state
        P = [len(r["prompt"]) for r in batch]
        return [model.lm_head(h[i, P[i] - 1: P[i] - 1 + len(r["out"])]).float() for i, r in enumerate(batch)]


def qwen_sampling(logits, T=0.7, k=20, top_p=0.8):
    """HF 샘플링 순서(temperature → top-k → top-p)로 자른 분포. SD 를 샘플링으로 돌리면 p, q 둘 다 이걸 거친다."""
    import torch
    v, i = (logits / T).topk(k, -1)
    pr = v.softmax(-1)
    keep = pr.cumsum(-1) - pr < top_p  # 누적 확률이 top_p 에 닿는 최소 집합 (1등은 항상 남는다)
    pr = pr * keep
    return torch.zeros_like(logits).scatter(1, i, pr / pr.sum(-1, keepdim=True))


def score(args):
    import torch
    from transformers import AutoTokenizer

    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.target)
    rows, _ = rows_with_prompts(out, tok, args.data)
    rows.sort(key=lambda r: len(r["prompt"]) + len(r["out"]))
    target = load_model(args.target)
    drafts = {short(d): load_model(d) for d in args.drafts}
    files = {n: gzip.open(out / f"score_{n}.jsonl.gz", "wt", encoding="utf-8") for n in drafts}

    def rnd(t):
        return [round(v, 4) for v in t.tolist()]

    t0 = time.time()
    for b in range(0, len(rows), args.bs):
        batch = rows[b: b + args.bs]
        T = out_logits(target, batch, tok.pad_token_id)
        for n, model in drafts.items():
            D = out_logits(model, batch, tok.pad_token_id)
            for r, tl, dl in zip(batch, T, D):
                # 첫 출력 토큰은 target prefill 이 만든다 (sd.py 도 draft 가 두 번째 토큰부터 제안) → j ≥ 1 만
                tl, dl = tl[1:], dl[1:]
                y = torch.tensor(r["out"][1:], device="mps")
                lp, lq = tl.log_softmax(-1), dl.log_softmax(-1)
                p, q = lp.exp(), lq.exp()
                p7, q7 = (tl / 0.7).softmax(-1), (dl / 0.7).softmax(-1)
                pq, qq = qwen_sampling(tl), qwen_sampling(dl)
                ly = dl.gather(1, y[:, None])
                rec = dict(src=r["src"], tgt=r["tgt"], sid=r["sid"], y=r["out"][1:],
                           m=(dl.argmax(-1) == y).int().tolist(),          # greedy 수락 (draft 1등 == target 출력)
                           t_ok=(tl.argmax(-1) == y).int().tolist(),       # teacher forcing 에서도 target 1등 == y 인가 (수치 확인)
                           H=rnd(-(p * lp).sum(-1)),                       # target 엔트로피 (nats)
                           py=rnd(p.gather(1, y[:, None])[:, 0]),          # target 이 y 에 준 확률
                           tv=rnd(0.5 * (p - q).abs().sum(-1)),            # T=1 샘플링 SD 의 거절 확률 = TV(p,q)
                           tv07=rnd(0.5 * (p7 - q7).abs().sum(-1)),        # T=0.7
                           tvq=rnd(0.5 * (pq - qq).abs().sum(-1)),         # Qwen3 권장 샘플링 (T=0.7, top-k 20, top-p 0.8)
                           kl=rnd((p * (lp - lq)).sum(-1)),                # KL(p‖q)
                           klr=rnd((q * (lq - lp)).sum(-1)),               # KL(q‖p)
                           Hq=rnd(-(q * lq).sum(-1)),                      # draft 엔트로피
                           ce=rnd(-lq.gather(1, y[:, None])[:, 0]),        # draft 가 y 에 준 -log q
                           rank=(dl > ly).sum(-1).clamp(max=99).tolist())  # draft 분포에서 y 의 순위 (0 = 1등)
                files[n].write(json.dumps(rec) + "\n")
        el = time.time() - t0
        print(f"[{b + len(batch)}/{len(rows)}] {el / 60:.1f} min, 남은 ~{el / (b + len(batch)) * (len(rows) - b - len(batch)) / 60:.0f} min", flush=True)
    for f in files.values():
        f.close()


def online(args):
    import torch
    from transformers import AutoTokenizer
    from bench import fwd_time
    from sd import generate

    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.target)
    eos = set(eos_ids(tok))
    _, sents = rows_with_prompts(out, tok)
    sids = sorted(random.Random(1).sample(sorted(sents), args.n_online))
    target = load_model(args.target)
    drafts = {short(d): load_model(d) for d in args.drafts}
    d0 = short(args.drafts[0])
    # (draft, γ, temperature). T=0.7 은 sd.py 의 순수 temperature rejection sampling (top-k/p 없음)
    configs = [(d0, 4, 0.0), (d0, 8, 0.0)] + [(short(d), 4, 0.0) for d in args.drafts[1:]] + [(d0, 4, 0.7)]
    if args.cfg:  # 예: --cfg Qwen3-4B-Instruct-2507:1:0 Qwen3-4B-Instruct-2507:2:0
        configs = [(c.split(":")[0], int(c.split(":")[1]), float(c.split(":")[2])) for c in args.cfg]
    dirs = [tuple(d.split("-")) for d in args.dirs] if args.dirs else DIRECTIONS
    tag = f"_{args.tag}" if args.tag else ""

    def measure_costs():  # forward 비용: 짧은(en 원문) / 긴(ko 원문) 프롬프트 두 문맥에서
        costs = {}
        for name, (s, t) in {"ctx_en": ("en", "ko"), "ctx_ko": ("ko", "en")}.items():
            ctx = encode(tok, prompt(s, t, sents[sids[0]][s]))
            costs[name] = dict(ctx_len=len(ctx), target={k: fwd_time(target, ctx, k) for k in range(1, 26)},
                               drafts={n: fwd_time(m, ctx, 1) for n, m in drafts.items()})
        print(json.dumps({k: {"T_t(1) ms": v["target"][1] * 1e3, **{f"c[{n}]": t / v["target"][1] for n, t in v["drafts"].items()}}
                          for k, v in costs.items()}), flush=True)
        return costs

    path_csv, path_out, path_cost = out / f"online{tag}.csv", out / f"online_out{tag}.jsonl", out / f"online_costs{tag}.json"
    done, prev = set(), {}
    if path_csv.exists():  # 끊긴 데서 이어 돌리기: 끝난 블록은 건너뛴다 (블록의 행은 블록이 끝날 때 한꺼번에 쓴다)
        for r in csv.DictReader(open(path_csv)):
            prev.setdefault((r["src"], r["tgt"], int(r["sid"]), int(r["attempt"])), {})[r["draft"]] = float(r["decode_tps"])
        done = {k for k, d in prev.items() if "vanilla" in d and "vanilla_after" in d}
        print(f"이어 돌리기: 끝난 블록 {len(done)}개 건너뜀", flush=True)
    costs = json.loads(path_cost.read_text()) if done and path_cost.exists() else {}
    costs["resume" if done else "start"] = measure_costs()
    path_cost.write_text(json.dumps(costs, indent=2))

    for m in drafts.values():  # 워밍업
        generate(target, encode(tok, "Hi"), 16, draft=m, gamma=4, eos=eos)

    blocks = [(s, t, sid) for s, t in dirs for sid in sids]
    random.Random(2).shuffle(blocks)  # 방향과 측정 시각(발열·다른 부하)이 겹치지 않게
    fields = ["src", "tgt", "sid", "attempt", "draft", "gamma", "temp", "n_new", "ttft", "total", "decode_tps", "rounds",
              "drafted", "accepted", "rejects", "t_draft", "t_verify", "same", "match_prefix", "load", "swapins"]
    f = open(path_csv, "a" if done else "w", newline="")
    w = csv.DictWriter(f, fields)
    if not done:
        w.writeheader()
    fo = open(path_out, "a" if done else "w", encoding="utf-8")

    def tps(so):
        return (so["n_new"] - 1) / (so["total"] - so["ttft"])

    def swapins():  # 측정 중 메모리가 swap 으로 밀려났는지 (W03: 다른 작업의 메모리 압박으로 vanilla 15 → 2 tok/s 까지 떨어졌다)
        out_ = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        return int(next(l for l in out_.splitlines() if l.startswith("Swapins")).split()[-1].rstrip("."))

    def run_block(bi, s, t, sid, attempt):
        ids = encode(tok, prompt(s, t, sents[sid][s]))
        load_avg = os.getloadavg()[0]
        sw0 = swapins()
        ref, st = generate(target, ids, args.max_new, eos=eos)
        runs = [("vanilla", 0, 0.0, ref, st)]
        order = configs[:]
        random.Random(bi).shuffle(order)
        for n, g, temp in order:
            torch.manual_seed(bi)  # T>0 샘플링 재현용
            o, so = generate(target, ids, args.max_new, draft=drafts[n], gamma=g, temperature=temp, eos=eos)
            runs.append((n, g, temp, o, so))
        o, so = generate(target, ids, args.max_new, eos=eos)
        runs.append(("vanilla_after", 0, 0.0, o, so))
        dsw = swapins() - sw0
        for n, g, temp, o, so in runs:
            mp = next((i for i, (a, b) in enumerate(zip(o, ref)) if a != b), min(len(o), len(ref)))
            w.writerow(dict(src=s, tgt=t, sid=sid, attempt=attempt, draft=n, gamma=g, temp=temp, same=o == ref, match_prefix=mp,
                            load=round(load_avg, 1), swapins=dsw, decode_tps=tps(so), **{k: so[k] for k in fields if k in so}))
            fo.write(json.dumps(dict(src=s, tgt=t, sid=sid, attempt=attempt, draft=n, gamma=g, temp=temp, out=o, log=so.get("log", []))) + "\n")
        f.flush()
        fo.flush()
        return tps(runs[0][4]), tps(runs[-1][4]), load_avg, dsw

    t0 = time.time()
    redo, n_run = [], 0
    for bi, (s, t, sid) in enumerate(blocks):
        if (s, t, sid, 0) in done:  # 이미 잰 블록: 드리프트였고 재측정이 없으면 끝에서 다시
            d = prev[(s, t, sid, 0)]
            if abs(d["vanilla_after"] / d["vanilla"] - 1) > 0.10 and (s, t, sid, 1) not in done:
                redo.append((bi, s, t, sid))
            continue
        n_run += 1
        v0, v1, load_avg, dsw = run_block(bi, s, t, sid, 0)
        if abs(v1 / v0 - 1) > 0.10:
            redo.append((bi, s, t, sid))  # 드리프트 → 끝에서 한 번 더 (방향마다 같은 문장 수를 유지)
        el = time.time() - t0
        left = len(blocks) - len(done) - n_run
        print(f"[{bi + 1}/{len(blocks)}] {s}→{t}#{sid} vanilla {v0:.1f}→{v1:.1f} tok/s load {load_avg:.0f} swapins {dsw} "
              f"{el / 60:.1f} min, 남은 ~{el / n_run * left / 60:.0f} min", flush=True)
    for bi, s, t, sid in redo:
        v0, v1, _, _ = run_block(bi, s, t, sid, 1)
        print(f"[redo] {s}→{t}#{sid} vanilla {v0:.1f}→{v1:.1f} tok/s", flush=True)
    f.close()
    fo.close()
    costs["end"] = measure_costs()  # 세션 중 비용이 흔들렸는지 (lang_plot.py 가 가장 빠른 세트보다 15% 넘게 느린 세트를 버린다)
    path_cost.write_text(json.dumps(costs, indent=2))


def replay(args):
    """online SD 출력을 teacher forcing 으로 다시 채점 → <out>/replay<tag>.jsonl.gz
    greedy: m 으로 라운드를 다시 시뮬레이션하면 online 로그(라운드별 수락·거절)와 같아야 한다 (시뮬레이터 검증)
    T>0  : 위치마다 β = 1 - TV(p_T, q_T). 실제로 검사된 위치의 Σβ ≈ Σ수락 이어야 한다 (샘플링 α 검증)"""
    import torch
    from transformers import AutoTokenizer

    out = Path(args.out)
    tag = f"_{args.tag}" if args.tag else ""
    tok = AutoTokenizer.from_pretrained(args.target)
    _, sents = rows_with_prompts(out, tok)
    runs = [r for r in map(json.loads, open(out / f"online_out{tag}.jsonl", encoding="utf-8")) if not r["draft"].startswith("vanilla")]
    for r in runs:
        r["prompt"] = encode(tok, prompt(r["src"], r["tgt"], sents[r["sid"]][r["src"]]))
    target = load_model(args.target)
    with gzip.open(out / f"replay{tag}.jsonl.gz", "wt", encoding="utf-8") as fz:
        for n in sorted({r["draft"] for r in runs}):
            model = load_model(str(HERE / "models" / n))
            rs = sorted((r for r in runs if r["draft"] == n), key=lambda r: len(r["prompt"]) + len(r["out"]))
            for b in range(0, len(rs), args.bs):
                batch = rs[b: b + args.bs]
                for r, tl, dl in zip(batch, out_logits(target, batch, tok.pad_token_id), out_logits(model, batch, tok.pad_token_id)):
                    y = torch.tensor(r["out"], device="mps")
                    rec = {k: r[k] for k in ("src", "tgt", "sid", "attempt", "draft", "gamma", "temp", "log")}
                    rec.update(n_out=len(r["out"]), m=(dl.argmax(-1) == y).int().tolist(), t_ok=(tl.argmax(-1) == y).int().tolist())
                    if r["temp"] > 0:
                        T = r["temp"]
                        rec["beta"] = [round(v, 4) for v in (1 - 0.5 * ((tl / T).softmax(-1) - (dl / T).softmax(-1)).abs().sum(-1)).tolist()]
                    fz.write(json.dumps(rec) + "\n")
            del model
            torch.mps.empty_cache()
            print("replayed", n, len(rs), flush=True)


def costs(args):
    """비용만 따로 깨끗하게 → <out>/costs_clean.json. target 과 draft 하나씩만 올리고(메모리 압박을 피해서) 워밍업 뒤
    FLORES 문맥 2개에서 T_t(k), k=1..25 와 draft 1토큰 시간을 median 으로. online 중 잰 값은 swap 에 오염될 수 있다."""
    import torch
    from transformers import AutoTokenizer
    from bench import fwd_time

    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.target)
    _, sents = rows_with_prompts(out, tok)
    sid0 = sorted(random.Random(1).sample(sorted(sents), args.n_online))[0]
    ctxs = {name: encode(tok, prompt(s, t, sents[sid0][s])) for name, (s, t) in {"ctx_en": ("en", "ko"), "ctx_ko": ("ko", "en")}.items()}
    target = load_model(args.target)
    for _ in range(3):  # 워밍업 (가중치를 메모리에 다 올린다)
        fwd_time(target, ctxs["ctx_en"], 8, reps=3)
    res = {name: dict(ctx_len=len(c), target={k: fwd_time(target, c, k, reps=20) for k in range(1, 26)}, drafts={})
           for name, c in ctxs.items()}
    for d in args.drafts:
        m = load_model(d)
        for _ in range(3):
            fwd_time(m, ctxs["ctx_en"], 1, reps=3)
        for name, c in ctxs.items():
            res[name]["drafts"][short(d)] = fwd_time(m, c, 1, reps=20)
        del m
        torch.mps.empty_cache()
    for name in res:  # target 은 draft 를 다 잰 뒤 한 번 더 (세션 안 흔들림 확인용)
        res[name]["target_again_1"] = fwd_time(target, ctxs[name], 1, reps=20)
    (out / "costs_clean.json").write_text(json.dumps({"clean": res}, indent=2))
    print(json.dumps({k: {"T_t(1) ms": v["target"][1] * 1e3, "T_t(1) again": v["target_again_1"] * 1e3,
                          **{f"c[{n}]": t / v["target"][1] for n, t in v["drafts"].items()}} for k, v in res.items()}, indent=1))


def short(name):
    return Path(name).name


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2])
        sys.exit()
    if sys.argv[1] == "prep_wmt":
        prep_wmt()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--target", default=str(HERE / "models/Qwen3-8B"))
    ap.add_argument("--drafts", nargs="+", default=[str(HERE / f"models/Qwen3-{s}") for s in ("0.6B", "1.7B")])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-online", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", default=str(HERE / "results/lang"))
    ap.add_argument("--cfg", nargs="*")   # online: draft:γ:T 목록 (기본 설정 대신)
    ap.add_argument("--dirs", nargs="*")  # online: 방향 부분집합, 예: en-ko ko-en
    ap.add_argument("--tag", default="")  # online/replay: 결과 파일 이름 뒤에 붙임
    ap.add_argument("--data", default=str(HERE / "data" / "flores.jsonl"))  # gen/score: 문장 파일 (대조 실험은 data/wmt20.jsonl)
    ap.add_argument("--instr", default="en", choices=["en", "ko"])  # gen: 지시문 언어
    args = ap.parse_args()
    {"gen": gen, "score": score, "online": online, "replay": replay, "costs": costs}[args.cmd](args)
