# E0 정합성 검사: greedy SD 출력이 vanilla 와 토큰 단위로 같은지 dtype 별로 확인
#   W02(MPS bf16)에서 같은 출력이 77% 뿐이었다. 알고리즘 버그가 아니라 bf16 수치(동점 뒤집힘) 때문이라면
#   fp32 에서는 100% 같아야 하고, bf16 에서 처음 갈라지는 위치는 target top-1/top-2 logit 차이가 거의 0 이어야 한다.
#   결과: <out>/runs.csv (한 줄 = SD 생성 1회), <out>/env.json
#
#   python e0_fp32.py                                       # CPU, target 1.7B / draft 0.6B, fp32 와 bf16
#   python e0_fp32.py --target models/Qwen3-8B --device mps --dtypes bfloat16   # W02 조건 재현
import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from bench import PROMPTS, short
from sd import generate, step


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


@torch.no_grad()
def tie_gap(model, ids):
    """ids 다음 토큰에 대한 target logits (캐시 없이 한 번에 forward) → (top1 - top2 차이, logits)"""
    lg = step(model, DynamicCache(), ids)[-1]
    top = lg.topk(2).values
    return (top[0] - top[1]).item(), lg


def main():
    sys.stdout.reconfigure(errors="replace")  # Windows 콘솔(cp949)에서 γ·— 등 출력 오류 방지
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="models/Qwen3-1.7B")
    ap.add_argument("--draft", default="models/Qwen3-0.6B")
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--tasks", nargs="+", default=list(PROMPTS))
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)  # CPU 스레드 수 (0 = torch 기본값)
    ap.add_argument("--out", default="results/e0")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.target)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return tok(text).input_ids

    env = dict(date=datetime.now().isoformat(timespec="seconds"), chip=platform.processor(), os=platform.platform(),
               python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
               torch_threads=torch.get_num_threads(), **{k: v for k, v in vars(args).items() if k != "out"})
    (out / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))

    fields = ["dtype", "task", "pid", "draft", "gamma", "n_ref", "n_sd", "same", "match_prefix",
              "gap", "ref_tok", "sd_tok", "sd_rank", "accepted", "rejects", "rounds"]
    f = open(out / "runs.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fields)
    w.writeheader()

    summary = {}
    for dt in args.dtypes:
        print(f"loading {args.target} + {args.draft} ({dt}, {dev})", flush=True)
        load = lambda n: AutoModelForCausalLM.from_pretrained(n, dtype=getattr(torch, dt)).to(dev).eval()
        target, draft = load(args.target), load(args.draft)
        for task in args.tasks:
            for pid, p in enumerate(PROMPTS[task][: args.n_prompts]):
                ids = encode(p)
                ref, _ = generate(target, ids, args.max_new, eos=eos)
                for g in args.gammas:
                    o, s = generate(target, ids, args.max_new, draft=draft, gamma=g, eos=eos)
                    mp = first_diff(o, ref)
                    row = dict(dtype=dt, task=task, pid=pid, draft=short(args.draft), gamma=g, n_ref=len(ref), n_sd=len(o), same=o == ref,
                               match_prefix=mp, accepted=s["accepted"], rejects=s["rejects"], rounds=s["rounds"])
                    if o != ref and mp < min(len(o), len(ref)):
                        # 갈라진 위치에서 target 이 두 토큰을 얼마나 구분하는지 (bf16 이면 0 근처일 것)
                        gap, lg = tie_gap(target, ids + ref[:mp])
                        row.update(gap=round(gap, 5), ref_tok=tok.decode([ref[mp]]), sd_tok=tok.decode([o[mp]]),
                                   sd_rank=int((lg > lg[o[mp]]).sum().item()))
                    w.writerow(row)
                    f.flush()
                    summary.setdefault((dt, g), []).append(row)
                    print(f"  {dt} {task}#{pid} γ={g}: same={o == ref}  prefix {mp}/{len(ref)}"
                          + (f"  gap {row['gap']:.4f} ({row['ref_tok']!r} vs {row['sd_tok']!r})" if "gap" in row else ""), flush=True)
        del target, draft
    f.close()

    print("\n| dtype | γ | vanilla와 출력 동일 | 갈라진 위치 top1-top2 gap (median) |")
    print("|---|---|---|---|")
    for (dt, g), rs in summary.items():
        gaps = sorted(r["gap"] for r in rs if "gap" in r)
        med = f"{gaps[len(gaps) // 2]:.4f}" if gaps else "-"
        print(f"| {dt} | {g} | {sum(r['same'] for r in rs)}/{len(rs)} | {med} |")
    print("saved", out)


if __name__ == "__main__":
    main()
