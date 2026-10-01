# 노트북(MPS)에서 speculative decoding 속도 측정
#   draft 크기 × γ × 태스크 sweep 을 vanilla(target 혼자) 와 같은 프롬프트로 비교한다.
#   결과: <out>/runs.csv (한 줄 = 생성 1회), <out>/costs.json (forward 비용), <out>/env.json
import argparse
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from sd import cut, generate, step, sync

EN2KO = "Translate the following English text into Korean. Output only the translation.\n\n"
KO2EN = "다음 한국어 글을 영어로 번역하세요. 번역문만 출력하세요.\n\n"
PROMPTS = {
    "en_write": [
        "Explain how a refrigerator keeps food cold, in about 150 words.",
        "Describe the main causes of the French Revolution in one paragraph.",
        "Write a short story about a robot learning to paint.",
    ],
    "code": [
        "Write a Python function that checks whether a string is a palindrome, with comments and a few test cases.",
        "Implement binary search in Python and explain its time complexity.",
        "Write a Python class for a simple LRU cache with get and put methods.",
    ],
    "en2ko": [
        EN2KO + "The city council approved a new plan to expand public transportation. Starting next year, buses will run every ten minutes during rush hour, and two new subway lines will connect the northern suburbs to downtown. Officials say the project will reduce traffic congestion and air pollution.",
        EN2KO + "Sleep plays a critical role in memory. During deep sleep, the brain replays experiences from the day and strengthens the connections between neurons. Researchers have found that students who sleep well after studying remember more than those who stay up all night.",
        EN2KO + "Small businesses are increasingly using artificial intelligence to handle customer service. Chatbots can answer common questions at any time of day, which allows employees to focus on more complex problems. However, many customers still prefer to talk to a real person.",
    ],
    "ko2en": [
        KO2EN + "서울시는 내년부터 한강 공원에 자전거 전용 도로를 확대하기로 했다. 시 관계자는 주말마다 많은 시민들이 공원을 찾으면서 보행자와 자전거 사이의 사고가 늘어났다고 설명했다. 새 도로는 보행로와 완전히 분리되어 설치될 예정이다.",
        KO2EN + "커피를 하루에 두세 잔 마시는 것은 대부분의 성인에게 건강에 큰 문제가 되지 않는다. 하지만 늦은 오후에 마시는 커피는 수면의 질을 떨어뜨릴 수 있다. 전문가들은 잠들기 최소 여섯 시간 전에는 카페인 섭취를 피하라고 조언한다.",
        KO2EN + "최근 많은 대학생들이 졸업 후 바로 취업하기보다 창업을 선택하고 있다. 정부는 청년 창업을 지원하기 위해 다양한 프로그램을 운영하고 있지만, 실제로 성공하는 경우는 많지 않다. 전문가들은 충분한 시장 조사가 무엇보다 중요하다고 강조한다.",
    ],
}

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()


def short(name):
    return name.split("/")[-1]


def fwd_time(model, ctx, k, reps=10):
    """캐시에 ctx 가 들어있는 상태에서 k 토큰 forward 1번 걸리는 시간 (median)"""
    cache = DynamicCache()
    step(model, cache, ctx)
    L = cache.get_seq_length()
    ts = []
    for _ in range(reps + 2):
        sync(DEV)
        t = time.perf_counter()
        step(model, cache, ctx[-k:])
        sync(DEV)
        ts.append(time.perf_counter() - t)
        cut(cache, L)
    return statistics.median(ts[2:])  # 앞 2번은 워밍업


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="models/Qwen3-8B")
    ap.add_argument("--drafts", nargs="+", default=["models/Qwen3-0.6B", "models/Qwen3-1.7B"])
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 12])
    ap.add_argument("--tasks", nargs="+", default=list(PROMPTS))
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--first-rep", type=int, default=0)  # 일부 반복만 다시 돌릴 때
    ap.add_argument("--out", default="results/main")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.target)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return tok(text).input_ids

    print("loading", args.target, *args.drafts)
    target = load(args.target)
    drafts = {short(d): (target if d == args.target else load(d)) for d in args.drafts}

    env = dict(date=datetime.now().isoformat(timespec="seconds"), device=str(DEV),
               chip=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
               mem_gb=round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30),
               macos=platform.mac_ver()[0], python=platform.python_version(),
               torch=torch.__version__, transformers=transformers.__version__,
               dtype="bfloat16", **{k: v for k, v in vars(args).items() if k != "out"})
    (out / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))

    # 1) forward 비용: target 은 k=1..13 토큰(검증 비용), draft 는 1토큰 → c = T_d(1) / T_t(1)
    ctx = encode(PROMPTS["en2ko"][0])
    costs = {"ctx_len": len(ctx), "target": {k: fwd_time(target, ctx, k) for k in range(1, max(args.gammas) + 2)}}
    costs["drafts"] = {n: fwd_time(m, ctx, 1) for n, m in drafts.items()}
    (out / "costs.json").write_text(json.dumps(costs, indent=2))
    print("T_target(1) = %.1f ms" % (costs["target"][1] * 1e3),
          *("c[%s] = %.3f" % (n, t / costs["target"][1]) for n, t in costs["drafts"].items()))

    # 워밍업 (MPS 커널 컴파일 등)
    for m in drafts.values():
        generate(target, encode("Hi"), 16, draft=m, gamma=4, eos=eos)

    fields = ["rep", "task", "pid", "draft", "gamma", "n_new", "ttft", "total", "decode_tps",
              "rounds", "drafted", "accepted", "rejects", "t_draft", "t_verify", "same", "match_prefix", "load"]
    f = open(out / "runs.csv", "w", newline="")
    w = csv.DictWriter(f, fields)
    w.writeheader()

    configs = [(n, g) for n in drafts for g in args.gammas]
    total_runs = args.reps * len(args.tasks) * args.n_prompts * (len(configs) + 2)
    done, t_start = 0, time.time()
    for rep in range(args.first_rep, args.first_rep + args.reps):
        for task in args.tasks:
            for pid, p in enumerate(PROMPTS[task][: args.n_prompts]):
                ids = encode(p)
                load_avg = os.getloadavg()[0]  # 다른 프로세스가 끼어들었는지 확인용
                ref, st = generate(target, ids, args.max_new, eos=eos)
                runs = [("vanilla", 0, ref, st)]
                order = configs[:]
                random.Random(rep * 1000 + pid).shuffle(order)  # 측정 순서와 γ 가 겹치지 않게 (발열 등)
                for n, g in order:
                    o, s = generate(target, ids, args.max_new, draft=drafts[n], gamma=g, eos=eos)
                    runs.append((n, g, o, s))
                # 드리프트 검사: 블록 끝에 vanilla 한 번 더. 앞뒤가 10% 넘게 다르면 plot.py 에서 블록 제외
                o, s = generate(target, ids, args.max_new, eos=eos)
                runs.append(("vanilla_after", 0, o, s))
                for n, g, o, s in runs:
                    mp = next((i for i, (a, b) in enumerate(zip(o, ref)) if a != b), min(len(o), len(ref)))
                    w.writerow(dict(rep=rep, task=task, pid=pid, draft=n, gamma=g, same=o == ref, match_prefix=mp, load=round(load_avg, 1),
                                    decode_tps=(s["n_new"] - 1) / (s["total"] - s["ttft"]),
                                    **{k: s[k] for k in fields if k in s}))
                f.flush()
                with open(out / "outputs.jsonl", "a") as fo:  # vanilla 출력 원문 (번역 확인, 토큰당 바이트 계산용)
                    fo.write(json.dumps(dict(rep=rep, task=task, pid=pid, n_tok=len(ref), text=tok.decode(ref, skip_special_tokens=True)), ensure_ascii=False) + "\n")
                done += len(runs)
                el = time.time() - t_start
                tps = [(r[3]["n_new"] - 1) / (r[3]["total"] - r[3]["ttft"]) for r in (runs[0], runs[-1])]
                print(f"[{done}/{total_runs}] rep{rep} {task}#{pid}  vanilla {tps[0]:.1f}→{tps[1]:.1f} tok/s  load {load_avg:.0f}  "
                      f"{el / 60:.1f} min, 남은 시간 ~{el / done * (total_runs - done) / 60:.0f} min", flush=True)
    f.close()
    print("saved", out)


if __name__ == "__main__":
    main()
