# W05 · 3090 결과 "왜 GPU 가 맥북만큼 느린가" 진단 (spec-decode 폴더에서 실행)
#   SD_DEVICE=cuda .venv/bin/python /workspace/w05-3090/diag_cpu.py
# 재는 것 (모델마다, 1토큰 forward 기준)
#   1) wall      : 1토큰 forward 실제 시간 (bench.fwd_time 과 같은 방식, median)
#   2) gpu_busy  : 그중 GPU 가 실제로 계산한 시간 (profiler 의 CUDA 커널 시간 합)
#   3) launches  : CPU 가 GPU 에 보낸 커널 실행 명령 수 (cudaLaunchKernel 횟수)
#   4) gpu_util  : 측정 중 nvidia-smi 가 본 GPU 사용률(%)
#   5) 위 1) 을 torch CPU 스레드 수(기본/13/4/1)를 바꿔 가며 → 컨테이너 CPU 할당(13.3코어) 초과로 느려지는지
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
import lang  # noqa: E402
from bench import fwd_time  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = Path("results/lang_3090")
MODELS = {"Qwen3-8B": 36, "Qwen3-0.6B": 28, "Qwen3-1.7B": 28}  # 이름: 층 수 (HF config.json num_hidden_layers)


class GpuSampler:
    def __init__(self):
        self.vals, self.p = [], None

    def __enter__(self):
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits",
                                   "-lms", "100"], stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()
        return self

    def _read(self):
        for line in self.p.stdout:
            try:
                u, c = (float(x) for x in line.strip().split(","))
                self.vals.append((u, c))
            except ValueError:
                pass

    def __exit__(self, *a):
        self.p.terminate()
        self.t.join(timeout=2)

    def mean(self):
        if not self.vals:
            return None, None
        return round(statistics.fmean(v[0] for v in self.vals), 1), round(statistics.fmean(v[1] for v in self.vals))


def profile_one(model, ctx):
    from torch.profiler import ProfilerActivity, profile
    from sd import step
    from transformers import DynamicCache
    cache = DynamicCache()
    step(model, cache, ctx)
    L = cache.get_seq_length()
    for _ in range(3):  # 워밍업
        step(model, cache, ctx[-1:]); torch.cuda.synchronize(); lang_cut(cache, L)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(model, cache, ctx[-1:])
        torch.cuda.synchronize()
    lang_cut(cache, L)
    ev = prof.key_averages()
    launches = sum(e.count for e in ev if "LaunchKernel" in e.key)
    dev = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ev
              if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type))
    return launches, dev / 1e3  # ms


def lang_cut(cache, keep):
    from sd import cut
    cut(cache, keep)


def main():
    tok = AutoTokenizer.from_pretrained("models/Qwen3-8B")
    _, sents = lang.rows_with_prompts(OUT, tok)
    sid0 = sorted(random.Random(1).sample(sorted(sents), 20))[0]  # costs 와 같은 문장 (ctx_ko)
    ctx = lang.encode(tok, lang.prompt("ko", "en", sents[sid0]["ko"]))
    default_threads = torch.get_num_threads()
    res = dict(ctx_len=len(ctx), torch_default_threads=default_threads, models={})
    print(f"ctx_ko {len(ctx)} 토큰, torch 기본 스레드 {default_threads}", flush=True)
    for name, layers in MODELS.items():
        m = lang.load_model(f"models/{name}")
        r = dict(layers=layers, threads={})
        for nt in [default_threads, 13, 4, 1]:
            torch.set_num_threads(nt)
            fwd_time(m, ctx, 1, reps=3)
            c0 = time.process_time()
            with GpuSampler() as s:
                t = fwd_time(m, ctx, 1, reps=40)
            cpu_s = (time.process_time() - c0) / 42
            util, sm = s.mean()
            r["threads"][nt] = dict(wall_ms=round(t * 1e3, 2), cpu_ms_per_fwd=round(cpu_s * 1e3, 2), gpu_util=util, sm_mhz=sm)
            print(f"{name:12s} threads={nt:2d}  1토큰 {t * 1e3:6.2f} ms  CPU {cpu_s * 1e3:6.2f} ms  GPU 사용률 {util}%  SM {sm} MHz", flush=True)
        torch.set_num_threads(default_threads)
        launches, gpu_ms = profile_one(m, ctx)
        wall = r["threads"][default_threads]["wall_ms"]
        r.update(launches_per_fwd=launches, gpu_busy_ms=round(gpu_ms, 2), wall_ms=wall,
                 gpu_busy_ratio=round(gpu_ms / wall, 3), wall_ms_per_layer=round(wall / layers, 3),
                 us_per_launch=round((wall - gpu_ms) * 1e3 / launches, 1) if launches else None)
        print(f"{name:12s} 커널 실행 {launches}회 · GPU 실제 계산 {gpu_ms:.2f} ms / 전체 {wall:.2f} ms "
              f"({gpu_ms / wall * 100:.0f}%) · 층당 {wall / layers:.2f} ms", flush=True)
        res["models"][name] = r
        del m
        torch.cuda.empty_cache()
    t8 = res["models"]["Qwen3-8B"]["wall_ms"]
    res["c_measured"] = {n: round(v["wall_ms"] / t8, 3) for n, v in res["models"].items() if n != "Qwen3-8B"}
    res["c_if_layer_bound"] = {n: round(v["layers"] / 36, 3) for n, v in res["models"].items() if n != "Qwen3-8B"}
    res["c_if_byte_bound"] = {"Qwen3-0.6B": round(1.2 / 16.4, 3), "Qwen3-1.7B": round(4.1 / 16.4, 3)}
    (OUT / "diag_cpu.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("c_measured", "c_if_layer_bound", "c_if_byte_bound")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
