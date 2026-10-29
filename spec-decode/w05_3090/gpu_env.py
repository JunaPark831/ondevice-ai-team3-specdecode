# W05 · 기기 스펙 카드 (하드웨어-조건-맞추기.md §4) + 실효 메모리 대역폭
#   python gpu_env.py <저장폴더>   → <저장폴더>/env_card.json, env_raw.txt
# 공칭값이 아니라 "이 인스턴스에서 직접 잰 값"을 남기는 것이 목적이다.
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(실패: {e})"


def bandwidth(torch, gib=2, reps=20):
    """GPU 메모리 안에서 복사 (읽기 1 + 쓰기 1) → GB/s. 공칭 대역폭과 비교할 실효값"""
    n = gib * 2**30 // 2
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); b.copy_(a); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    ts.sort()
    t = ts[len(ts) // 2]
    del a, b
    torch.cuda.empty_cache()
    return 2 * n * 2 / t / 1e9


def pcie(torch, mib=512, reps=10):
    """CPU(pinned) → GPU 복사 GB/s. 모델 로딩·호스트 품질 확인용 (측정 구간 밖)"""
    h = torch.empty(mib * 2**20, dtype=torch.uint8).pin_memory()
    d = torch.empty_like(h, device="cuda")
    d.copy_(h); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); d.copy_(h, non_blocking=True); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort()
    return h.numel() / ts[len(ts) // 2] / 1e9


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers

    card = dict(
        date_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        vast=dict(container_label=os.environ.get("VAST_CONTAINERLABEL"), container_id=os.environ.get("CONTAINER_ID"),
                  public_ip=os.environ.get("PUBLIC_IPADDR")),
        gpu=sh("nvidia-smi --query-gpu=name,memory.total,driver_version,vbios_version,pcie.link.gen.max,pcie.link.width.max,"
               "power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader"),
        cpu=sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket' | sed 's/  */ /g'"),
        cpu_visible_to_container=os.cpu_count(),
        cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        ram=sh("free -g | head -2"),
        disk=sh("df -h . | tail -1"),
        os=platform.platform(), python=platform.python_version(),
        torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        transformers=transformers.__version__,
        cuda_available=torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        card.update(gpu_name=p.name, gpu_mem_gib=round(p.total_memory / 2**30, 2), sm_count=p.multi_processor_count,
                    capability=f"{p.major}.{p.minor}", bf16_supported=torch.cuda.is_bf16_supported())
        card["effective_bandwidth_GBps"] = round(bandwidth(torch), 1)
        card["pcie_h2d_GBps"] = round(pcie(torch), 2)
        card["other_gpu_processes"] = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    (out / "env_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    raw = "\n\n".join(f"### {c}\n{sh(c)}" for c in ("nvidia-smi", "nvidia-smi -q", "lscpu", "free -m", "df -h", "uname -a", "pip freeze"))
    (out / "env_raw.txt").write_text(raw)
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
