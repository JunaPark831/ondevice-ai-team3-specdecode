# Speculative decoding 직접 구현 (Leviathan et al., ICML 2023, Algorithm 1)
#   temperature == 0 : greedy 검증. draft 토큰이 target argmax 와 같으면 수락
#   temperature >  0 : rejection sampling. min(1, p/q) 로 수락, 거절되면 max(0, p-q) 에서 다시 뽑음
#                      → 출력 분포가 target 분포와 같다 (맨 아래 self-check 로 확인)
#
# KV 캐시 규칙: 매 라운드 시작 시 "캐시 = seq[:-1]" (마지막 토큰은 아직 안 넣은 상태)
#   target 은 draft 토큰까지 넣어서 검증한 뒤, 수락된 데까지만 남기고 crop 한다.
import time

import torch
from transformers import DynamicCache


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def step(model, cache, ids):
    """ids(list[int]) 를 캐시 뒤에 이어 붙여 forward → 각 위치의 logits [len(ids), V]"""
    x = torch.tensor([ids], device=model.device)
    return model(input_ids=x, past_key_values=cache, use_cache=True).logits[0].float()


def cut(cache, keep):
    """캐시를 앞에서부터 keep 토큰만 남긴다 (crop 은 음수 = 뒤에서 몇 개 지울지)"""
    cache.crop(min(0, keep - cache.get_seq_length()))


def pick(logits, temperature):
    if temperature == 0:
        return logits.argmax(-1).item()
    return torch.multinomial(torch.softmax(logits / temperature, -1), 1).item()


def verify(t_logits, drafts, d_logits, temperature):
    """t_logits [γ+1, V], drafts list[γ], d_logits [γ, V]
    반환: 이번 라운드에 확정되는 토큰 = 수락된 draft n개 + target 이 정한 1개"""
    g = len(drafts)
    if temperature == 0:
        best = t_logits.argmax(-1).tolist()
        n = 0
        while n < g and drafts[n] == best[n]:
            n += 1
        return drafts[:n] + [best[n]]

    p = torch.softmax(t_logits / temperature, -1)
    q = torch.softmax(d_logits / temperature, -1)
    for i, x in enumerate(drafts):
        if torch.rand(()).item() < (p[i, x] / q[i, x]).item():  # min(1, p/q)
            continue
        resid = (p[i] - q[i]).clamp(min=0)
        return drafts[:i] + [torch.multinomial(resid / resid.sum(), 1).item()]
    return drafts + [torch.multinomial(p[g], 1).item()]  # 전부 수락 → 보너스 토큰 1개


@torch.no_grad()
def generate(target, prompt, max_new, draft=None, gamma=4, temperature=0.0, eos=()):
    """draft=None 이면 vanilla (target 만 한 토큰씩). 아니면 speculative decoding."""
    dev = target.device
    seq = list(prompt)
    t_cache = DynamicCache()
    d_cache = DynamicCache()
    st = dict(rounds=0, drafted=0, accepted=0, rejects=0, t_draft=0.0, t_verify=0.0)

    sync(dev)
    t0 = time.perf_counter()
    seq.append(pick(step(target, t_cache, prompt)[-1], temperature))  # prefill → 첫 토큰
    if draft is not None:
        step(draft, d_cache, prompt)
    sync(dev)
    ttft = time.perf_counter() - t0

    while len(seq) - len(prompt) < max_new and seq[-1] not in eos:
        if draft is None:
            seq.append(pick(step(target, t_cache, seq[-1:])[-1], temperature))
            st["rounds"] += 1
            continue

        # 1) draft 가 γ 개를 한 개씩 뽑는다 (직전 라운드에 전부 수락됐으면 2토큰부터 넣음)
        ta = time.perf_counter()
        feed = seq[d_cache.get_seq_length():]
        drafts, d_logits = [], []
        for _ in range(gamma):
            lg = step(draft, d_cache, feed)[-1]
            drafts.append(pick(lg, temperature))
            d_logits.append(lg)
            feed = drafts[-1:]

        # 2) target 이 [캐시에 없는 토큰 + draft γ개] 를 forward 한 번으로 검증
        tb = time.perf_counter()
        t_logits = step(target, t_cache, seq[t_cache.get_seq_length():] + drafts)[-(gamma + 1):]
        new = verify(t_logits, drafts, torch.stack(d_logits), temperature)
        tc = time.perf_counter()

        n = len(new) - 1
        st["rounds"] += 1
        st["drafted"] += gamma
        st["accepted"] += n
        st["rejects"] += n < gamma
        st["t_draft"] += tb - ta
        st["t_verify"] += tc - tb

        # 3) 확정된 토큰까지만 캐시를 남긴다
        seq += new
        cut(t_cache, len(seq) - 1)
        cut(d_cache, len(seq) - 1)
        if any(t in eos for t in new):
            break

    sync(dev)
    total = time.perf_counter() - t0

    out = seq[len(prompt):][:max_new]
    for i, t in enumerate(out):
        if t in eos:
            out = out[: i + 1]
            break
    return out, dict(ttft=ttft, total=total, n_new=len(out), **st)


if __name__ == "__main__":
    # self-check 1: greedy 검증 — 두 번째 draft 에서 어긋나면 [d0, target 의 답]
    t = torch.full((4, 5), -9.0)
    for i, b in enumerate([1, 2, 3, 4]):
        t[i, b] = 0
    assert verify(t, [1, 0, 3], None, 0) == [1, 2]
    assert verify(t, [1, 2, 3], None, 0) == [1, 2, 3, 4]  # 전부 수락 + 보너스

    # self-check 2: rejection sampling 출력 분포 == target 분포 (draft 분포와 상관없이)
    torch.manual_seed(0)
    V, N = 6, 40_000
    t_logits, d_logits = torch.randn(2, V) * 2, torch.randn(1, V) * 2
    p, q = torch.softmax(t_logits[0], -1), torch.softmax(d_logits[0], -1)
    cnt = torch.zeros(V)
    for _ in range(N):
        x = torch.multinomial(q, 1).item()
        cnt[verify(t_logits, [x], d_logits, 1.0)[0]] += 1
    tv = 0.5 * (cnt / N - p).abs().sum().item()
    tv_q = 0.5 * (q - p).abs().sum().item()
    print(f"TV(output, target) = {tv:.4f}   (참고: TV(draft, target) = {tv_q:.4f})")
    assert tv < 0.02
    print("ok")
