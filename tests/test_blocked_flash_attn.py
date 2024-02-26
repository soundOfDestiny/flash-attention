import torch
from torch.nn.functional import scaled_dot_product_attention
from flash_attn.flash_attn_interface import get_kvcache_block_size, flash_attn_with_blocked_kvcache, flash_attn_func


b, s, h_q, h_kv, d = 1, 131072, 16, 2, 128
block_size = get_kvcache_block_size(d)
dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item()


def timer(func):
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    e = 100
    for _ in range(e):
        func()
    en.record()
    torch.cuda.synchronize()
    t = st.elapsed_time(en) / e
    bytes = b * s * h_kv * d * 2 * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {bytes / 10**6 / t} GB/s")
    return t


@torch.inference_mode()
def test_flash_attention():
    s_q = 1
    q = torch.randn(b, s_q, h_q, d)
    k = torch.randn(b, s, h_kv, d)
    v = torch.randn(b, s, h_kv, d)
    full_k = k[:, :, :, None, :].expand(b, s, h_kv, h_q // h_kv, d).reshape(b, s, h_q, d)
    full_v = v[:, :, :, None, :].expand(b, s, h_kv, h_q // h_kv, d).reshape(b, s, h_q, d)
    blocked_k = k.view(-1, block_size, h_kv, d)
    blocked_v = v.view(-1, block_size, h_kv, d)
    block_table = torch.arange(b * s // block_size, dtype=torch.int32).view(b, s // block_size)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32)

    alibi_slopes = torch.rand(h_q, dtype=torch.float32)
    alibi_exps = torch.rand(h_q, dtype=torch.float32)
    alibi_mask = (torch.arange(s)[None, :] - torch.arange(s_q)[:, None] - (s - s_q))[None, None, :, :].float().abs()
    alibi_mask = -alibi_mask.pow(alibi_exps[None, :, None, None]) * alibi_slopes[None, :, None, None]
    causal_mask = ~torch.ones(s_q, s, dtype=torch.bool).tril(diagonal=s-s_q)
    mask = alibi_mask.masked_fill(causal_mask, torch.finfo(torch.float32).min)

    # Warm up
    for _ in range(100):
        torch.ones(1 << 20)

    def blocked_flash(): return flash_attn_with_blocked_kvcache(q, blocked_k, blocked_v, block_table, cache_seqlens, causal=True, alibi_slopes=alibi_slopes, alibi_exps=alibi_exps)
    def flash(): return flash_attn_func(q, k, v, causal=True, alibi_slopes=alibi_slopes, alibi_exps=alibi_exps)
    def torch_attn(): return scaled_dot_product_attention(q.transpose(1, 2), full_k.transpose(1, 2), full_v.transpose(1, 2), attn_mask=mask.to(dtype)).transpose(1, 2)
    def ref(): return scaled_dot_product_attention(q.transpose(1, 2).double(), full_k.transpose(1, 2).double(), full_v.transpose(1, 2).double(), attn_mask=mask.double()).transpose(1, 2)

    timer(blocked_flash)
    timer(blocked_flash)

    out_blocked_flash = blocked_flash()
    out_flash = flash()
    out_torch_attn = torch_attn()
    out_ref = ref()
    print("blocked flash diff:", calc_diff(out_blocked_flash, out_ref), (out_blocked_flash - out_ref).abs().max().item())
    print("flash diff:", calc_diff(out_flash, out_ref), (out_flash - out_ref).abs().max().item())
    print("torch_attn diff:", calc_diff(out_torch_attn, out_ref), (out_torch_attn - out_ref).abs().max().item())
    assert calc_diff(out_blocked_flash, out_ref) <= calc_diff(out_torch_attn, out_ref)
    assert (out_blocked_flash - out_ref).abs().max().item() <= (out_torch_attn - out_ref).abs().max().item()


if __name__ == "__main__":
    test_flash_attention()
