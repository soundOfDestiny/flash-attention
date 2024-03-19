import torch
from torch.nn.functional import scaled_dot_product_attention

from flash_attn import flash_attn_varlen_qkvpacked_func


b, s, h, d = 1, 4096, 56, 128
dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator < 1e-12:
        return 0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item()


def assert_close(x, y, name=""):
    diff = calc_diff(x, y)
    amax = (x - y).abs().max()
    print(f"{name} diff: {diff}, amax: {amax}")
    assert diff < 1e-5


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
    FLOPS = b * s * s * h * d * 2 * 6
    bytes = b * s * h * d * 2 * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10**9 / t} tflops, {bytes / 10**6 / t} GB/s")
    return t


def test_flash_attention(b, s, h, d):
    print(b, h, s, d)

    qkv = torch.randn(b, s, 3, h, d)
    alibi_slopes = torch.rand(h, dtype=torch.float32)

    qkv1 = qkv.clone().requires_grad_()
    alibi_slopes1 = alibi_slopes.clone().requires_grad_()
    cu_seqlens = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32)

    qkv2 = qkv.clone().requires_grad_()
    q2, k2, v2 = qkv2.transpose(1, 3).chunk(3, dim=2)
    q2, k2, v2 = q2.squeeze(2), k2.squeeze(2), v2.squeeze(2)  # [b, s, h, d]
    alibi_slopes2 = alibi_slopes.clone().requires_grad_()
    dis = (torch.arange(s)[None, :] - torch.arange(s)[:, None])[None, None, :, :].float().abs()
    alibi_mask = -dis * alibi_slopes2[None, :, None, None]
    causal_mask = ~torch.ones(s, s, dtype=torch.bool).tril()
    mask = alibi_mask.masked_fill(causal_mask, torch.finfo(torch.float32).min)

    def flash_attn(): return flash_attn_varlen_qkvpacked_func(qkv1.view(b * s, 3, h, d), cu_seqlens, s, causal=True, alibi_slopes=alibi_slopes1).view(b, s, h, d)
    def torch_attn(): return scaled_dot_product_attention(q2.float(), k2.float(), v2.float(), attn_mask=mask).transpose(1, 2)

    out_flash_attn = flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out")

    grad_triton = torch.randn_like(out_flash_attn)
    out_flash_attn.backward(grad_triton)
    out_torch_attn.backward(grad_triton)
    assert_close(qkv1.grad[:, :, 0], qkv2.grad[:, :, 0], "dq")
    assert_close(qkv1.grad[:, :, 1], qkv2.grad[:, :, 1], "dk")
    assert_close(qkv1.grad[:, :, 2], qkv2.grad[:, :, 2], "dv")
    assert_close(alibi_slopes1.grad, alibi_slopes2.grad, "dslopes")

    grad_triton = torch.randn_like(out_flash_attn)
    timer(lambda: flash_attn().backward(grad_triton))
    timer(lambda: flash_attn().backward(grad_triton))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn().backward(grad_triton)
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100))


if __name__ == "__main__":
    test_flash_attention(b, s, h, d)
