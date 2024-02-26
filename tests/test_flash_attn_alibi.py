import torch
from torch.nn.functional import scaled_dot_product_attention
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func


b, s, h_q, h_kv, d = 1, 4096, 56, 56, 128
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
    FLOPS = b * s * s * h_kv * d * 2 * 6

    print(f"{t} ms, {FLOPS / 10**9 / t} tflops")
    return t


def test_flash_attention():
    qkv = torch.randn(b, s, 3, h_q, d)
    qkv1 = qkv.clone().requires_grad_()
    qkv2 = qkv.clone().requires_grad_()
    cu_seqlens = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=qkv.device)

    alibi_slopes = torch.rand(h_q, dtype=torch.float32)
    alibi_slopes1 = alibi_slopes.clone().requires_grad_()
    alibi_slopes2 = alibi_slopes.clone().requires_grad_()
    alibi_exps = torch.rand(h_q, dtype=torch.float32)
    alibi_exps1 = alibi_exps.clone().requires_grad_()
    alibi_exps2 = alibi_exps.clone().requires_grad_()
    alibi_mask_ = (torch.arange(s)[None, :] - torch.arange(s)[:, None])[None, None, :, :].float().abs()
    alibi_mask = -alibi_mask_.pow(alibi_exps2[None, :, None, None]) * alibi_slopes2[None, :, None, None]
    causal_mask = ~torch.ones(s, s, dtype=torch.bool).tril()
    mask = alibi_mask.masked_fill(causal_mask, torch.finfo(torch.float32).min)

    # Warm up
    for _ in range(100):
        torch.ones(1 << 20)

    def flash(): return flash_attn_varlen_qkvpacked_func(qkv1.view(b * s, 3, h_q, d), cu_seqlens, s, causal=True, alibi_slopes=alibi_slopes1, alibi_exps=alibi_exps1)
    def torch_attn(): return scaled_dot_product_attention(qkv2[:, :, 0].transpose(1, 2).float(), qkv2[:, :, 1].transpose(1, 2).float(), qkv2[:, :, 2].transpose(1, 2).float(), attn_mask=mask).transpose(1, 2)

    out_flash = flash()
    out_flash.sum().backward()
    out_torch_attn = torch_attn()
    out_torch_attn.sum().backward()
    print("flash diff:", calc_diff(out_flash, out_torch_attn), (out_flash - out_torch_attn).abs().max().item())
    print("flash_grad diff:", calc_diff(qkv1.grad, qkv2.grad), (qkv1.grad - qkv2.grad).abs().max().item())
    print("slopes_grad diff:", calc_diff(alibi_slopes1.grad, alibi_slopes2.grad), (alibi_slopes1.grad - alibi_slopes2.grad).abs().max().item())
    print("exps_grad diff:", calc_diff(alibi_exps1.grad, alibi_exps2.grad), (alibi_exps1.grad - alibi_exps2.grad).abs().max().item())

    timer(lambda: flash().sum().backward())
    timer(lambda: flash().sum().backward())


if __name__ == "__main__":
    test_flash_attention()
