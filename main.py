import numpy as np
import time
import pwlf
import torch
import tiktoken

# input the any text to check the code execution
text = (
    "DFX : Low-latency FPGA Appliance for Accelerate Transformer based Text Generation"
)
enc = tiktoken.get_encoding("gpt2")
token_ids = enc.encode(text)
embedding_dim = 768
np_embeddings = np.random.randn(len(token_ids), embedding_dim).astype(np.float32)

# Compare with PyTorch LayerNorm
torch_input = torch.tensor(np_embeddings, dtype=torch.float32)
layer_norm = torch.nn.LayerNorm(embedding_dim)
normalized_torch = layer_norm(torch_input)
normalized_np = (np_embeddings - np_embeddings.mean(axis=-1, keepdims=True)) / np.sqrt(
    np.var(np_embeddings, axis=-1, keepdims=True) + 1e-5
)
diff = np.abs(normalized_np - normalized_torch.detach().numpy()).mean()
print("[PyTorch LN diff]:", diff)


# Variance calculation functions
def true_variance(x):
    mean = x.mean(axis=-1, keepdims=True)
    return ((x - mean) ** 2).mean(axis=-1, keepdims=True)


def one_pass_variance(x):
    mean = x.mean(axis=-1, keepdims=True)
    mean_sq = (x**2).mean(axis=-1, keepdims=True)
    return mean_sq - mean**2


def pairwise_variance(x):
    N = x.shape[-1]
    if N % 2 != 0:
        x = x[..., :-1]
        N -= 1
    x1, x2 = x[..., : N // 2], x[..., N // 2 :]
    mu1 = x1.mean(axis=-1, keepdims=True)
    mu2 = x2.mean(axis=-1, keepdims=True)
    var1 = ((x1 - mu1) ** 2).mean(axis=-1, keepdims=True)
    var2 = ((x2 - mu2) ** 2).mean(axis=-1, keepdims=True)
    delta = mu1 - mu2
    return var1 + var2 + (delta**2) * (N // 2) * (N // 2) / N


# PWL approximation function 8 stage
def pwl_approx(x, breakpoints, slopes, intercepts):
    x = np.clip(x, breakpoints[0], breakpoints[-1])
    out = np.zeros_like(x)
    for i in range(len(slopes)):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        out[mask] = slopes[i] * x[mask] + intercepts[i]
    out[x >= breakpoints[-1]] = slopes[-1] * x[x >= breakpoints[-1]] + intercepts[-1]
    return out


# Fit PWL for sqrt(x) and 1/sqrt(x)
x_vals = np.linspace(0.01, 128, 1000)
sqrt_vals = np.sqrt(x_vals)
recip_vals = 1 / sqrt_vals

sqrt_model = pwlf.PiecewiseLinFit(x_vals, sqrt_vals)
sqrt_breaks = sqrt_model.fit(8)
sqrt_slopes = sqrt_model.slopes
sqrt_intercepts = sqrt_model.intercepts

recip_model = pwlf.PiecewiseLinFit(x_vals, recip_vals)
recip_breaks = recip_model.fit(8)
recip_slopes = recip_model.slopes
recip_intercepts = recip_model.intercepts


# Timing function
def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000  # in ms


# Run tests on sample input
x = np.random.randn(10, 768).astype(np.float32)

true_var, t_true = measure_time(true_variance, x)
onepass_var, t_onepass = measure_time(one_pass_variance, x)
pairwise_var, t_pairwise = measure_time(pairwise_variance, x)

sqrt_result, t_sqrt = measure_time(
    pwl_approx, true_var + 1e-5, sqrt_breaks, sqrt_slopes, sqrt_intercepts
)
recip_result, t_recip = measure_time(
    pwl_approx, sqrt_result, recip_breaks, recip_slopes, recip_intercepts
)

# Output timing results
print(f"[True Variance]        {t_true:.4f} ms")
print(f"[One-pass Variance]    {t_onepass:.4f} ms")
print(f"[Pairwise Variance]    {t_pairwise:.4f} ms")
print(f"[PWL sqrt approx]      {t_sqrt:.4f} ms")
print(f"[PWL reciprocal approx]{t_recip:.4f} ms")
