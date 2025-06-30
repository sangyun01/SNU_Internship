import numpy as np
import time
import pwlf
import torch

# Generate input
N, embedding_dim = 100, 768
np_embeddings = np.random.randn(N, embedding_dim).astype(np.float32)
torch_input = torch.tensor(np_embeddings, dtype=torch.float32)


# Variance methods
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


# PWL fit
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


# PWL approximation
def pwl_approx(x, breakpoints, slopes, intercepts):
    x = np.clip(x, breakpoints[0], breakpoints[-1])
    out = np.zeros_like(x)
    for i in range(len(slopes)):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        out[mask] = slopes[i] * x[mask] + intercepts[i]
    out[x >= breakpoints[-1]] = slopes[-1] * x[x >= breakpoints[-1]] + intercepts[-1]
    return out


# Timer
def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000


# Measure variance
x = np_embeddings
true_var, t_true = measure_time(true_variance, x)
onepass_var, t_onepass = measure_time(one_pass_variance, x)
pairwise_var, t_pairwise = measure_time(pairwise_variance, x)

# PyTorch variance (ground truth)
with torch.no_grad():
    var_torch = torch.var(torch_input, dim=-1, unbiased=False, keepdim=True).numpy()

# Accuracy (% error)
err_true = np.abs(true_var - var_torch) / (var_torch + 1e-8) * 100
err_one = np.abs(onepass_var - var_torch) / (var_torch + 1e-8) * 100
err_pair = np.abs(pairwise_var - var_torch) / (var_torch + 1e-8) * 100

# Sqrt & reciprocal comparisons
sqrt_exact, t_sqrt_exact = measure_time(np.sqrt, true_var + 1e-5)
sqrt_pwl, t_sqrt = measure_time(
    pwl_approx, true_var + 1e-5, sqrt_breaks, sqrt_slopes, sqrt_intercepts
)

recip_exact, t_recip_exact = measure_time(np.reciprocal, sqrt_exact)
recip_pwl, t_recip = measure_time(
    pwl_approx, sqrt_pwl, recip_breaks, recip_slopes, recip_intercepts
)

err_sqrt = np.abs(sqrt_exact - sqrt_pwl) / (sqrt_exact + 1e-8) * 100
err_recip = np.abs(recip_exact - recip_pwl) / (recip_exact + 1e-8) * 100

# Print results
print("===== Accuracy (% Error vs PyTorch) =====")
print(f"[True Var]     {err_true.mean():.4f}%")
print(f"[One-Pass Var] {err_one.mean():.4f}%")
print(f"[Pairwise Var] {err_pair.mean():.4f}%")
print(f"[Sqrt PWL]     {err_sqrt.mean():.4f}%")
print(f"[Recip PWL]    {err_recip.mean():.4f}%")

print("\n===== Timing (ms) =====")
print(f"[True Var]        {t_true:.4f} ms")
print(f"[One-Pass Var]    {t_onepass:.4f} ms")
print(f"[Pairwise Var]    {t_pairwise:.4f} ms")
print(f"[Sqrt Exact]      {t_sqrt_exact:.4f} ms")
print(f"[Sqrt PWL]        {t_sqrt:.4f} ms")
print(f"[Recip Exact]     {t_recip_exact:.4f} ms")
print(f"[Recip PWL]       {t_recip:.4f} ms")
