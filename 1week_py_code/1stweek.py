import numpy as np
import pwlf
import torch

# Input variance tensor
N, D = 100, 768
np_embeddings = np.random.randn(N, D).astype(np.float32)
torch_input = torch.tensor(np_embeddings, dtype=torch.float32)

# Generate data
x_vals = np.linspace(0.01, 128, 1000)
sqrt_vals = np.sqrt(x_vals)
recip_vals = 1 / sqrt_vals

# linear model (sqrt)
sqrt_model = pwlf.PiecewiseLinFit(x_vals, sqrt_vals)
sqrt_breaks = sqrt_model.fit(8)
sqrt_slopes = sqrt_model.slopes
sqrt_intercepts = sqrt_model.intercepts

# linear model (recip)
recip_model = pwlf.PiecewiseLinFit(x_vals, recip_vals)
recip_breaks = recip_model.fit(8)
recip_slopes = recip_model.slopes
recip_intercepts = recip_model.intercepts


# PWL approximation function
def pwl_approx(x, breakpoints, slopes, intercepts):
    x = np.clip(x, breakpoints[0], breakpoints[-1])
    out = np.zeros_like(x)
    for i in range(len(slopes)):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        out[mask] = slopes[i] * x[mask] + intercepts[i]
    out[x >= breakpoints[-1]] = slopes[-1] * x[x >= breakpoints[-1]] + intercepts[-1]
    return out


# Ground-truth variance
with torch.no_grad():
    true_var = torch.var(torch_input, dim=-1, unbiased=False, keepdim=True).numpy()

# Sqrt and Recip: Exact vs PWL
sqrt_exact = np.sqrt(true_var + 1e-5)
sqrt_pwl = pwl_approx(true_var + 1e-5, sqrt_breaks, sqrt_slopes, sqrt_intercepts)

recip_exact = np.reciprocal(sqrt_exact)
recip_pwl = pwl_approx(sqrt_pwl, recip_breaks, recip_slopes, recip_intercepts)

# Accuracy
acc_sqrt = 100 - np.abs(sqrt_exact - sqrt_pwl) / (sqrt_exact + 1e-8) * 100
acc_recip = 100 - np.abs(recip_exact - recip_pwl) / (recip_exact + 1e-8) * 100

# Print results
print("===== Accuracy (% Error) =====")
print(f"[Sqrt PWL]     Mean Accuracy: {acc_sqrt.mean():.4f}%")
print(f"[Recip PWL]    Mean Accuracy: {acc_recip.mean():.4f}%")
