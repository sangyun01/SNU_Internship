import numpy as np
import torch
import pwlf

# === 입력 생성 ===
N, D = 96, 768
np_embeddings_f = np.random.randn(N, D).astype(np.float32)
torch_input = torch.tensor(np_embeddings_f, dtype=torch.float32)


# === Q8.8 변환 함수 ===
def float_to_q8_8(x):
    return np.round(x * 256).astype(np.int16)


def q8_8_to_float(x_q):
    return x_q.astype(np.float32) / 256


np_embeddings_q8_8 = float_to_q8_8(np_embeddings_f).astype(np.int32)


# === Q8.8 기반 pairwise 분산 계산 ===
def pairwise_variance_q8_8(x_q, G=16):
    D = x_q.shape[-1]
    splits = np.split(x_q, G, axis=-1)
    n_list = [s.shape[-1] for s in splits]
    mu_list = [np.sum(s, axis=-1, keepdims=True) // s.shape[-1] for s in splits]
    M_list = [
        np.sum(((s - mu).astype(np.int64)) ** 2, axis=-1, keepdims=True)
        for s, mu in zip(splits, mu_list)
    ]
    while len(mu_list) > 1:
        next_mu, next_M, next_n = [], [], []
        for i in range(0, len(mu_list), 2):
            μ1, μ2 = mu_list[i], mu_list[i + 1]
            M1, M2 = M_list[i], M_list[i + 1]
            n1, n2 = n_list[i], n_list[i + 1]
            n_total = n1 + n2
            delta = μ1 - μ2
            delta_sq = (delta.astype(np.int64)) ** 2
            delta_term = (delta_sq * n1 * n2) // n_total
            M12 = M1 + M2 + delta_term
            mu12 = (μ1 * n1 + μ2 * n2) // n_total
            next_mu.append(mu12)
            next_M.append(M12)
            next_n.append(n_total)
        mu_list, M_list, n_list = next_mu, next_M, next_n
    var_q16 = M_list[0] // D
    return q8_8_to_float(var_q16 // 256)


# === PWL 모델 준비 (sqrt, reciprocal) ===
x_vals = np.linspace(0.01, 64, 1000)
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


# === PWL 근사 함수 ===
def pwl_approx(x, breakpoints, slopes, intercepts):
    x = np.clip(x, breakpoints[0], breakpoints[-1])
    out = np.zeros_like(x)
    for i in range(len(slopes)):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        out[mask] = slopes[i] * x[mask] + intercepts[i]
    out[x >= breakpoints[-1]] = slopes[-1] * x[x >= breakpoints[-1]] + intercepts[-1]
    return out


# === 최종 근사 LayerNorm ===
def approx_layernorm_q8_8(x_q, eps=1e-5):
    mu_q = np.mean(x_q, axis=-1, keepdims=True)
    x_centered = x_q - mu_q
    var_approx = pairwise_variance_q8_8(x_q)  # float
    sqrt_val = pwl_approx(var_approx + eps, sqrt_breaks, sqrt_slopes, sqrt_intercepts)
    recip_val = pwl_approx(sqrt_val, recip_breaks, recip_slopes, recip_intercepts)
    x_float = q8_8_to_float(x_centered)
    return x_float * recip_val


# === 단계별 비교 ===

# 1. Pairwise + 정확 sqrt + 정확 reciprocal
var_pairwise = pairwise_variance_q8_8(np_embeddings_q8_8)
sqrt_exact = np.sqrt(var_pairwise + 1e-5)
recip_exact = 1.0 / sqrt_exact
mu_q = np.mean(np_embeddings_q8_8, axis=-1, keepdims=True)
Y_exact = q8_8_to_float(np_embeddings_q8_8 - mu_q) * recip_exact

# 2. Pairwise + PWL sqrt + 정확 reciprocal
sqrt_pwl = pwl_approx(var_pairwise + 1e-5, sqrt_breaks, sqrt_slopes, sqrt_intercepts)
recip_exact2 = 1.0 / sqrt_pwl
Y_pwl_sqrt = q8_8_to_float(np_embeddings_q8_8 - mu_q) * recip_exact2

# 3. Pairwise + PWL sqrt + PWL reciprocal (최종 구조)
recip_pwl = pwl_approx(sqrt_pwl, recip_breaks, recip_slopes, recip_intercepts)
Y_pwl_full = q8_8_to_float(np_embeddings_q8_8 - mu_q) * recip_pwl

# === 기준값: PyTorch LayerNorm ===
layernorm_torch = torch.nn.LayerNorm(D, elementwise_affine=False)
Y_true = layernorm_torch(torch_input).numpy()

# === 근사 계산값 ===
Y_approx = approx_layernorm_q8_8(np_embeddings_q8_8)

# === 정확도 측정 ===
mae = np.mean(np.abs(Y_true - Y_approx))
relative_acc = 100 - (mae / (np.abs(Y_true).mean() + 1e-8)) * 100

print("===== LayerNorm Approximation =====")
print(f"Relative Accuracy:    {relative_acc:.4f}%")

# 기준값
Y_true = layernorm_torch(torch_input).numpy()


# 정확도 비교
def acc(y_hat):
    return (
        100 - (np.mean(np.abs(Y_true - y_hat)) / (np.mean(np.abs(Y_true)) + 1e-8)) * 100
    )


def mae(y_hat):
    return np.mean(np.abs(Y_true - y_hat))


print("===== 단계별 정확도 비교 =====")
print(f"[1] Pairwise + Exact Sqrt/Recip   → Accuracy: {acc(Y_exact):.4f}%")
print(f"[2] Pairwise + PWL Sqrt + Exact   → Accuracy: {acc(Y_pwl_sqrt):.4f}%")
print(f"[3] Pairwise + PWL Sqrt + Recip   → Accuracy: {acc(Y_pwl_full):.4f}%")
