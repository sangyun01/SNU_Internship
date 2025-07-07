import numpy as np
import torch

# === Input Configuration ===
D = 768  # Dimension

# === Mapping user input (1~6) to corresponding file names ===
file_map = {
    "1": "bert_sst2_embed.npy",
    "2": "bert_qnli_embed.npy",
    "3": "bert_mnli_embed.npy",
    "4": "bert_sst2_embed_100.npy",
    "5": "distilbert_sst2_embed.npy",
    "6": "distilbert_qnli_embed.npy",
    "7": "distilbert_mnli_embed.npy",
    "8": "distilbert_sst2_embed_100.npy",
}

# === Get user selection ===
choice = input("Select input (1~8): ").strip()

if choice in file_map:
    file_name = file_map[choice]
    print(f"Loading file: {file_name}")
    np_embeddings_f = np.load(file_name)
else:
    raise ValueError("Invalid input: choose a number from 1 to 8.")


torch_input = torch.tensor(np_embeddings_f, dtype=torch.float32)


# === Q8.8 Conversion Functions ===
def float_to_q8_8(x):
    return np.round(x * 256).astype(np.int16)


def q8_8_to_float(x_q):
    return x_q.astype(np.float32) / 256


np_embeddings_q8_8 = float_to_q8_8(np_embeddings_f).astype(np.int32)


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

            n_total_shift = int(np.log2(n_total))
            delta_term = (delta_sq * n1 * n2) >> n_total_shift

            M12 = M1 + M2 + delta_term
            mu12 = (μ1 * n1 + μ2 * n2) >> n_total

            next_mu.append(mu12)
            next_M.append(M12)
            next_n.append(n_total)
        mu_list, M_list, n_list = next_mu, next_M, next_n

    var_q16 = M_list[0] // D
    return q8_8_to_float(var_q16 >> 8)


# === Load PWL Approximation Parameters ===
sqrt_npz = np.load("pwl_sqrt.npz")
sqrt_breaks = sqrt_npz["breaks"]
sqrt_slopes = sqrt_npz["slopes"]
sqrt_intercepts = sqrt_npz["intercepts"]

recip_npz = np.load("pwl_recip.npz")
recip_breaks = recip_npz["breaks"]
recip_slopes = recip_npz["slopes"]
recip_intercepts = recip_npz["intercepts"]


# === Generic PWL Approximation Function ===
def pwl_approx(x, breakpoints, slopes, intercepts):
    x = np.clip(x, breakpoints[0], breakpoints[-1])
    out = np.zeros_like(x)
    for i in range(len(slopes)):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        out[mask] = slopes[i] * x[mask] + intercepts[i]
    out[x >= breakpoints[-1]] = slopes[-1] * x[x >= breakpoints[-1]] + intercepts[-1]
    return out


# === Final Approximate LayerNorm using Q8.8 and PWL ===
def approx_layernorm_q8_8(x_q, eps=1e-5):
    mu_q = np.mean(x_q, axis=-1, keepdims=True)
    x_centered = x_q - mu_q
    var_approx = pairwise_variance_q8_8(x_q)
    sqrt_val = pwl_approx(var_approx + eps, sqrt_breaks, sqrt_slopes, sqrt_intercepts)
    recip_val = pwl_approx(sqrt_val, recip_breaks, recip_slopes, recip_intercepts)
    x_float = q8_8_to_float(x_centered)
    return x_float * recip_val


# === Reference Output using PyTorch LayerNorm ===
layernorm_torch = torch.nn.LayerNorm(D, elementwise_affine=False)
Y_true = layernorm_torch(torch_input).numpy()

# === Step-by-step Approximate LN Outputs ===
mu_q = np.mean(np_embeddings_q8_8, axis=-1, keepdims=True)
centered = np_embeddings_q8_8 - mu_q

# [1] Pairwise + exact sqrt + exact reciprocal
var_pairwise = pairwise_variance_q8_8(np_embeddings_q8_8)
sqrt_exact = np.sqrt(var_pairwise + 1e-5)
recip_exact = 1.0 / sqrt_exact
Y_exact = q8_8_to_float(centered) * recip_exact

# [2] Pairwise + PWL sqrt + exact reciprocal
sqrt_pwl = pwl_approx(var_pairwise + 1e-5, sqrt_breaks, sqrt_slopes, sqrt_intercepts)
recip_exact2 = 1.0 / sqrt_pwl
Y_pwl_sqrt = q8_8_to_float(centered) * recip_exact2

# [3] Pairwise + PWL sqrt + PWL reciprocal
recip_pwl = pwl_approx(sqrt_pwl, recip_breaks, recip_slopes, recip_intercepts)
Y_pwl_full = q8_8_to_float(centered) * recip_pwl

# [4] Final output from approx_layernorm_q8_8()
Y_approx = approx_layernorm_q8_8(np_embeddings_q8_8)


# === Evaluation Functions ===
def acc(y_hat):
    return (
        100 - (np.mean(np.abs(Y_true - y_hat)) / (np.mean(np.abs(Y_true)) + 1e-8)) * 100
    )


def mae(y_hat):
    return np.mean(np.abs(Y_true - y_hat))


# === Print Step-wise Accuracy Comparison ===
print("\n===== Step-wise Accuracy Comparison =====")
print(f"[1] Pairwise + Exact Sqrt/Recip   → Accuracy: {acc(Y_exact):.4f}%")
print(f"[2] Pairwise + PWL Sqrt + Exact   → Accuracy: {acc(Y_pwl_sqrt):.4f}%")
print(f"[3] Pairwise + PWL Sqrt + Recip   → Accuracy: {acc(Y_pwl_full):.4f}%")
