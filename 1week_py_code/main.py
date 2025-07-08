import numpy as np
import torch

# Input Configuration & Q8.8 Conversion Functions
D = 768  # Dimension GPT - 2


def float_to_q8_8(x):
    return np.round(x * 256).astype(np.int16)


def q8_8_to_float(x_q):
    return x_q.astype(np.float32) / 256


# Mapping input 1 ~ 8 and selection to using sample data
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

choice = input("Select input (1~8): ").strip()

if choice in file_map:
    file_name = file_map[choice]
    np_embeddings_f = np.load(file_name)

np_embeddings_q8_8 = float_to_q8_8(np_embeddings_f).astype(
    np.int32
)  # convert the input data 16bit(8 int, 8 fractional)


def pairwise_variance_q8_8(
    x_q, G=16
):  # Efficient pairwise variance computation using 16 groups (for fixed-point Q8.8 input)
    D = x_q.shape[-1]  # D = 768
    splits = np.split(
        x_q, G, axis=-1
    )  # divide 768-dim vector into 16 groups (each of 48 dims)

    n_list = [s.shape[-1] for s in splits]
    mu_list = [np.sum(s, axis=-1, keepdims=True) // s.shape[-1] for s in splits]
    M_list = [
        np.sum(((s - mu).astype(np.int64)) ** 2, axis=-1, keepdims=True)
        for s, mu in zip(splits, mu_list)
    ]

    while len(mu_list) > 1:  # 16 -> 8 -> 4 -> 2 -> 1
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


# Load PWL Approximation Parameters
sqrt_npz = np.load("pwl_sqrt.npz")
sqrt_breaks = sqrt_npz["breaks"]  # separate 8 points
sqrt_slopes = sqrt_npz["slopes"]
sqrt_intercepts = sqrt_npz["intercepts"]

recip_npz = np.load("pwl_recip.npz")
recip_breaks = recip_npz["breaks"]  # separate 8 points
recip_slopes = recip_npz["slopes"]
recip_intercepts = recip_npz["intercepts"]


# PWL Approximation Function -> calculate sqrt(variance)
def pwl_approx(variance, breakpoints, slopes, intercepts):
    variance = np.clip(variance, breakpoints[0], breakpoints[-1])
    out = np.zeros_like(variance)
    for i in range(len(slopes)):
        mask = (variance >= breakpoints[i]) & (variance < breakpoints[i + 1])
        out[mask] = slopes[i] * variance[mask] + intercepts[i]
    out[variance >= breakpoints[-1]] = (
        slopes[-1] * variance[variance >= breakpoints[-1]] + intercepts[-1]
    )
    return out


# Final Approximate LayerNorm using Q8.8 and PWL
def approx_layernorm(x_q, eps=1e-5):
    mu_q = np.mean(x_q, axis=-1, keepdims=True)  # 1st calculate mean
    x_centered = x_q - mu_q  # 2nd x - μ
    var_approx = pairwise_variance_q8_8(x_q)  # 3rd s1 -> variance of input data
    sqrt_val = pwl_approx(
        var_approx + eps, sqrt_breaks, sqrt_slopes, sqrt_intercepts
    )  # Approximate sqrt using LUT-based PWL approximation
    recip_val = pwl_approx(
        sqrt_val, recip_breaks, recip_slopes, recip_intercepts
    )  # Approximate reciprocal using LUT-based PWL approximation
    x_float = q8_8_to_float(x_centered)
    return x_float * recip_val  # final LN approximation


# Set the critriea of PyTorch output result
torch_input = torch.tensor(np_embeddings_f, dtype=torch.float32)
layernorm_torch = torch.nn.LayerNorm(D, elementwise_affine=False)
Y_true = layernorm_torch(torch_input).numpy()

# Final output from approx_layernorm_q8_8()
Y_approx = approx_layernorm(np_embeddings_q8_8)


# Evaluation Functions
def accuracy(y_hat):
    return (
        100 - (np.mean(np.abs(Y_true - y_hat)) / (np.mean(np.abs(Y_true)) + 1e-8)) * 100
    )


# Print Step-wise Accuracy Comparison
print("\n===== Step-wise Accuracy Comparison =====")
print(f"Pairwise + PWL Sqrt + Recip   → Accuracy: {accuracy(Y_approx):.4f}%")
