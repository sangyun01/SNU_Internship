import numpy as np
import time
import torch

# Generate float32 input and convert to Q8.8 fixed-point
N, embedding_dim = 96, 768
np_embeddings_f = np.random.randn(N, embedding_dim).astype(np.float32)
torch_input = torch.tensor(np_embeddings_f, dtype=torch.float32)


def float_to_q8_8(x):
    return np.round(x * 256).astype(np.int16)


def q8_8_to_float(x_q):
    return x_q.astype(np.float32) / 256


# Convert float input to Q8.8
np_embeddings_q8_8 = float_to_q8_8(np_embeddings_f).astype(np.int32)


# Variance methods (Q8.8 integer input)
def true_variance_q8_8(x_q):
    N = x_q.shape[-1]
    mean = np.sum(x_q, axis=-1, keepdims=True) // N
    var = np.sum((x_q - mean) ** 2, axis=-1, keepdims=True) // N
    return q8_8_to_float(var // 256)


def one_pass_variance_q8_8(x_q):
    N = x_q.shape[-1]
    sum_x = np.sum(x_q, axis=-1, keepdims=True)
    sum_x2 = np.sum(x_q * x_q, axis=-1, keepdims=True)
    mean = sum_x // N
    mean_sq = sum_x2 // N
    var = mean_sq - (mean * mean)
    return q8_8_to_float(var // 256)  # mean^2 is Q16.16 → Q8.8


import numpy as np


def float_to_q8_8(x):
    return np.round(x * 256).astype(np.int16)


def q8_8_to_float(x_q):
    return x_q.astype(np.float32) / 256


def pairwise_variance_q8_8(x_q):
    G = 16
    D = x_q.shape[-1]
    assert D % G == 0

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
            delta_sq = (delta.astype(np.int64)) ** 2  # Q16.16
            delta_term = (delta_sq * n1 * n2) // n_total  # Q16.16

            M12 = M1 + M2 + delta_term  # Q16.16
            mu12 = (μ1 * n1 + μ2 * n2) // n_total

            next_mu.append(mu12)
            next_M.append(M12)
            next_n.append(n_total)

        mu_list, M_list, n_list = next_mu, next_M, next_n

    var_q16 = M_list[0] // D
    return q8_8_to_float(var_q16 // 256)  # Q16.16 → Q8.8 → float


# Timer
def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000


# Accuracy 계산
def percent_accuracy(est, ref):
    return 100 - np.abs(est - ref) / (ref + 1e-8) * 100


# 실행
x_q = np_embeddings_q8_8
true_var, t_true = measure_time(true_variance_q8_8, x_q)
onepass_var, t_onepass = measure_time(one_pass_variance_q8_8, x_q)
pairwise_var, t_pairwise = measure_time(pairwise_variance_q8_8, x_q)

# 기준값: float32 PyTorch
with torch.no_grad():
    var_torch = torch.var(torch_input, dim=-1, unbiased=False, keepdim=True).numpy()

# 정확도
acc_true = percent_accuracy(true_var, var_torch)
acc_one = percent_accuracy(onepass_var, var_torch)
acc_pair = percent_accuracy(pairwise_var, var_torch)

# 출력
print("===== Accuracy (% Error vs PyTorch) =====")
print(f"[True Var Q8.8]     {acc_true.mean():.4f}%")
print(f"[One-Pass Q8.8]     {acc_one.mean():.4f}%")
print(f"[Pairwise Q8.8]     {acc_pair.mean():.4f}%")

print("\n===== Timing (ms) =====")
print(f"[True Var Q8.8]     {t_true:.4f} ms")
print(f"[One-Pass Q8.8]     {t_onepass:.4f} ms")
print(f"[Pairwise Q8.8]     {t_pairwise:.4f} ms")
