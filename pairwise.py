import numpy as np
import time
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


import numpy as np


def pairwise_variance(x):
    # Number of groups to split into (must be a power of two)
    G = 16
    D = x.shape[-1]
    # Ensure the last dimension is divisible by G
    assert D % G == 0, f"Last dim {D} must be divisible by {G}"

    # Split the input tensor into G equal parts along the last axis
    splits = np.split(x, G, axis=-1)

    # Compute per-group counts, means, and sum of squared deviations:
    # M_i = sum_j (x_ij - mu_i)^2 = n_i * var_i
    n_list = [s.shape[-1] for s in splits]  # number of elements in each group
    mu_list = [s.mean(axis=-1, keepdims=True) for s in splits]  # group means (mu_i)
    M_list = [
        np.var(s, axis=-1, keepdims=True, ddof=0)
        * n  # group sum of squared deviations (M_i)
        for s, n in zip(splits, n_list)
    ]

    # Iteratively merge groups in pairs: 16 -> 8 -> 4 -> 2 -> 1
    while len(mu_list) > 1:
        next_mu, next_M, next_n = [], [], []
        for i in range(0, len(mu_list), 2):
            # Grab two adjacent groups
            μ1, μ2 = mu_list[i], mu_list[i + 1]
            M1, M2 = M_list[i], M_list[i + 1]
            n1, n2 = n_list[i], n_list[i + 1]

            # Compute merged sum of squared deviations:
            # M_12 = M1 + M2 + (mu1 - mu2)^2 * (n1 * n2) / (n1 + n2)
            delta = μ1 - μ2
            M12 = M1 + M2 + delta**2 * (n1 * n2) / (n1 + n2)
            next_M.append(M12)

            # Compute merged mean and count:
            # mu_12 = (n1 * mu1 + n2 * mu2) / (n1 + n2)
            next_mu.append((μ1 * n1 + μ2 * n2) / (n1 + n2))
            next_n.append(n1 + n2)

        # Prepare for next iteration
        mu_list, M_list, n_list = next_mu, next_M, next_n

    # After merging, compute final biased variance:
    M_total = M_list[0]  # total sum of squared deviations
    n_total = n_list[0]  # total number of elements
    var_total = M_total / n_total  # biased variance = M_total / n_total
    return var_total


# Timer function
def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000


# Accuracy 계산 함수
def percent_accuracy(est, ref):
    return 100 - np.abs(est - ref) / (ref + 1e-8) * 100


# 실행
x = np_embeddings
true_var, t_true = measure_time(true_variance, x)
onepass_var, t_onepass = measure_time(one_pass_variance, x)
pairwise_var, t_pairwise = measure_time(pairwise_variance, x)

# PyTorch 기준값
with torch.no_grad():
    var_torch = torch.var(torch_input, dim=-1, unbiased=False, keepdim=True).numpy()

# 정확도 계산
acc_true = percent_accuracy(true_var, var_torch)
acc_one = percent_accuracy(onepass_var, var_torch)
acc_pair = percent_accuracy(pairwise_var, var_torch)


# 출력
print("===== Accuracy (% Error vs PyTorch) =====")
print(f"[True Var]     {acc_true.mean():.4f}%")
print(f"[One-Pass Var] {acc_one.mean():.4f}%")
print(f"[Pairwise Var] {acc_pair.mean():.4f}%")

print("\n===== Timing (ms) =====")
print(f"[True Var]     {t_true:.4f} ms")
print(f"[One-Pass Var] {t_onepass:.4f} ms")
print(f"[Pairwise Var] {t_pairwise:.4f} ms")
