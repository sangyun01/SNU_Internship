import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Q8.8 → float 변환 함수
# ─────────────────────────────────────────────
def q8_8_to_float_unsigned(q):
    return q / 256.0


def q8_8_to_float_signed(q):
    if q >= 0x8000:
        q -= 0x10000
    return q / 256.0


def convert_q88_list(q_list, signed=False):
    return [
        q8_8_to_float_signed(q) if signed else q8_8_to_float_unsigned(q) for q in q_list
    ]


# ─────────────────────────────────────────────
# 안전한 PWL 근사 평가 함수
# ─────────────────────────────────────────────
def evaluate_pwl_safe(x, breakpoints, slopes, intercepts):
    y = np.zeros_like(x)
    for i in range(len(slopes)):
        x0, x1 = breakpoints[i], breakpoints[i + 1]
        mask = (x >= x0) & (x < x1)
        y[mask] = slopes[i] * x[mask] + intercepts[i]
    y[x >= breakpoints[-1]] = slopes[-1] * x[x >= breakpoints[-1]] + intercepts[-1]
    return y


# ─────────────────────────────────────────────
# Q8.8 LUT 값 (from .h)
# ─────────────────────────────────────────────
rsqrt_breakpoints_q88 = [
    0x0003,
    0x0025,
    0x0093,
    0x0177,
    0x039D,
    0x08D5,
    0x1599,
    0x3466,
    0x7F00,
]
rsqrt_slopes_q88 = [0xC694, 0xFDCF, 0xFF7C, 0xFFDE, 0xFFF7, 0xFFFE, 0xFFFF, 0x0000]
rsqrt_intercepts_q88 = [0x0A93, 0x0280, 0x018B, 0x00FB, 0x00A0, 0x0066, 0x0042, 0x002A]

sqrt_breakpoints_q88 = [
    0x0003,
    0x00F7,
    0x04A9,
    0x0C03,
    0x17BF,
    0x28A1,
    0x3F38,
    0x5BDB,
    0x7F00,
]
sqrt_slopes_q88 = [0x00E2, 0x004F, 0x002D, 0x001F, 0x0017, 0x0012, 0x000F, 0x000C]
sqrt_intercepts_q88 = [0x0038, 0x00C6, 0x0166, 0x0214, 0x02CE, 0x0393, 0x0461, 0x0536]

# 변환
rsqrt_breakpoints = convert_q88_list(rsqrt_breakpoints_q88, signed=False)
rsqrt_slopes = convert_q88_list(rsqrt_slopes_q88, signed=True)
rsqrt_intercepts = convert_q88_list(rsqrt_intercepts_q88, signed=True)

sqrt_breakpoints = convert_q88_list(sqrt_breakpoints_q88, signed=False)
sqrt_slopes = convert_q88_list(sqrt_slopes_q88, signed=True)
sqrt_intercepts = convert_q88_list(sqrt_intercepts_q88, signed=True)

# 데이터 생성 및 근사
x_vals = np.linspace(0.01, 127, 1000)
true_sqrt = np.sqrt(x_vals)
true_rsqrt = 1 / true_sqrt

sqrt_approx = evaluate_pwl_safe(x_vals, sqrt_breakpoints, sqrt_slopes, sqrt_intercepts)
rsqrt_approx = evaluate_pwl_safe(
    x_vals, rsqrt_breakpoints, rsqrt_slopes, rsqrt_intercepts
)

# ─────────────────────────────────────────────
# 그래프 출력
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 5))

# sqrt(x)
plt.subplot(1, 2, 1)
plt.plot(x_vals, true_sqrt, label="True sqrt(x)", color="blue")
plt.plot(x_vals, sqrt_approx, label="PWL sqrt(x)", color="red", linestyle="--")
for bp in sqrt_breakpoints:
    plt.axvline(x=bp, color="gray", linestyle=":", linewidth=0.8)
plt.title("PWL Approximation of sqrt(x)")
plt.xlabel("x")
plt.ylabel("sqrt(x)")
plt.grid(True)
plt.legend()

# 1/sqrt(x)
plt.subplot(1, 2, 2)
plt.plot(x_vals, true_rsqrt, label="True 1/sqrt(x)", color="blue")
plt.plot(x_vals, rsqrt_approx, label="PWL 1/sqrt(x)", color="red", linestyle="--")
for bp in rsqrt_breakpoints:
    plt.axvline(x=bp, color="gray", linestyle=":", linewidth=0.8)
plt.title("PWL Approximation of 1/sqrt(x)")
plt.xlabel("x")
plt.ylabel("1/sqrt(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
