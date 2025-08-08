import numpy as np
import matplotlib.pyplot as plt


# Q8.8 → float 변환 함수
def q8_8_to_float(q, signed=False):
    if signed and q >= 0x8000:
        q -= 0x10000
    return q / 256.0


# LUT 데이터 (Q8.8, from .h file)
recip_sqrtx_breakpoints_q88 = [
    0x001A,
    0x0024,
    0x0038,
    0x005A,
    0x0099,
    0x0114,
    0x0216,
    0x0482,
    0x0B45,
]
recip_sqrtx_slopes_q88 = [
    0xB541,
    0xDFA3,
    0xF339,
    0xFB5F,
    0xFE80,
    0xFF93,
    0xFFE6,
    0xFFFB,
]
recip_sqrtx_intercepts_q88 = [
    0x1164,
    0x0B78,
    0x0738,
    0x045A,
    0x027C,
    0x0154,
    0x00A6,
    0x0047,
]

# Q8.8 → float 변환
breakpoints = [q8_8_to_float(q) for q in recip_sqrtx_breakpoints_q88]
slopes = [q8_8_to_float(q, signed=True) for q in recip_sqrtx_slopes_q88]
intercepts = [q8_8_to_float(q, signed=True) for q in recip_sqrtx_intercepts_q88]


# 안전한 PWL 평가 함수
def evaluate_pwl(x, breaks, slopes, intercepts):
    y = np.zeros_like(x)
    for i in range(len(slopes)):
        x0, x1 = breaks[i], breaks[i + 1]
        mask = (x >= x0) & (x < x1)
        y[mask] = slopes[i] * x[mask] + intercepts[i]
    y[x >= breaks[-1]] = slopes[-1] * x[x >= breaks[-1]] + intercepts[-1]
    return y


# 도메인: sqrt(x) 자체를 x축으로 사용
sqrt_x_vals = np.linspace(breakpoints[0], breakpoints[-1], 1000)
true_rsqrt = 1 / sqrt_x_vals
pwl_rsqrt = evaluate_pwl(sqrt_x_vals, breakpoints, slopes, intercepts)

# 그래프 출력
plt.figure(figsize=(6, 4))
plt.plot(sqrt_x_vals, true_rsqrt, label="True 1/sqrt(x)", color="blue")
plt.plot(
    sqrt_x_vals, pwl_rsqrt, label="PWL via recip_sqrtx LUT", color="red", linestyle="--"
)
for bp in breakpoints:
    plt.axvline(x=bp, color="gray", linestyle=":", linewidth=0.6)
plt.xlabel("sqrt(x)")
plt.ylabel("1/sqrt(x)")
plt.title("LUT-based PWL Approximation of 1/sqrt(x) (input = sqrt(x))")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
