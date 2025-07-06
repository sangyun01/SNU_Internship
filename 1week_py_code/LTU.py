import numpy as np
import pwlf

# === 사전 계산 데이터 생성 ===
x_vals = np.linspace(0.01, 64, 1000)
sqrt_vals = np.sqrt(x_vals)
recip_vals = 1 / sqrt_vals

# sqrt 근사 모델
sqrt_model = pwlf.PiecewiseLinFit(x_vals, sqrt_vals)
sqrt_breaks = sqrt_model.fit(8)
np.savez(
    "pwl_sqrt.npz",
    breaks=sqrt_model.fit_breaks,
    slopes=sqrt_model.slopes,
    intercepts=sqrt_model.intercepts,
)

# reciprocal 근사 모델
recip_model = pwlf.PiecewiseLinFit(x_vals, recip_vals)
recip_breaks = recip_model.fit(8)
np.savez(
    "pwl_recip.npz",
    breaks=recip_model.fit_breaks,
    slopes=recip_model.slopes,
    intercepts=recip_model.intercepts,
)
