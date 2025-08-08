import numpy as np
import pwlf

# Generate data
x_vals = np.linspace(0.01, 128, 1000)  # Input domain: avoid zero for stability
sqrt_vals = np.sqrt(x_vals)  # True sqrt values
recip_vals = 1 / sqrt_vals  # True reciprocal sqrt values

# linear model(sqrt)
sqrt_model = pwlf.PiecewiseLinFit(x_vals, sqrt_vals)
sqrt_breaks = sqrt_model.fit(8)  # Fit with 8 segments
np.savez(  # Save model parameters as .npz
    "pwl_sqrt.npz",
    breaks=sqrt_model.fit_breaks,
    slopes=sqrt_model.slopes,
    intercepts=sqrt_model.intercepts,
)

# linear model(recip)
recip_model = pwlf.PiecewiseLinFit(x_vals, recip_vals)
recip_breaks = recip_model.fit(8)  # Fit with 8 segments
np.savez(  # Save model parameters as .npz
    "pwl_recip.npz",
    breaks=recip_model.fit_breaks,
    slopes=recip_model.slopes,
    intercepts=recip_model.intercepts,
)
