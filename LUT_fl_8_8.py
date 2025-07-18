import numpy as np
import pwlf


def float_to_q8_8(val):
    scaled = int(round(val * 256))
    if not -32768 <= scaled <= 32767:
        raise OverflowError(f"{val} is out of Q8.8 range")
    if scaled < 0:
        scaled = (1 << 16) + scaled
    return scaled


def print_q88_array(name, arr):
    print(f"{name} (Q8.8, hex):")
    for val in arr:
        q = float_to_q8_8(val)
        print(f"0x{q:04X}", end=" ")
    print("\n")


# Generate domain and function values
x_vals = np.linspace(0.01, 127, 1000)
sqrt_vals = np.sqrt(x_vals)
recip_vals = 1 / sqrt_vals

# √x PWL model
sqrt_model = pwlf.PiecewiseLinFit(x_vals, sqrt_vals)
sqrt_model.fit(8)

print("==== PWL Approximation for sqrt(x) ====")
print_q88_array("Breakpoints", sqrt_model.fit_breaks)
print_q88_array("Slopes", sqrt_model.slopes)
print_q88_array("Intercepts", sqrt_model.intercepts)

# 1/√x PWL model
recip_model = pwlf.PiecewiseLinFit(x_vals, recip_vals)
recip_model.fit(8)

print("==== PWL Approximation for 1/sqrt(x) ====")
print_q88_array("Breakpoints", recip_model.fit_breaks)
print_q88_array("Slopes", recip_model.slopes)
print_q88_array("Intercepts", recip_model.intercepts)
