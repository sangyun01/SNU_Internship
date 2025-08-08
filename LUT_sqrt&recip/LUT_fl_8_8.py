import numpy as np
import pwlf


def float_to_q8_8(val):
    scaled = int(round(val * 256))
    if not -32768 <= scaled <= 32767:
        raise OverflowError(f"{val} is out of Q8.8 range")
    if scaled < 0:
        scaled = (1 << 16) + scaled  # two's complement
    return scaled


def generate_header_file(name_prefix, breakpoints, slopes, intercepts):
    def format_array(arr, signed=False):
        lines = []
        for val in arr:
            q = float_to_q8_8(val)
            if signed and q >= (1 << 15):  # convert to signed 16-bit two's complement
                q -= 1 << 16
            hex_val = f"0x{q & 0xFFFF:04X}"
            lines.append(hex_val)
        return lines

    bp_lines = format_array(breakpoints, signed=False)
    slope_lines = format_array(slopes, signed=True)
    intercept_lines = format_array(intercepts, signed=True)

    h_content = f"""#ifndef {name_prefix.upper()}_TABLE_H
#define {name_prefix.upper()}_TABLE_H

// Q8.8 format lookup tables for {name_prefix}(x) approximation (8 segments)
const uint16_t {name_prefix}_breakpoints[9] = {{
    {', '.join(bp_lines)}
}};

const int16_t {name_prefix}_slopes[8] = {{
    {', '.join(slope_lines)}
}};

const int16_t {name_prefix}_intercepts[8] = {{
    {', '.join(intercept_lines)}
}};

#endif // {name_prefix.upper()}_TABLE_H
"""
    with open(f"{name_prefix}_table.h", "w") as f:
        f.write(h_content)


# Prepare domain and true values
x_vals = np.linspace(0.01, 127, 1000)
sqrt_vals = np.sqrt(x_vals)
recip_vals = 1 / sqrt_vals

# Fit sqrt(x)
sqrt_model = pwlf.PiecewiseLinFit(x_vals, sqrt_vals)
sqrt_model.fit(8)

# Fit 1/sqrt(x)
recip_model = pwlf.PiecewiseLinFit(x_vals, recip_vals)
recip_model.fit(8)

# Generate header files
generate_header_file(
    "sqrt", sqrt_model.fit_breaks, sqrt_model.slopes, sqrt_model.intercepts
)
generate_header_file(
    "rsqrt", recip_model.fit_breaks, recip_model.slopes, recip_model.intercepts
)
