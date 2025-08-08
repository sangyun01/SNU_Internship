import numpy as np
import pwlf


# Q8.8 변환 함수
def float_to_q8_8(val):
    scaled = int(round(val * 256))
    if not -32768 <= scaled <= 32767:
        raise OverflowError(f"{val} is out of Q8.8 range")
    if scaled < 0:
        scaled = (1 << 16) + scaled
    return scaled


# .h 파일 생성 함수
def generate_header_file(name_prefix, breakpoints, slopes, intercepts):
    def format_array(arr, signed=False):
        lines = []
        for val in arr:
            q = float_to_q8_8(val)
            if signed and q >= (1 << 15):  # 2's complement
                q -= 1 << 16
            lines.append(f"0x{q & 0xFFFF:04X}")
        return lines

    bp_lines = format_array(breakpoints, signed=False)
    slope_lines = format_array(slopes, signed=True)
    intercept_lines = format_array(intercepts, signed=True)

    h_content = f"""#ifndef {name_prefix.upper()}_TABLE_H
#define {name_prefix.upper()}_TABLE_H

// Q8.8 format lookup tables for {name_prefix}(sqrt(x)) approximation (8 segments)
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


# √x 범위 도메인 (입력이 sqrt(x))
s_vals = np.linspace(np.sqrt(0.01), np.sqrt(127), 1000)
recip_s_vals = 1 / s_vals  # 타겟 함수: 1/sqrt(x) = 1/s

# PWL 근사 (8 구간)
model = pwlf.PiecewiseLinFit(s_vals, recip_s_vals)
model.fit(12)

# .h LUT 파일 생성
generate_header_file("recip_sqrtx", model.fit_breaks, model.slopes, model.intercepts)
