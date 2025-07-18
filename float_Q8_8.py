def float_to_q8_8(value: float) -> int:
    scaled = int(round(value * 256))  # Multiply by 2^8 to shift into Q8.8 scale

    # Check if the scaled value fits into 16-bit signed range
    if not -32768 <= scaled <= 32767:
        raise OverflowError(f"Value {value} is out of Q8.8 representable range.")

    # If negative, convert to 2's complement unsigned 16-bit
    if scaled < 0:
        scaled = (1 << 16) + scaled

    return scaled  # Return as unsigned 16-bit integer (0 ~ 65535)


# Example usage
test_values = [-2.5, -1.0, 0.0, 0.5, 1.25, 3.75]
for val in test_values:
    q88 = float_to_q8_8(val)
    print(f"{val:>6} → Q8.8: 0x{q88:04X} ({q88})")
