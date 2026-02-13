#!/usr/bin/env python3
"""Convert a .cubin binary to a C byte-array header file.

Usage: py cubin_to_header.py <input.cubin> <output.h>

Generates a header with:
  static const unsigned char qwen_asr_kernels_cubin[] = { 0x7f, 0x45, ... };
  static const unsigned int qwen_asr_kernels_cubin_len = 12345;
"""

import sys
import os


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.cubin> <output.h>", file=sys.stderr)
        sys.exit(1)

    cubin_path = sys.argv[1]
    header_path = sys.argv[2]

    with open(cubin_path, "rb") as f:
        data = f.read()

    if len(data) == 0:
        print(f"Error: {cubin_path} is empty", file=sys.stderr)
        sys.exit(1)

    with open(header_path, "w") as f:
        f.write("/* Auto-generated from qwen_asr_kernels.cu -- do not edit */\n")
        f.write("#ifndef QWEN_ASR_KERNELS_CUBIN_H\n")
        f.write("#define QWEN_ASR_KERNELS_CUBIN_H\n\n")
        f.write("static const unsigned char qwen_asr_kernels_cubin[] = {\n")

        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + 16 < len(data):
                f.write(f"    {hex_bytes},\n")
            else:
                f.write(f"    {hex_bytes}\n")

        f.write("};\n\n")
        f.write(f"static const unsigned int qwen_asr_kernels_cubin_len = {len(data)};\n\n")
        f.write("#endif /* QWEN_ASR_KERNELS_CUBIN_H */\n")

    print(f"Generated {header_path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
