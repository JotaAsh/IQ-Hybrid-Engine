"""Utility functions for parsing llama-quantize output."""

import re


def parse_quant_size(output: str) -> float | None:
    """Extract quantized model size (MiB) from llama-quantize output.

    Prefers the ``quant size`` line over ``model size`` when both are present.
    Returns None if neither can be parsed.
    """
    match = re.search(r"quant size\s*=\s*([0-9.]+)\s*MiB", output)
    if match:
        return float(match.group(1))

    match = re.search(r"model size\s*=\s*([0-9.]+)\s*MiB", output)
    if match:
        return float(match.group(1))

    return None


def parse_fallback_warnings(output: str) -> list[str]:
    """Extract genuine quantization fallback or warning lines from llama-quantize output.

    A fallback occurs when the quantizer cannot apply the requested tier
    (e.g. due to block-size alignment constraints) and reverts to an alternative.
    """
    return [
        line.strip()
        for line in output.splitlines()
        if re.search(r"(?:fallback|falling back|cannot quantize|warning:)", line, re.IGNORECASE)
    ]
