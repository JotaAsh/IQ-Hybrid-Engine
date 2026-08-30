"""Quantizer module — wraps llama-quantize for dry-run and full quantization.

Resolves the quantizer binary path from (in priority order):
  1. Explicit ``quantizer_path`` argument passed at call site
  2. ``LLAMA_QUANTIZE_PATH`` in ``.env`` / environment
  3. ``llama-quantize`` found on the system ``PATH``
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

from dotenv import dotenv_values

from iq_hybrid.utils.utils import parse_fallback_warnings, parse_quant_size

logger = logging.getLogger(__name__)

# ── Binary resolution ────────────────────────────────────────────────────────

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
if not _ENV_FILE.exists():
    _ENV_FILE = Path.cwd() / ".env"
_ENV_KEY = "LLAMA_QUANTIZE_PATH"


def resolve_quantizer_path(override: str | None = None) -> str:
    """Return a validated path to the llama-quantize binary.

    Resolution order:
      1. *override* argument (e.g. from ``--quantizer-path`` CLI flag)
      2. ``LLAMA_QUANTIZE_PATH`` from ``.env`` or environment variable
      3. ``llama-quantize`` / ``llama-quantize.exe`` discovered via ``PATH``

    Raises:
        FileNotFoundError: If no usable binary can be located.
    """
    candidates: list[str] = []

    if override:
        candidates.append(override)

    # .env file takes precedence over OS environment for explicit config
    env_values = dotenv_values(_ENV_FILE)
    env_path = env_values.get(_ENV_KEY) or os.environ.get(_ENV_KEY)
    if env_path:
        candidates.append(env_path)

    # Fallback: search system PATH
    which_result = shutil.which("llama-quantize")
    if which_result:
        candidates.append(which_result)

    for path in candidates:
        if Path(path).is_file():
            logger.debug("Resolved quantizer binary: %s", path)
            return path

    raise FileNotFoundError(
        "Could not locate llama-quantize binary. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\n\nSet LLAMA_QUANTIZE_PATH in .env or pass --quantizer-path."
        if candidates
        else "Could not locate llama-quantize binary.\n"
        "Set LLAMA_QUANTIZE_PATH in .env, add it to PATH, or pass --quantizer-path."
    )


def resolve_output_dir() -> str | None:
    """Return the output directory from ``.env`` or environment, if set."""
    env_values = dotenv_values(_ENV_FILE)
    return env_values.get("OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or None


# ── Command builder ──────────────────────────────────────────────────────────


def _build_cmd(
    flags: dict,
    model_in: str,
    model_out: str,
    *,
    dry_run: bool = False,
    quantizer_path: str | None = None,
) -> list[str]:
    """Build the llama-quantize command line. Options precede positional args."""
    binary = quantizer_path or resolve_quantizer_path()
    cmd = [binary, "--allow-requantize"]

    if dry_run:
        cmd.append("--dry-run")

    if flags.get("imatrix"):
        imatrix = flags["imatrix"]
        if isinstance(imatrix, list):
            if len(imatrix) > 1:
                logger.info(
                    "Passing primary imatrix '%s' to llama-quantize (data was combined across %d files in the classifier)",
                    imatrix[0],
                    len(imatrix),
                )
            imatrix = imatrix[0]
        cmd.extend(["--imatrix", imatrix])

    cmd.extend(["--output-tensor-type", flags["output_tensor_type"]])
    cmd.extend(["--token-embedding-type", flags["token_embedding_type"]])

    for rule in flags["tensor_type_rules"]:
        parts = rule.split(" ", 1)
        if len(parts) == 2:
            cmd.extend(["--tensor-type", parts[1].strip('"')])

    # Positional args: model_in  model_out  type
    cmd.extend([model_in, model_out, flags["base_type"]])
    return cmd


# ── Public API ───────────────────────────────────────────────────────────────


def run_dry_run(
    flags: dict,
    model_in: str,
    *,
    verbose: bool = False,
    quantizer_path: str | None = None,
) -> tuple[float | None, list[str]]:
    """Execute a dry-run quantization and return (estimated_size_mib, fallbacks)."""
    devnull = "NUL" if platform.system() == "Windows" else "/dev/null"
    cmd = _build_cmd(flags, model_in, devnull, dry_run=True, quantizer_path=quantizer_path)

    if verbose:
        logger.debug("Dry-run command:\n%s", " ".join(f'"{c}"' if " " in c else c for c in cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as exc:
        logger.error("Quantizer binary not found: %s", exc)
        return None, []
    except subprocess.TimeoutExpired:
        logger.error("Dry-run timed out after 300 s")
        return None, []

    output = (result.stdout or "") + (result.stderr or "")
    size = parse_quant_size(output)
    fallbacks = parse_fallback_warnings(output)

    if verbose:
        if result.stdout:
            logger.debug("Dry-run stdout:\n%s", result.stdout)
        if result.stderr:
            logger.debug("Dry-run stderr:\n%s", result.stderr)
    elif size is None:
        logger.warning("Could not parse size from dry-run output. STDERR: %.2000s", result.stderr)

    return size, fallbacks


def run_quantization(
    flags: dict,
    model_in: str,
    model_out: str,
    *,
    verbose: bool = False,
    quantizer_path: str | None = None,
) -> bool:
    """Run the actual quantization and return True on success."""
    cmd = _build_cmd(flags, model_in, model_out, dry_run=False, quantizer_path=quantizer_path)

    if verbose:
        logger.debug(
            "Full quantize command:\n%s", " ".join(f'"{c}"' if " " in c else c for c in cmd)
        )
    else:
        logger.info("Running: %s ...", " ".join(cmd[:6]))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as exc:
        logger.error("Quantizer binary not found: %s", exc)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Quantization timed out after 3600 s")
        return False

    output = (result.stdout or "") + (result.stderr or "")
    fallbacks = parse_fallback_warnings(output)

    if verbose or result.returncode != 0:
        logger.info("--- STDOUT ---\n%s", result.stdout)
        logger.info("--- STDERR ---\n%s", result.stderr)

    if fallbacks:
        logger.warning(
            "%d tensor fallback(s) occurred during quantization:\n%s",
            len(fallbacks),
            "\n".join(f"    {fb}" for fb in fallbacks),
        )

    success = result.returncode == 0
    if success:
        size_mib = os.path.getsize(model_out) / 1024 / 1024
        logger.info("Done: %s (%.0f MiB)", model_out, size_mib)
    else:
        logger.error("Quantization failed with return code %d", result.returncode)

    return success
