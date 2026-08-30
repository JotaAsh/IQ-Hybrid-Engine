"""Independent GGUF iMatrix Merger module.

Combines 2 to 3 GGUF activation importance matrices (imatrix) into a single
unified GGUF imatrix file for consumption by llama-quantize and iq-hybrid.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import gguf
import numpy as np

logger = logging.getLogger(__name__)

MAX_IMATRIX_FILES = 3
MIN_IMATRIX_FILES = 2


def _read_raw_imatrix_tensors(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Extract raw in_sum2 and counts tensors plus metadata from an imatrix GGUF file."""
    reader = gguf.GGUFReader(str(path))
    in_sum2_dict: dict[str, np.ndarray] = {}
    counts_dict: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}

    for key, field in reader.fields.items():
        try:
            if key == "imatrix.datasets":
                datasets_list: list[str] = []
                for item in field.data:
                    if isinstance(item, (bytes, bytearray, memoryview)):
                        decoded = bytes(item).decode("utf-8", errors="ignore").strip("\x00")
                        datasets_list.append(decoded if decoded else "dataset")
                    elif isinstance(item, str):
                        datasets_list.append(
                            item.strip("\x00") if item.strip("\x00") else "dataset"
                        )
                    else:
                        datasets_list.append(str(item))
                meta[key] = datasets_list if datasets_list else ["dataset"]
            else:
                data = field.data
                if isinstance(data, np.ndarray):
                    data = data.tolist()
                elif isinstance(data, np.generic):
                    data = data.item()
                meta[key] = data
        except Exception:
            meta[key] = str(field)

    for tensor in reader.tensors:
        name = tensor.name
        arr = np.array(tensor.data, dtype=np.float32)

        if name.endswith(".in_sum2"):
            base = name[:-8]
            in_sum2_dict[base] = arr
        elif name.endswith(".counts"):
            base = name[:-7]
            counts_dict[base] = arr

    return in_sum2_dict, counts_dict, meta


def merge_imatrix_files(
    paths: list[str | Path],
    output_path: str | Path | None = None,
    method: str = "add",
    output_dir: str | Path = "iMatrix",
) -> Path:
    """Merge 2 or 3 imatrix GGUF files into a single GGUF imatrix.

    Args:
        paths: List of 2 to 3 paths to GGUF imatrix files.
        output_path: Optional explicit output file path.
        method: Merge method: 'add' (sum activations, default), 'max' (conservative max pooling),
                or 'mean' (weighted mean).
        output_dir: Directory where the merged file will be saved if output_path is omitted.

    Returns:
        Path to the generated merged GGUF imatrix file.

    Raises:
        ValueError: If fewer than 2 or more than 3 paths are supplied, or if method is unknown.
        FileNotFoundError: If any of the input files does not exist.
    """
    n_files = len(paths)
    if n_files < MIN_IMATRIX_FILES or n_files > MAX_IMATRIX_FILES:
        raise ValueError(
            f"iMatrix merger requires between {MIN_IMATRIX_FILES} and {MAX_IMATRIX_FILES} files (received {n_files})."
        )

    resolved_paths = [Path(p).resolve() for p in paths]
    for p in resolved_paths:
        if not p.is_file():
            raise FileNotFoundError(f"iMatrix file not found: {p}")

    method = method.lower()
    if method not in ("add", "max", "mean"):
        raise ValueError(f"Unknown merge method '{method}'. Allowed methods: 'add', 'max', 'mean'.")

    # Load all input files
    loaded_data: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]] = []
    all_tensor_names: set[str] = set()

    for p in resolved_paths:
        sum2_dict, counts_dict, meta = _read_raw_imatrix_tensors(p)
        loaded_data.append((sum2_dict, counts_dict, meta))
        all_tensor_names.update(sum2_dict.keys())

    if not all_tensor_names:
        raise ValueError("No valid imatrix tensors (.in_sum2) found in input files.")

    # Determine destination path
    if output_path is not None:
        dest_path = Path(output_path)
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_names = "_".join(
            p.stem.replace("-imatrix", "").replace(".imatrix", "") for p in resolved_paths
        )
        dest_path = out_dir / f"merged_{stem_names}_{method}.gguf"

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Merging %d imatrix files using method '%s' -> %s", n_files, method, dest_path)

    # Collect metadata fields for llama.cpp compatibility
    all_datasets: list[str] = []
    total_chunk_count = 0
    max_chunk_size = 512

    for p, (_, _, meta) in zip(resolved_paths, loaded_data, strict=True):
        ds = meta.get("imatrix.datasets")
        if isinstance(ds, list) and ds:
            all_datasets.extend(ds)
        else:
            all_datasets.append(p.stem)

        cc = meta.get("imatrix.chunk_count", 1)
        if isinstance(cc, list) and cc:
            cc = cc[0]
        try:
            total_chunk_count += int(cc)
        except ValueError, TypeError:
            total_chunk_count += 1

        cs = meta.get("imatrix.chunk_size", 512)
        if isinstance(cs, list) and cs:
            cs = cs[0]
        try:
            max_chunk_size = max(max_chunk_size, int(cs))
        except ValueError, TypeError:
            pass

    if not all_datasets:
        all_datasets = [p.name for p in resolved_paths]

    # Initialize GGUFWriter
    writer = gguf.GGUFWriter(str(dest_path), "imatrix")
    writer.add_string("general.type", "imatrix")
    writer.add_array("imatrix.datasets", all_datasets)
    writer.add_uint32("imatrix.chunk_count", max(1, total_chunk_count))
    writer.add_uint32("imatrix.chunk_size", max(1, max_chunk_size))
    writer.add_int32("imatrix.dataset_count", len(all_datasets))
    writer.add_string("imatrix.merge_method", method)

    source_names = ", ".join(p.name for p in resolved_paths)
    writer.add_string("imatrix.sources", source_names)

    # Process and write each tensor
    merged_tensor_count = 0
    for name in sorted(all_tensor_names):
        sum2_list: list[np.ndarray] = []
        counts_list: list[np.ndarray] = []

        for sum2_dict, counts_dict, _ in loaded_data:
            if name in sum2_dict:
                sum2_list.append(sum2_dict[name])
                # Ensure counts is a float32 array
                c = counts_dict.get(name, np.array([1.0], dtype=np.float32))
                counts_list.append(c)

        if not sum2_list:
            continue

        ref_shape = sum2_list[0].shape
        # Filter for shape consistency
        valid_pairs = [
            (s, c) for s, c in zip(sum2_list, counts_list, strict=True) if s.shape == ref_shape
        ]

        if method == "add":
            merged_sum2 = np.sum([s for s, _ in valid_pairs], axis=0).astype(np.float32)
            merged_counts = np.sum([c for _, c in valid_pairs], axis=0).astype(np.float32)
        elif method == "max":
            # Normalized max pooling: max(in_sum2 / counts) * mean_counts
            means = [s / np.maximum(1e-7, c) for s, c in valid_pairs]
            merged_mean = np.maximum.reduce(means).astype(np.float32)
            avg_counts = np.mean([c for _, c in valid_pairs], axis=0).astype(np.float32)
            merged_sum2 = (merged_mean * avg_counts).astype(np.float32)
            merged_counts = avg_counts
        elif method == "mean":
            # Token-weighted mean
            sum_sum2 = np.sum([s for s, _ in valid_pairs], axis=0)
            sum_counts = np.sum([c for _, c in valid_pairs], axis=0)
            avg_counts = (sum_counts / len(valid_pairs)).astype(np.float32)
            merged_mean = (sum_sum2 / np.maximum(1e-7, sum_counts)).astype(np.float32)
            merged_sum2 = (merged_mean * avg_counts).astype(np.float32)
            merged_counts = avg_counts

        writer.add_tensor(
            f"{name}.in_sum2",
            merged_sum2,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.add_tensor(
            f"{name}.counts",
            merged_counts,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        merged_tensor_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info("Successfully merged %d tensor pairs into %s", merged_tensor_count, dest_path)
    return dest_path


def main() -> None:
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Merge 2 to 3 GGUF iMatrix files into a single unified GGUF imatrix.",
    )
    parser.add_argument(
        "-i",
        "--imatrix",
        nargs="+",
        required=True,
        help="2 or 3 GGUF imatrix file paths to merge",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output GGUF file path (default: auto-named in iMatrix/ folder)",
    )
    parser.add_argument(
        "--output-dir",
        default="iMatrix",
        help="Directory to save the merged imatrix (default: iMatrix)",
    )
    parser.add_argument(
        "--method",
        choices=["add", "max", "mean"],
        default="add",
        help="Merge strategy: 'add' (additive accumulation, default), 'max' (conservative max pooling), 'mean' (weighted mean)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        out_file = merge_imatrix_files(
            paths=args.imatrix,
            output_path=args.output,
            method=args.method,
            output_dir=args.output_dir,
        )
        print(f"\n[OK] Merged iMatrix saved to: {out_file}")
    except Exception as e:
        logger.error("Failed to merge imatrix files: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
