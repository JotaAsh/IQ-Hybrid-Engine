# IQ-Hybrid Engine

An advanced selective hybrid quantization engine for GGUF models. It optimizes per-tensor quantization tier assignments by combining activation importance matrices (iMatrix), depth-aware heuristics, and dual optimization solvers (Greedy priority-queue and SciPy-powered Mixed-Integer Linear Programming / MCKP) to maximize model fidelity within exact file size (MiB) or bits-per-weight (BPW) budgets.

---

## Operational Runbook

### 1. Requirements and Installation

* **Python 3.14+** (Built and validated on native Python 3.14 standard).
* Compiled `llama-quantize` binary in system `PATH` or configured in `.env`.

```bash
# Clone repository and install dependencies
git clone https://github.com/JotaAsh/iq-hybrid-engine.git
cd iq-hybrid-engine
pip install -r requirements.txt
```

---

### 2. Execution Workflows

#### A. Fast Inspection Dry-Run (Estimate size without disk write)
```bash
python -m iq_hybrid.cli --model "Models/model-bf16.gguf" --imatrix "iMatrix/model.imatrix.gguf" --profile quality_ladder
```

#### B. Full Quantization Execution (`--run`)
```bash
python -m iq_hybrid.cli --model "Models/model-bf16.gguf" --imatrix "iMatrix/model.imatrix.gguf" --size 6310 --run
```

#### C. Exact Global Optimization Solver (MCKP / MILP)
```bash
python -m iq_hybrid.cli --model "Models/model-bf16.gguf" --imatrix "iMatrix/model.imatrix.gguf" --size 6310 --solver mckp --wide-ladder
```

#### D. Multi-Dataset iMatrix Fusion (Up to 3 datasets)
Pass multiple `--imatrix` arguments. The engine will merge activations into a unified GGUF file in `iMatrix/` before running quantization:
```bash
python -m iq_hybrid.cli --model "Models/model-bf16.gguf" \
  --imatrix "iMatrix/dataset_en.imatrix.gguf" \
  --imatrix "iMatrix/dataset_es.imatrix.gguf" \
  --imatrix-method add \
  --profile quality_ladder --run
```

The merger utility can also be invoked directly as a standalone tool:
```bash
python -m iq_hybrid.io.imatrix_merger -i iMatrix/data1.imatrix.gguf iMatrix/data2.imatrix.gguf --method add -o iMatrix/merged.gguf
```

---

## Quantization Profiles & Presets

Quantization behavior is governed by declarative profiles. You can select built-in presets or pass external JSON/YAML recipe files via `--profile` or `--size`:

### Built-in Presets

| Profile | Target BPW / Budget | Description |
| :--- | :--- | :--- |
| `quality_ladder` | Custom / MiB | Strict high-quality profile enforcing a hard floor of `IQ4_XS` (strictly eliminates `IQ1`, `IQ2`, and `IQ3`). |
| `wide_ladder` | Custom / MiB | Generic wide-ladder profile allowing all lower rungs down to `IQ1_S` for manual MiB targets. |

### Custom Recipe Files

To run a custom quantization recipe, create a JSON or YAML file (e.g. `recipes/custom_q4.json`):

```json
{
  "name": "CUSTOM_Q4",
  "target_bpw": 4.85,
  "force_wide_ladder": true,
  "ladders": {
    "attn_proj": ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"],
    "ffn_down":  ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K"],
    "ffn_gate_up": ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"]
  },
  "bases": {
    "attn_proj": "IQ4_XS",
    "ffn_down": "IQ4_XS",
    "ffn_gate_up": "IQ4_XS"
  }
}
```

Run with:
```bash
python -m iq_hybrid.cli --model "Models/model-bf16.gguf" --imatrix "iMatrix/model.imatrix.gguf" --profile "recipes/custom_q4.json" --run
```

---

## Configuration Reference

### `config.json`
Default configuration file at repository root:
```json
{
  "solver": "greedy",
  "wide_ladder": false,
  "imatrix_method": "add",
  "imatrix_output_dir": "iMatrix",
  "verbose": false
}
```

### Environment Variables (`.env`)
```ini
LLAMA_QUANTIZE_PATH=C:\path\to\llama.cpp\build\bin\llama-quantize.exe
OUTPUT_DIR=Models\quantized
```

---

## Extension Guide

### 1. Adding or Modifying Quantization Ladders
Universal ladder definitions, ceilings, base rungs, MTP safety rules, and vocabulary embedding scaling are centralized in [`src/iq_hybrid/architectures/base.py`](src/iq_hybrid/architectures/base.py).

* Modify `DEFAULT_LADDERS` in `base.py` to change default progression paths.
* Modify `get_fixed_tensors` in `BaseArchitecture` to adjust global safety anchors (e.g. MTP blocks to `Q8_0` or normalization layers to `F16`).

### 2. Adding a New Model Architecture
1. Create `src/iq_hybrid/architectures/<arch_name>.py` inheriting from `BaseArchitecture`:
   ```python
   from iq_hybrid.architectures.base import BaseArchitecture
   from iq_hybrid.core.constants import strip_weight
   from iq_hybrid.core.types import TensorClass, TensorName

   MY_TENSOR_MAP: dict[str, TensorClass] = {
       "attn_q": "attn_proj",
       "attn_k": "attn_proj",
       "ffn_gate": "ffn_gate_up",
       "ffn_down": "ffn_down",
       "norm": "norms",
   }

   class MyModelArchitecture(BaseArchitecture):
       name: str = "my_model"
       depth_alpha_max: float = 0.85
       depth_decay_rate: float = 2.0

       def classify_tensor(self, name: TensorName) -> TensorClass:
           clean = strip_weight(name)
           suffix = clean.split(".")[-1]
           return MY_TENSOR_MAP.get(suffix, "default")
   ```

2. Register the class in [`src/iq_hybrid/architectures/registry.py`](src/iq_hybrid/architectures/registry.py):
   * Add the class to `_ARCHITECTURES`.
   * Add the detection rule in `detect_architecture_strategy`.

---

## Quality Assurance & Testing

Run unit tests and static analysis:

```bash
# Execute test suite
pytest tests/ -v

# Run linter and formatting checks
ruff check src tests
ruff format src tests
```

---

## License

This project is licensed under the terms of the [Apache 2.0 License](LICENSE).
