@echo off
set "PYTHONPATH=src"

:: --solver mckp ^    --run     --wide-ladder ^ --verbose ^   .\venv\Scripts\activate::

python -m iq_hybrid ^
  --model C:\Users\USER\Models\model.gguf ^
  --imatrix iMatrix\model-iMatrix.gguf ^
  --profile quality_ladder ^
  --solver mckp ^
  --size 6310
  --run