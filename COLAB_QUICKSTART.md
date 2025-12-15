# 🚀 CLSO Colab Quick Start Guide

## Import Fix for Google Colab

If you encounter `ModuleNotFoundError`, use this code block:

```python
# Setup CLSO in Colab
import sys
sys.path.insert(0, '/content/CLSO-training')

# Run setup helper
!python /content/CLSO-training/colab_setup.py
```

## Alternative Manual Import

If the setup helper doesn't work, try manual imports:

```python
import sys
sys.path.insert(0, '/content/CLSO-training')

# Option 1: Package imports (preferred)
from src import BasisLibrary, CrystallineGPT2, GeneticOptimizer

# Option 2: Direct imports (if Option 1 fails)
from src.basis_library import BasisLibrary
from src.crystalline_model import CrystallineGPT2  
from src.genetic_optimizer import GeneticOptimizer
```

## Quick Test Command

Once imports work, run using the wrapper script (recommended):

```bash
python /content/CLSO-training/run_training.py \
    --generations 5 \
    --pop-size 16 \
    --batch-size 16 \
    --device cuda \
    --output-dir experiments/quick_test_gpu
```

Or run directly (also works now):

```bash
cd /content/CLSO-training
python src/train_clso.py \
    --generations 5 \
    --pop-size 16 \
    --batch-size 16 \
    --device cuda \
    --output-dir experiments/quick_test_gpu
```

## Full Workflow

1. **Clone repo**: `!git clone https://github.com/codenlighten/CLSO-training.git`
2. **Install deps**: `!pip install -q torch transformers datasets nvidia-ml-py3`
3. **Setup imports**: Run `colab_setup.py` 
4. **Select A100**: Runtime → Change runtime type → A100
5. **Run training**: Execute training commands

## Common Issues

### ModuleNotFoundError
- **Cause**: Python can't find the `src` package
- **Fix**: Use `sys.path.insert(0, '/content/CLSO-training')` before imports

### Import from src.X fails
- **Cause**: Relative imports not working
- **Fix**: Use the setup helper or manual direct imports

### CUDA not available
- **Cause**: GPU not selected
- **Fix**: Runtime → Change runtime type → Hardware accelerator → GPU

## Need Help?

Check the main README.md or open an issue on GitHub:
https://github.com/codenlighten/CLSO-training/issues

---

**Author:** Gregory J Ward  
**SmartLedger.Technology | Codenlighten.org**
