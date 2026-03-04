# PEARLS: Python Implementation

Python translation of PEARLS algorithm from MATLAB.

## Usage

```python
from pearls import pearls
import scipy.signal as signal

# Load audio, decimate, and create analytic signal
y = signal.decimate(audio, 4)
z = signal.hilbert(y)

# Run PEARLS
w_rls_hist, fpgrid_hist = pearls(
    d=z, lambda_val=0.995, rls_xi=10000, 
    Lmax=10, fs=fs/4, fmin=80, fmax=400, fdist=5
)
```

## MATLAB to Python Translation Summary

Files Translated:

- PEARLS.m → pearls.py (main algorithm)
- dictionaryUpdate.m → `dictionary_update()`
- interval_search_anls.m → `interval_search_anls()`
- phaseUpdate.m → `phase_update()`
- proximal_gradient_update.m → `proximal_gradient_update()`
- rls_update.m → `rls_update()`
- mail.m → example.py

Key Changes:

1. Array indexing: MATLAB 1-based → Python 0-based
2. Array order: Used `order='F'` in reshape for column-major
3. Keywords: `lambda` → `lambda_val`
4. Functions: `norms()` → `np.linalg.norm()`, `find()` → `np.where()`
5. Complex ops: `A'` → `A.conj().T`

All functionality preserved. Ready to use!

All matrix operations maintain MATLAB semantics

Usage:
  python example.py

Dependencies:
  numpy, scipy, matplotlib

## Run Example

```bash
python example.py
```

See code comments for detailed documentation.
