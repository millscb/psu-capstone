# NuPIC (Legacy) Build and Install Guide on Windows

This guide follows the legacy NuPIC README you provided and adapts it for a modern Windows setup. NuPIC is in maintenance mode and many binaries are old. Expect potential friction.

Important notes:
- Legacy software: NuPIC 1.0.x is no longer actively maintained. Wheels may be unavailable for modern Python versions.
- Recommended alternative: If you don’t strictly need NuPIC’s legacy API, use `htm.core` (modern, maintained). See `environment-htmcore.yml`.
- Best chance on Windows: Python 3.6 with pinned NumPy; try pip wheels first. If wheels aren’t available, a source build on Windows is often painful; consider WSL (Ubuntu) instead.

## 1) Attempt Binary Install (Windows 64-bit)

We provide `environment-nupic.yml` to create a Python 3.6 environment and install NuPIC via pip.

Steps:

1. Create and activate the env
```powershell
conda env create -f environment-nupic.yml
conda activate nupic36
```

2. Ensure tooling is up-to-date inside the env
```powershell
python -m pip install --upgrade pip
```

3. Install NuPIC
```powershell
pip install nupic==1.0.5
```

4. Smoke test
```powershell
python -c "import nupic; print('NuPIC version:', nupic.__version__)"
python .\src\nupic_smoke_test.py
```

Troubleshooting:
- If you see wheel/ABI errors, pin NumPy lower and reinstall:
```powershell
pip install numpy==1.18.5 --force-reinstall
pip install --no-cache-dir --force-reinstall nupic==1.0.5
```
- If import fails with DLL errors, fully close terminals, reactivate env, and retry. Ensure 64-bit Python.
- If Windows wheels are missing for your combo, consider WSL (Ubuntu) or use `htm.core` instead.

## 2) Source Install (Advanced; often fragile on Windows)

The legacy README suggests building `nupic.core` (and `nupic.bindings`) prior to installing `nupic` from source.

High-level requirements:
- Python 3.6 (or legacy Python 2.7 per very old directions, not recommended due to EOL)
- C++11 compiler toolchain
  - Windows: Visual Studio Build Tools (MSVC v142 or compatible), 64-bit
- CMake
- Git
- (Historical) capnproto and SWIG for older code paths; exact versions are tricky on Windows

Process overview (guidance only):
1. Prepare a build-friendly environment (example names; adjust as needed):
```powershell
conda create -n nupic-src python=3.6 pip cmake git -y
conda activate nupic-src
python -m pip install --upgrade pip setuptools wheel
```
2. Obtain and build nupic.core/nupic.bindings (the exact steps vary by commit and are not guaranteed to work on Windows today). If you succeed in building and installing `nupic.bindings` into the env, then:
```powershell
# In the NuPIC repo root
pip install .  # or: pip install -e . for editable install
```
3. Test:
```powershell
python -c "import nupic; print('NuPIC version:', nupic.__version__)"
```

Given the difficulty of building on Windows now, we recommend:
- Try WSL (Ubuntu) with Python 3.6 and system toolchain
- Or switch to `htm.core` where builds are active and documented

## 3) Modern Alternative: htm.core

Create and test the modern environment:
```powershell
conda env create -f environment-htmcore.yml
conda activate htmcore
python -c "import htm; print('htm.core version:', htm.__version__)"
```

## 4) FAQ

- Q: Can I use Python 3.11 with NuPIC?
  - A: Almost certainly no. Legacy NuPIC wheels target old Pythons (2.7 / 3.6).
- Q: Can I make NuPIC run on Windows?
  - A: Sometimes via binary wheels (pip) with Python 3.6; otherwise source builds are non-trivial.
- Q: Why use `python -m pip`?
  - A: Ensures you’re installing into the currently active interpreter (right Conda env).

## 5) Commands Recap

Binary attempt:
```powershell
conda env create -f environment-nupic.yml
conda activate nupic36
python -m pip install --upgrade pip
pip install nupic==1.0.5
python -c "import nupic; print('NuPIC version:', nupic.__version__)"
```

If failing, try lower NumPy:
```powershell
pip install numpy==1.18.5 --force-reinstall
pip install --no-cache-dir --force-reinstall nupic==1.0.5
```

Modern path:
```powershell
conda env create -f environment-htmcore.yml
conda activate htmcore
python -c "import htm; print('htm.core version:', htm.__version__)"
```
