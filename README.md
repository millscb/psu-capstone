# SWENG480
Capstone Project

## Environment Setup (Conda)

This project uses a local (project-directory) Conda environment defined by `environment.yml` with a `prefix` pointing to `./.venv/rl_env`.

### 1. Install Miniconda (Recommended)
Download from: https://docs.conda.io/en/latest/miniconda.html (Windows 64-bit, Python 3.11) or use winget (Windows PowerShell):

```powershell
winget install -e --id Anaconda.Miniconda3
```

Initialize Conda for PowerShell:

```powershell
conda init powershell
```
Close and reopen the terminal (or run a new VS Code terminal). Verify:

```powershell
conda --version
```

### 2. Create the Environment

From the project root (`SWENG480/`):

```powershell
conda env create -f environment.yml
```

Because `environment.yml` uses a `prefix`, Conda will create the environment at:
```
./.venv/rl_env
```

### 3. Activate the Environment

Activation (path-based):
```powershell
conda activate (Resolve-Path .\.venv\rl_env)
```

If you prefer a named environment (optional):
1. Edit `environment.yml`: replace the first line `prefix: ./.venv/rl_env` with `name: rl_env`.
2. Recreate:
	```powershell
	conda env remove --name rl_env  # (only if it exists already)
	conda env create -f environment.yml
	conda activate rl_env
	```

### 4. Verify Core Packages

```powershell
python -c "import torch, tensorflow, gymnasium, pandas; print('OK')"
```

### 5. Updating the Environment

After editing `environment.yml` (adding/removing dependencies):
```powershell
conda env update -f environment.yml --prune
```

### 6. Jupyter Kernel (Optional)

```powershell
python -m ipykernel install --user --name rl_env --display-name "RL Env"
```

---

## Training & Running the CartPole Policy Gradient

Main training script: `src/test/cartPole.py`

CLI arguments:
```
--iterations             Number of policy update iterations (default 200)
--episodes-per-update    Episodes collected per iteration (default 5)
--max-steps              Max steps per episode (default 200)
--discount               Discount factor gamma (default 0.95)
--no-render              Disable environment rendering window
--save-name              Output model filename inside models/ (default cartpole_policy.keras)
--load-model             Path to existing .keras model to resume training
```

### Example Commands

Basic training with rendering:
```powershell
python .\src\test\cartPole.py
```

Headless (faster) short run:
```powershell
python .\src\test\cartPole.py --iterations 20 --episodes-per-update 10 --no-render --save-name run1.keras
```

Resume training from a saved model:
```powershell
python .\src\test\cartPole.py --load-model models\run1.keras --iterations 50 --no-render --save-name run1_cont.keras
```

Minimal smoke test:
```powershell
python .\src\test\cartPole.py --iterations 1 --episodes-per-update 1 --no-render --save-name quick.keras
```

### Early Stopping
Training stops early if mean episode length ≥ 200 for 2 consecutive iterations (CartPole “solved” criterion). This logic is inside `train()`.

---

## Saved Models

Saved models are written to `models/` in Keras `.keras` format (architecture + weights + optimizer state). Example load:

```python
from tensorflow import keras
model = keras.models.load_model("models/cartpole_policy.keras")
proba_left = model([[0.0, 0.1, 0.0, -0.05]])
print(float(proba_left.numpy()[0][0]))
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `conda` not recognized | Shell not initialized | `conda init powershell` then new terminal |
| Long solve time | Heavy TF+Torch combo | Try creating env without one first, then `pip install` the other |
| Render window slow | Human rendering each step | Use `--no-render` while training |
| No GPU usage | CPU build installed | Install GPU builds (PyTorch via official index URL, ensure CUDA drivers) |
| Model not improving | High variance PG | Increase episodes per update or add baseline (future enhancement) |

### Clean Recreate
```powershell
conda deactivate
Remove-Item -Recurse -Force .\.venv\rl_env
conda env create -f environment.yml
conda activate (Resolve-Path .\.venv\rl_env)
```

---

## Optional Improvements (Not Yet Implemented)
* Baseline / advantage normalization
* TensorBoard logging
* Checkpoint best model
* Seed reproducibility utilities

---

## At a Glance

```powershell
# Setup
conda env create -f environment.yml
conda activate (Resolve-Path .\.venv\rl_env)

# Train
python .\src\test\cartPole.py --no-render --iterations 50

# Resume later
python .\src\test\cartPole.py --load-model models\cartpole_policy.keras --iterations 25 --no-render
```

---

## License
Add license info here if applicable.
