# Setting up the training machine (Windows + RTX 5050)

Do these once, in PowerShell, on the laptop with the GPU.

## 1. NVIDIA driver
Install the latest Game Ready or Studio driver from the NVIDIA App or nvidia.com.
Check it: `nvidia-smi` should list the RTX 5050 and a CUDA version of 12.8 or higher.

## 2. Python
Install **Python 3.12** from python.org (tick "Add python.exe to PATH").
3.12 has the widest support across PyTorch and the audio libraries.

```powershell
git clone <your-ada-repo-url> ada
cd ada
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 3. PyTorch for the RTX 50-series
The RTX 5050 (Blackwell) needs a **CUDA 12.8** build. The default `pip install torch`
may give a CPU-only or older build, so install it explicitly:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 4. ADA with training extras
```powershell
pip install -e ".[train,export,dev]"
```

## 5. Check everything
```powershell
python scripts/check_gpu.py
pytest
```
`check_gpu.py` must end with "All good". If it says a kernel did not run, repeat step 3.

## Tips for long training runs
- Plug in the charger and set Windows power mode to **Best performance**.
- Settings → System → Power & battery → Screen and sleep → sleep when plugged in: **Never**.
- Keep the laptop on a hard surface or cooling pad; watch temps in the NVIDIA App.
- Close games/browsers using the GPU. Check usage with `nvidia-smi`.
- "CUDA out of memory": halve `per_device_train_batch_size` and double
  `gradient_accumulation_steps` in the config. Effective batch size stays the same.
- Training saves checkpoints every `save_steps`, so a crash loses at most that much work.
