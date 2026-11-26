"""Step 1 on your laptop: confirm PyTorch can actually train on the RTX 5050.

Run:  python scripts/check_gpu.py
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. See docs/SETUP_WINDOWS.md, step 3.")
        return 1

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Built for CUDA: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\nFAIL: CUDA not available. Update the NVIDIA driver and install the cu128 build.")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU           : {name}  (compute {major}.{minor}, {vram_gb:.1f} GB VRAM)")
    print(f"bf16 support  : {torch.cuda.is_bf16_supported()}")

    # RTX 50-series (Blackwell) is compute capability 12.x and needs a CUDA 12.8+ build.
    # A wrong build only fails when a kernel actually runs, so run one.
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        print(f"\nFAIL: a GPU kernel did not run: {exc}")
        print("This usually means PyTorch was not built for CUDA 12.8+. Reinstall with:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"Matmul test   : OK ({y:.1f})")
    print("\nAll good: this machine can train ADA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
