"""Record the E0 environment: hardware, driver, CUDA, torch, commit hashes."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"error: {e}"


def git_head(path):
    return sh(f"git -C {path} rev-parse HEAD")


def main():
    import torch
    env = {
        "gpus": sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                   "--format=csv,noheader").splitlines(),
        "nvcc": sh("nvcc --version | tail -1"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_capability": [torch.cuda.get_device_capability(i)
                              for i in range(torch.cuda.device_count())],
        "commits": {
            "HolmesVAU": git_head(os.path.join(common.THIRD_PARTY, "HolmesVAU")),
            "LAVIDA": git_head(os.path.join(common.THIRD_PARTY, "LAVIDA")),
            "VadCLIP": git_head(os.path.join(common.THIRD_PARTY, "VadCLIP")),
            "PEL4VAD": git_head(os.path.join(common.THIRD_PARTY, "PEL4VAD")),
        },
        "data_sources": {
            "ucf_test_videos": "hf datasets backseollgi/UCF-Crime_TEST_SET "
                               "(UCF-Crime_TEST_SET.tar.gz, 7841518071 bytes, 290 mp4)",
            "ucf_clip_features": "VadCLIP official UCFClipFeatures.zip via "
                                 "myzhao1999/ucf-crime-clip-features (hf mirror; "
                                 "byte-identical size to the official OneDrive zip)",
            "vadclip_checkpoint": "model_ucf.pth, official VadCLIP OneDrive share",
            "holmesvau_weights": "hf ppxin321/HolmesVAU-2B (via hf-mirror.com)",
        },
        "network_notes": "huggingface.co, OneDrive API, Dropbox, crcv.ucf.edu, "
                         "Google Drive, Kaggle and openaipublic (Azure) unreachable "
                         "from the server; hf-mirror.com and SharePoint "
                         "download.aspx used instead. LAVIDA released no "
                         "checkpoints or usage instructions (commit 56b3058) - "
                         "Experiment 1 runs HolmesVAU-2B only.",
    }
    out = os.path.join(common.RESULTS_DIR, "environment.json")
    with open(out, "w") as f:
        json.dump(env, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
