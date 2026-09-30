"""E0 common utilities: config, seeding, IO, timing, GPU sampling."""
import json
import os
import random
import threading
import time

import numpy as np
import yaml

E0_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(E0_ROOT, "data")
THIRD_PARTY = os.path.join(E0_ROOT, "third_party")
RESULTS_DIR = os.path.join(E0_ROOT, "results")

VADCLIP_DIR = os.path.join(THIRD_PARTY, "VadCLIP")
VADCLIP_SRC = os.path.join(VADCLIP_DIR, "src")
VADCLIP_LIST = os.path.join(VADCLIP_DIR, "list")
HOLMES_DIR = os.path.join(THIRD_PARTY, "HolmesVAU")
HOLMES_MODEL_PATH = os.path.join(DATA_DIR, "HolmesVAU-2B")
VADCLIP_CKPT = os.path.join(DATA_DIR, "model_ucf.pth")
CLIP_FEATURES_DIR = os.path.join(DATA_DIR, "UCFClipFeatures")
UCF_TEST_VIDEOS = os.path.join(DATA_DIR, "ucf_test_videos")

UCF_FPS = 30.0  # UCF-Crime source fps
SNIPPET_FRAMES = 16  # VadCLIP feature granularity


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_result(out_path, config, payload):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = {"config": config, **payload}
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {out_path}")


def percentiles(latencies_ms):
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


class GpuUtilSampler(threading.Thread):
    """Sample GPU utilization + memory via NVML in a background thread."""

    def __init__(self, gpu_index=0, interval=0.1):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()

    def run(self):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        while not self._stop_event.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(handle)
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.samples.append({"util": u.gpu, "mem_used_mb": m.used / 1e6,
                                 "t": time.time()})
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)
        if not self.samples:
            return {"util_mean": 0.0, "util_p95": 0.0, "mem_used_mb_max": 0.0}
        utils = np.array([s["util"] for s in self.samples])
        mems = np.array([s["mem_used_mb"] for s in self.samples])
        return {"util_mean": float(utils.mean()), "util_p95": float(np.percentile(utils, 95)),
                "mem_used_mb_max": float(mems.max())}


def nvidia_smi_mem_mb(gpu_index=0):
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu_index)])
    return float(out.decode().strip())


# ---------- UCF-Crime test-split metadata ----------

UCF_LABEL_MAP = {
    'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson',
    'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion',
    'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
    'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
    'Vandalism': 'Vandalism',
}


def load_test_list():
    """Return list of (video_id, category, feature_path) for the 290 test videos,
    in the exact order of VadCLIP's ucf_CLIP_rgbtest.csv (GT order depends on it)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(VADCLIP_LIST, "ucf_CLIP_rgbtest.csv"))
    rows = []
    for _, r in df.iterrows():
        base = os.path.basename(r["path"])           # e.g. Abuse028_x264__5.npy
        subdir = os.path.basename(os.path.dirname(r["path"]))  # Abuse or Testing_Normal_Videos_Anomaly
        video_id = base.replace("__5.npy", "")       # Abuse028_x264
        category = r["label"]
        feat_path = os.path.join(CLIP_FEATURES_DIR, subdir, base)
        rows.append({"video_id": video_id, "category": category,
                     "feature_path": feat_path})
    return rows


def video_file_for(video_id, category):
    """Locate the extracted test video file for a given video id."""
    for sub in ("abnormal", "normal"):
        p = os.path.join(UCF_TEST_VIDEOS, sub, f"{video_id}.mp4")
        if os.path.exists(p):
            return p
    # fallback: search anywhere under the video root
    import glob
    hits = glob.glob(os.path.join(UCF_TEST_VIDEOS, "**", f"{video_id}.mp4"),
                     recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(video_id)


def load_frame_gt():
    """Frame-level GT for the concatenated test set (order = test csv order)."""
    return np.load(os.path.join(VADCLIP_LIST, "gt_ucf.npy"))


def load_temporal_annotations():
    """Parse Temporal_Anomaly_Annotation.txt ->
    {video_id: [(start_frame, end_frame), ...]} (1-based, inclusive, as in file)."""
    ann = {}
    with open(os.path.join(VADCLIP_LIST, "Temporal_Anomaly_Annotation.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            vid = parts[0].replace(".mp4", "")
            # file format: name class start1 end1 start2 end2  (-1 -1 when absent)
            segs = []
            for i in range(2, len(parts), 2):
                s, e = int(parts[i]), int(parts[i + 1])
                if s >= 0 and e >= 0:
                    segs.append((s, e))
            ann[vid] = segs
    return ann
