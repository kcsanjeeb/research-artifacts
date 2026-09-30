"""E0 Experiment 1 -- Capacity: VLM clips/sec on 4x V100.

Unit of work (clip spec), fixed from the HolmesVAU repo's own inference path
(inference.py + holmesvau/holmesvau_utils.py):
  - 12 frames, uniformly sampled (get_index, num_segments=12; the repo's default
    non-ATS path for short clips)
  - 448x448, single tile (dynamic_preprocess max_num=1)
  - prompt: "Could you specify the anomaly events present in the video?"
  - greedy decoding (do_sample=False)
  - max_new_tokens: set by config (64 default; the repo demo uses 1024 -- see REPORT
    section 4 for the rationale; 64 bounds a single anomaly *judgement*)

Modes:
  python capacity.py --config X --out Y            full experiment
  python capacity.py --worker --gpu G --batch B --duration S --out W.json
      single replica worker (used by the replica / multi-GPU sweeps)
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CLIP_SPEC = {
    "frames": 12,
    "resolution": "448x448",
    "tile": "single (max_num=1)",
    "sampling": "uniform (repo default get_index)",
    "prompt": "Could you specify the anomaly events present in the video?",
    "do_sample": False,
}


# ---------------------------------------------------------------- model utils

def load_holmes(model_path, device):
    """Load HolmesVAU-2B with the V100 (sm_70) patches: fp16, no flash-attn."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # --- sm_70 patches ---
    cfg.use_flash_attn = False
    if hasattr(cfg, "llm_config") and isinstance(cfg.llm_config, dict):
        cfg.llm_config["attn_implementation"] = "eager"
    cfg.torch_dtype = "float16"

    model = AutoModel.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True).eval()
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                        use_fast=False)
    load_s = time.time() - t0
    return model, tok, load_s


def build_clip_pool(model_path, n_videos, seed):
    """Preprocess n_videos test clips -> CPU tensors, following the repo's
    get_pixel_values() exactly."""
    sys.path.insert(0, common.HOLMES_DIR)
    from decord import VideoReader, cpu
    from holmesvau.internvl_utils import build_transform, dynamic_preprocess, get_index
    import torch
    from PIL import Image

    common.set_seed(seed)
    rows = common.load_test_list()
    import random
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n_videos, len(rows)))

    transform = build_transform(input_size=448)
    clips = []
    for r in picks:
        vf = common.video_file_for(r["video_id"], r["category"])
        vr = VideoReader(vf, ctx=cpu(0), num_threads=1)
        idx = get_index(bound=None, fps=float(vr.get_avg_fps()),
                        max_frame=len(vr) - 1, first_idx=0, num_segments=12)
        idx = [int(i) for i in idx]
        pvs = []
        for fi in idx:
            img = Image.fromarray(vr[fi].asnumpy()).convert("RGB")
            tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
            pvs.extend([transform(t) for t in tiles])
        pixel_values = torch.stack(pvs)          # (12, 3, 448, 448)
        clips.append({"pixel_values": pixel_values,
                      "num_patches": [1] * pixel_values.shape[0],
                      "video_id": r["video_id"]})
    return clips


def batch_generate(model, tok, pixel_values_gpu, questions, num_patches_list, gen_cfg):
    """InternVL2's batch_chat only replaces the first <image> per question;
    this is the same logic with the per-frame replacement loop from chat()."""
    import torch
    sys.path.insert(0, common.HOLMES_MODEL_PATH)
    from conversation import get_conv_template

    IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
    model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CTX)

    queries = []
    pos = 0
    for q in questions:
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], q)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        n_ph = query.count('<image>')
        for _ in range(n_ph):
            np_ = num_patches_list[pos]
            pos += 1
            image_tokens = IMG_START + IMG_CTX * model.num_image_token * np_ + IMG_END
            query = query.replace('<image>', image_tokens, 1)
        queries.append(query)

    tok.padding_side = 'left'
    inputs = tok(queries, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(model.device)
    attn = inputs['attention_mask'].to(model.device)
    eos = tok.convert_tokens_to_ids(template.sep)
    cfg = dict(gen_cfg)
    cfg['eos_token_id'] = eos
    with torch.no_grad():
        out = model.generate(pixel_values=pixel_values_gpu, input_ids=input_ids,
                             attention_mask=attn, **cfg)
    return out


def make_questions(batch_clips):
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(CLIP_SPEC["frames"])])
    return [video_prefix + CLIP_SPEC["prompt"] for _ in batch_clips]


# ---------------------------------------------------------------- worker mode

def run_worker(gpu, batch, duration, out_path, model_path, max_new_tokens):
    """Load one replica, serve batches for `duration` seconds, count clips."""
    import torch
    common.set_seed(0)
    device = torch.device(f"cuda:{gpu}")
    model, tok, load_s = load_holmes(model_path, device)
    pool = build_clip_pool(model_path, n_videos=16, seed=0)

    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
    lat = []
    n_clips = 0
    i = 0
    # warmup
    for _ in range(3):
        bc = [pool[(i + j) % len(pool)] for j in range(batch)]
        i += batch
        pv = torch.cat([c["pixel_values"] for c in bc]).to(torch.float16).to(device)
        npl = [p for c in bc for p in c["num_patches"]]
        batch_generate(model, tok, pv, make_questions(bc), npl, gen_cfg)
    torch.cuda.synchronize()
    t_start = time.time()
    while True:
        bc = [pool[(i + j) % len(pool)] for j in range(batch)]
        i += batch
        pv = torch.cat([c["pixel_values"] for c in bc]).to(torch.float16).to(device)
        npl = [p for c in bc for p in c["num_patches"]]
        torch.cuda.synchronize()
        t0 = time.time()
        batch_generate(model, tok, pv, make_questions(bc), npl, gen_cfg)
        torch.cuda.synchronize()
        lat.append((time.time() - t0) * 1000)
        n_clips += batch
        if time.time() - t_start >= duration:
            break
    wall = time.time() - t_start
    res = {"gpu": gpu, "batch": batch, "clips": n_clips, "wall_s": wall,
           "clips_per_s": n_clips / wall,
           "vram_torch_mb": torch.cuda.max_memory_allocated() / 1e6,
           **common.percentiles(lat)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)


# ------------------------------------------------------------- sweep drivers

def spawn_workers(gpus, batch, duration, tag):
    """Spawn one worker per gpu entry (same gpu may repeat = replicas)."""
    procs, outs = [], []
    for k, g in enumerate(gpus):
        out = os.path.join(common.RESULTS_DIR, f"worker_{tag}_{k}.json")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        p = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", "--gpu", "0",
             "--batch", str(batch), "--duration", str(duration),
             "--out", out, "--tag", f"{tag}_{k}"],
            env=env,
            stdout=open(os.path.join(common.E0_ROOT, "logs", f"worker_{tag}_{k}.log"), "w"),
            stderr=subprocess.STDOUT)
        procs.append(p)
        outs.append(out)
    for p in procs:
        p.wait()
    results = []
    for p, out in zip(procs, outs):
        if p.returncode != 0 or not os.path.exists(out):
            return None  # some worker OOMed / crashed
        with open(out) as f:
            results.append(json.load(f))
    return results


def aggregate(results):
    clips = sum(r["clips"] for r in results)
    wall = max(r["wall_s"] for r in results)
    return {"replicas": len(results), "total_clips": clips, "wall_s": wall,
            "agg_clips_per_s": clips / wall,
            "per_replica": [{"clips_per_s": r["clips_per_s"], "p50_ms": r["p50_ms"],
                             "p95_ms": r["p95_ms"], "vram_torch_mb": r["vram_torch_mb"]}
                            for r in results]}


def run_experiment(cfg, out_path):
    import torch
    common.set_seed(cfg.get("seed", 0))
    model_path = cfg["model_path"]
    device = torch.device("cuda:0")
    mnt = cfg["max_new_tokens"]

    # ---- 1. load on one GPU ----
    model, tok, load_s = load_holmes(model_path, device)
    torch.cuda.reset_peak_memory_stats()
    vram_after_load_torch = torch.cuda.max_memory_allocated() / 1e6
    vram_after_load_smi = common.nvidia_smi_mem_mb(0)

    pool = build_clip_pool(model_path, cfg.get("pool_videos", 64), cfg.get("seed", 0))
    gen_cfg = dict(max_new_tokens=mnt, do_sample=False)

    # ---- 2. batch sweep ----
    batch_results = []
    for b in cfg["batch_sweep"]:
        try:
            lat = []
            sampler = common.GpuUtilSampler(0)
            sampler.start()
            torch.cuda.reset_peak_memory_stats()
            n_iters = cfg["warmup_iters"] + cfg["timed_iters"]
            for it in range(n_iters):
                bc = [pool[(it * b + j) % len(pool)] for j in range(b)]
                pv = torch.cat([c["pixel_values"] for c in bc]).to(torch.float16).to(device)
                npl = [p for c in bc for p in c["num_patches"]]
                torch.cuda.synchronize()
                t0 = time.time()
                batch_generate(model, tok, pv, make_questions(bc), npl, gen_cfg)
                torch.cuda.synchronize()
                dt = (time.time() - t0) * 1000
                if it >= cfg["warmup_iters"]:
                    lat.append(dt)
            util = sampler.stop();  util and None
            total_s = sum(lat) / 1000
            rec = {"batch": b,
                   "clips_per_s": b * len(lat) / total_s,
                   **common.percentiles(lat),
                   "vram_torch_mb": torch.cuda.max_memory_allocated() / 1e6,
                   "vram_smi_mb": common.nvidia_smi_mem_mb(0),
                   **util}
            batch_results.append(rec)
            print(f"batch={b}: {rec['clips_per_s']:.2f} clips/s "
                  f"p50={rec['p50_ms']:.0f}ms vram={rec['vram_torch_mb']:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            batch_results.append({"batch": b, "oom": True})
            torch.cuda.empty_cache()
            print(f"batch={b}: OOM, stopping sweep")
            break

    valid = [r for r in batch_results if not r.get("oom")]
    best = max(valid, key=lambda r: r["clips_per_s"])
    best_batch = best["batch"]

    del model
    torch.cuda.empty_cache()
    time.sleep(5)

    # ---- 3. replica sweep on one GPU (workers each own a full model copy) ----
    replica_results = []
    for r in cfg["replica_sweep"]:
        res = spawn_workers([0] * r, best_batch, cfg["worker_duration_s"], f"rep{r}")
        if res is None:
            replica_results.append({"replicas": r, "failed": True})
            print(f"replicas={r}: worker failed (OOM?), stopping")
            break
        agg = aggregate(res)
        replica_results.append(agg)
        print(f"replicas={r}: {agg['agg_clips_per_s']:.2f} clips/s aggregate")
        time.sleep(5)

    ok_reps = [x for x in replica_results if not x.get("failed")]
    best_rep = max(ok_reps, key=lambda x: x["agg_clips_per_s"])
    best_replicas = best_rep["replicas"]

    # ---- 4. multi-GPU: best config on all 4 GPUs ----
    ngpus = cfg.get("num_gpus", 4)
    gpus = [g for g in range(ngpus) for _ in range(best_replicas)]
    res = spawn_workers(gpus, best_batch, cfg["worker_duration_s"], "multi")
    pool_tp = aggregate(res)["agg_clips_per_s"] if res else None

    payload = {
        "clip_spec": {**CLIP_SPEC, "max_new_tokens": mnt},
        "models": {
            "HolmesVAU-2B": {
                "load_s": load_s,
                "vram_gb_after_load_torch": vram_after_load_torch / 1000,
                "vram_gb_after_load_smi": vram_after_load_smi / 1000,
                "dtype": "float16",
                "attn": "eager (flash-attn disabled for sm_70)",
                "batch_sweep": batch_results,
                "best_batch": best_batch,
                "replica_sweep": replica_results,
                "replicas_per_gpu": best_replicas,
                "pool_throughput_clips_per_s_4gpu": pool_tp,
            }
        },
    }
    common.save_result(out_path, cfg, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--tag", default="w")
    args = ap.parse_args()

    if args.worker:
        cfg = common.load_config(args.config) if args.config else {}
        run_worker(args.gpu, args.batch, args.duration, args.out,
                   cfg.get("model_path", common.HOLMES_MODEL_PATH),
                   cfg.get("max_new_tokens", 64))
    else:
        cfg = common.load_config(args.config)
        run_experiment(cfg, args.out)


if __name__ == "__main__":
    main()
