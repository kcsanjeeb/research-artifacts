"""T4 batch microbenchmark: Qwen2.5-VL-7B narration jobs at fixed batch sizes.

Measures GPU-s/job (wall time on an exclusive GPU), throughput, and peak VRAM
for batch sizes {1,2,4,8,16} at 8 frames/job and 4 frames/job, plus mixed
4+8-frame batches. Adapted from ~/e1/narration_spike/narrate.py (schema variant):
same frame prep (448x336, multiples of 28), same sampling (T=0.2, top_p=0.9,
max_new_tokens=512).

V100/sm_70 requires two memory-only patches (v100_patches.py, equivalence-
verified vs unpatched at bs=1, greedy outputs identical):
  1. vision sdpa builds a dense [1,N,N] mask over the concatenated image batch
     -> math backend -> O(N^2) scores -> OOM at bs>=2. Patched to per-segment
     sdpa over cu_seqlens (block-diagonal mask -> identical math, O(N)).
  2. forward computes full-sequence fp32 logits (2 GiB at bs=2, ~24 GiB at
     bs=16) though generate only uses logits[:, -1, :]. Patched lm_head to
     last-token-only.

Usage:
  python bench_t4.py --gpu 1 --plan 8f     # batch sizes 1..16, 8 frames/job
  python bench_t4.py --gpu 2 --plan 4f     # 4 frames/job
  python bench_t4.py --gpu 3 --plan mixed  # mixed 4f+8f batches, b8 and b16
"""
import argparse, json, os, sys, time

import torch  # noqa: E402

BENCH = os.path.expanduser("~/e1/t4_batch_microbench")
sys.path.insert(0, BENCH)
import v100_patches  # noqa: E402

SPIKE = os.path.expanduser("~/e1/narration_spike")
EVENTS = os.path.join(SPIKE, "events.jsonl")
FRAMES_DIR = os.path.join(SPIKE, "frames")
QWEN_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
SCHEMA_INSTR = (
    "These 8 frames are sampled uniformly from a single surveillance video "
    "segment. Describe only what is directly visible in the frames. Respond "
    "with a single strict JSON object and no other text, with exactly these "
    'keys: "actors" (list of visible people/entities, e.g. "man in red '
    'shirt"), "actions" (list of observable actions), "objects" (list of '
    'notable visible objects), "location_cues" (list of visible setting '
    'cues), "summary" (one sentence describing what happens). If nothing is '
    "visible for a key, use an empty list. Use at most 6 entries per list. Do not speculate about intent, "
    "danger, or abnormality."
)
SCHEMA_INSTR_4 = SCHEMA_INSTR.replace("These 8 frames", "These 4 frames")

def load_event_ids():
    with open(EVENTS) as f:
        return [json.loads(l)["event_id"] for l in f]

def load_frames(event_id, n_frames):
    from PIL import Image
    fd = os.path.join(FRAMES_DIR, event_id)
    paths = [os.path.join(fd, "f%d.jpg" % j) for j in range(8)]
    if not all(os.path.exists(p) for p in paths):
        raise FileNotFoundError(fd)
    sel = range(8) if n_frames == 8 else [0, 2, 4, 6]
    return [Image.open(paths[j]).convert("RGB").resize((448, 336)) for j in sel]

_cache, _cycle = {}, {}
def frames_for(nf):
    if nf not in _cache:
        _cache[nf] = [load_frames(e, nf) for e in load_event_ids()]
        _cycle[nf] = 0
    lst = _cache[nf]
    _cycle[nf] = (_cycle[nf] + 1) % len(lst)
    return lst[_cycle[nf] - 1]

def build_batch_input(proc, n_frames_list, device):
    texts, flat_imgs = [], []
    for nf in n_frames_list:
        imgs = frames_for(nf)
        instr = SCHEMA_INSTR if nf == 8 else SCHEMA_INSTR_4
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": instr}]
        texts.append(proc.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True))
        flat_imgs.extend(imgs)
    return proc(text=texts, images=flat_imgs, padding=True,
                return_tensors="pt").to(device)

def run_point(model, proc, device, batch_size, frame_comp, n_jobs, warmup, out, tag):
    """frame_comp: list of frame counts cycled across batch slots."""
    n_batches = n_jobs // batch_size
    comp = [frame_comp[i % len(frame_comp)] for i in range(batch_size)]
    t_start = time.time()
    for b in range(n_batches + warmup):
        inputs = build_batch_input(proc, comp, device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=True, temperature=0.2, top_p=0.9)
        torch.cuda.synchronize()
        wall = time.time() - t0
        if b < warmup:
            print("[%s] warmup %d/%d %.1fs" % (tag, b + 1, warmup, wall), flush=True)
            del inputs, out_ids
            continue
        rec = {"tag": tag, "batch_size": batch_size,
               "n_4f": sum(1 for c in comp if c == 4),
               "n_8f": sum(1 for c in comp if c == 8),
               "batch_idx": b - warmup, "jobs": len(comp),
               "wall_s": round(wall, 3),
               "padded_input_tokens": int(inputs["input_ids"].shape[1]),
               "gen_tokens_total": int(out_ids.shape[1] - inputs["input_ids"].shape[1]) * len(comp),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
               "elapsed_s": round(time.time() - t_start, 1)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print("[%s] batch %d/%d wall=%.2fs vram=%.0fMB" %
              (tag, b - warmup + 1, n_batches, wall, rec["peak_vram_mb"]), flush=True)
        del inputs, out_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--plan", choices=["8f", "4f", "mixed"], required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    device = torch.device("cuda:0")

    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics_gpu%d_%s.jsonl" % (args.gpu, args.plan))
    out = open(metrics_path, "a")

    print("loading model on gpu%d ..." % args.gpu, flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).eval().to(device)
    v100_patches.apply_patches(model)
    proc = AutoProcessor.from_pretrained(QWEN_PATH)
    proc.tokenizer.padding_side = "left"
    print("model loaded in %.0fs" % (time.time() - t0), flush=True)

    if args.plan in ("8f", "4f"):
        nf = 8 if args.plan == "8f" else 4
        points = [(bs, [nf]) for bs in (1, 2, 4, 8, 16)]
    else:
        points = [(8, [4, 8]), (16, [4, 8])]
    for bs, comp in points:
        n_jobs = ((30 + bs - 1) // bs) * bs
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        run_point(model, proc, device, bs, comp, n_jobs,
                  warmup=2 if bs <= 8 else 1, out=out,
                  tag="%s_b%d" % (args.plan, bs))

    out.close()
    with open(os.path.join(args.run_dir, "status_gpu%d_%s.json" % (args.gpu, args.plan)), "w") as f:
        json.dump({"plan": args.plan, "gpu": args.gpu, "status": "done",
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
