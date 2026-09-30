"""E0 Experiment 2 -- Demand and contention (the GO/NO-GO experiment).

Step 1: tier-1 VadCLIP scores per snippet from precomputed CLIP features
        (sanity: must reproduce published AUC 88.02 on UCF-Crime).
Step 2: PerStreamGate policy + margin calibration.
Step 3: N-stream wall-clock simulation (pure arithmetic over the score file).
Step 4: crossover N where demand > CAPACITY.
Step 5: load factor rho.

Note on CLIP weights: openaipublic (Azure) is unreachable from our server, so
clip.load("ViT-B/16") cannot download the base model. model_ucf.pth contains
the full state_dict including clipmodel.* weights, so we monkeypatch clip.load
to rebuild the CLIP model from those keys (see _patch_clip_load).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

sys.path.insert(0, common.VADCLIP_SRC)


# ------------------------------------------------------------- tier-1 scores

def _patch_clip_load(ckpt_path):
    """Replace clip.load with a builder that sources weights from the VadCLIP
    checkpoint itself (clipmodel.* keys)."""
    import torch
    import clip.clip as clip_mod  # VadCLIP's bundled CLIP

    state = torch.load(ckpt_path, map_location="cpu")
    clip_state = {k[len("clipmodel."):]: v for k, v in state.items()
                  if k.startswith("clipmodel.")}
    if not clip_state:
        raise RuntimeError("checkpoint has no clipmodel.* keys; cannot rebuild CLIP")

    def fake_load(name, device=None, jit=False, download_root=None):
        model = clip_mod.build_model(clip_state)
        return model, None

    clip_mod.load = fake_load
    return state


def score_test_set(cfg):
    """Run VadCLIP over every test video's features. Returns per-video
    per-snippet score arrays plus the two headline AUCs."""
    import torch
    from torch.utils.data import DataLoader
    from model import CLIPVAD
    from utils.dataset import UCFDataset
    from utils.tools import get_batch_mask, get_prompt_text
    from sklearn.metrics import roc_auc_score

    state = _patch_clip_load(common.VADCLIP_CKPT)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    maxlen = 256
    testdataset = UCFDataset(maxlen, os.path.join(common.VADCLIP_LIST, "ucf_CLIP_rgbtest.csv"),
                             True, common.UCF_LABEL_MAP)
    # point the csv's absolute paths at our local feature dir
    testdataset.df["path"] = [r["feature_path"] for r in common.load_test_list()]
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(common.UCF_LABEL_MAP)

    model = CLIPVAD(14, 512, maxlen, 512, 1, 2, 8, 10, 10, device)
    model.load_state_dict(state)
    model.to(device).eval()

    rows = common.load_test_list()
    scores1, scores2 = {}, {}
    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = torch.zeros(int(length / maxlen) + 1)
            rem = length
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    rem -= maxlen
                elif rem > maxlen:
                    lengths[j] = maxlen
                    rem -= maxlen
                else:
                    lengths[j] = rem
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            vid = rows[i]["video_id"]
            scores1[vid] = prob1.cpu().numpy()
            scores2[vid] = prob2.cpu().numpy()

    gt = common.load_frame_gt()
    concat1 = np.concatenate([np.repeat(scores1[r["video_id"]], 16) for r in rows])
    concat2 = np.concatenate([np.repeat(scores2[r["video_id"]], 16) for r in rows])
    n = min(len(gt), len(concat1))
    auc1 = roc_auc_score(gt[:n], concat1[:n])
    auc2 = roc_auc_score(gt[:n], concat2[:n])
    return scores1, scores2, auc1, auc2


def load_or_compute_scores(cfg, force=False):
    out = os.path.join(common.RESULTS_DIR, "tier1_scores.npz")
    if os.path.exists(out) and not force:
        d = np.load(out, allow_pickle=True)
        return d["video_ids"].tolist(), d["scores"], float(d["auc_used"])
    scores1, scores2, auc1, auc2 = score_test_set(cfg)
    print(f"VadCLIP sanity: AUC1={auc1:.4f} AUC2={auc2:.4f} (published 88.02)")
    # use whichever branch reproduces the published number
    used, auc_used = (scores1, auc1) if abs(auc1 - 0.8802) <= abs(auc2 - 0.8802) else (scores2, auc2)
    rows = common.load_test_list()
    video_ids = [r["video_id"] for r in rows]
    lens = [len(used[v]) for v in video_ids]
    scores = np.zeros((len(video_ids), max(lens)), dtype=np.float32)
    for i, v in enumerate(video_ids):
        scores[i, :lens[i]] = used[v]
    np.savez(out, video_ids=np.array(video_ids), scores=scores,
             lens=np.array(lens), auc1=auc1, auc2=auc2, auc_used=auc_used)
    return video_ids, scores, float(auc_used)


# ------------------------------------------------------------------- the gate

def per_stream_gate(s, tau=0.5, margin=0.15):
    """Escalate when the score is ambiguous, i.e. near the decision boundary."""
    return np.abs(s - tau) < margin


def calibrate_margin_at_rate(all_scores, tau, target_rate):
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = per_stream_gate(all_scores, tau, mid).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate_margin_best(all_scores_val, gt_val, tau):
    """Margin that best targets actual tier-1 errors on the validation split:
    maximise F1 of (escalate <-> tier-1 misclassifies). Interpretation of
    E0 5.2(b); documented in REPORT section 4."""
    err = (all_scores_val > tau).astype(int) != gt_val
    best, best_f1 = 0.0, -1.0
    for margin in np.arange(0.01, 0.5, 0.01):
        esc = per_stream_gate(all_scores_val, tau, margin)
        tp = np.sum(esc & err)
        if np.sum(esc) == 0 or np.sum(err) == 0:
            continue
        prec = tp / np.sum(esc)
        rec = tp / np.sum(err)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best = f1, margin
    return best, best_f1


# --------------------------------------------------------------- simulation

def simulate(video_ids, scores, lens, n_streams, clip_period_s, tau, margin,
             fps=common.UCF_FPS, min_horizon_s=900.0):
    """Round-robin assign videos to n_streams cameras; each camera loops its
    playlist; a clip every clip_period_s; demand(t) = escalations per 1-s bin."""
    order = np.argsort(video_ids)  # deterministic
    # keep csv order (video_ids already in csv order); round-robin assignment:
    cameras = [[] for _ in range(n_streams)]
    for i in range(len(video_ids)):
        cameras[i % n_streams].append(i)

    spf = common.SNIPPET_FRAMES / fps  # seconds per snippet
    # per-camera concatenated snippet timeline
    cam_snippets = []   # list of list of (video_idx, snippet_idx)
    cam_durations = []
    for cam in cameras:
        tl = []
        for vi in cam:
            for si in range(lens[vi]):
                tl.append((vi, si))
        cam_snippets.append(tl)
        cam_durations.append(len(tl) * spf)

    horizon = max(max(cam_durations), min_horizon_s)
    n_bins = int(horizon) + 1
    demand_bins = np.zeros(n_bins)
    for tl, dur in zip(cam_snippets, cam_durations):
        if not tl:
            continue
        t = 0.0
        while t < horizon:
            pos = t % dur
            si_global = int(pos / spf)
            vi, si = tl[min(si_global, len(tl) - 1)]
            s = scores[vi, si]
            if abs(s - tau) < margin:
                demand_bins[int(t)] += 1
            t += clip_period_s
    rate = demand_bins  # escalations per 1-second bin == clips/s
    return {
        "mean_demand": float(rate.mean()),
        "p95_demand": float(np.percentile(rate, 95)),
        "p99_demand": float(np.percentile(rate, 99)),
        "peak_demand": float(rate.max()),
        "horizon_s": horizon,
    }


def run_experiment(cfg, out_path):
    common.set_seed(cfg.get("seed", 0))
    video_ids, scores, auc_used = load_or_compute_scores(cfg)
    lens = np.load(os.path.join(common.RESULTS_DIR, "tier1_scores.npz"))["lens"]
    tau = cfg["tau"]

    # ---- capacity from Experiment 1 ----
    with open(cfg["capacity_json"]) as f:
        cap_doc = json.load(f)
    capacity = cap_doc["models"]["HolmesVAU-2B"]["pool_throughput_clips_per_s_4gpu"]
    print(f"CAPACITY = {capacity:.2f} clips/s")

    # ---- gate calibration ----
    flat = np.concatenate([scores[i, :lens[i]] for i in range(len(video_ids))])
    margin_rate = calibrate_margin_at_rate(flat, tau, cfg["target_escalation_rate"])

    # validation split: even/odd videos
    gt_ann = common.load_temporal_annotations()
    val_idx = list(range(0, len(video_ids), 2))
    val_scores, val_gt = [], []
    for i in val_idx:
        segs = gt_ann.get(video_ids[i], [])
        for si in range(lens[i]):
            f0, f1 = si * 16 + 1, si * 16 + 16
            lab = any(not (f1 < a or f0 > b) for a, b in segs)
            val_scores.append(scores[i, si])
            val_gt.append(int(lab))
    margin_auc, f1 = calibrate_margin_best(np.array(val_scores), np.array(val_gt), tau)
    esc_rate = float(per_stream_gate(flat, tau, margin_rate).mean())

    gate_info = {"tau": tau, "margin_at_8.6pct": float(margin_rate),
                 "escalation_rate": esc_rate,
                 "margin_at_best_auc": float(margin_auc),
                 "margin_best_f1": float(f1),
                 "tier1_auc": auc_used}

    # ---- N sweep x clip-period sweep ----
    curve = []
    for period in cfg["clip_periods"]:
        for n in cfg["n_streams"]:
            r = simulate(video_ids, scores, lens, n, period, tau, margin_rate)
            r.update({"N": n, "clip_period_s": period,
                      "rho_mean": r["mean_demand"] / capacity,
                      "rho_p95": r["p95_demand"] / capacity})
            curve.append(r)
            print(f"P={period}s N={n}: mean={r['mean_demand']:.2f} "
                  f"p95={r['p95_demand']:.2f} rho_p95={r['rho_p95']:.2f}")

    # ---- crossover at clip_period = 1.0s ----
    def crossover(key):
        for r in sorted([c for c in curve if c["clip_period_s"] == 1.0],
                        key=lambda c: c["N"]):
            if r[key] > capacity:
                return r["N"]
        return None

    n_cross_mean = crossover("mean_demand")
    n_cross_p95 = crossover("p95_demand")
    p10 = [c for c in curve if c["clip_period_s"] == 1.0]
    burstiness = {c["N"]: c["p95_demand"] / max(c["mean_demand"], 1e-9) for c in p10}

    payload = {
        "capacity_clips_per_s": capacity,
        "gate": gate_info,
        "curve": curve,
        "N_crossover_mean": n_cross_mean,
        "N_crossover_p95": n_cross_p95,
        "burstiness": burstiness,
    }
    common.save_result(out_path, cfg, payload)


def main():
    import json  # noqa: used in run_experiment
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force-scores", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    if args.force_scores:
        load_or_compute_scores(cfg, force=True)
    run_experiment(cfg, args.out)


if __name__ == "__main__":
    import json
    main()
