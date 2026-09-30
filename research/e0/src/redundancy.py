"""E0 Experiment 3 -- Safe redundancy.

Question: what fraction of clips can SAFELY reuse an earlier VLM verdict?

Clip unit: one 16-frame snippet (aligned with the VadCLIP CLIP features and the
frame-level GT). e_scene = the snippet's CLIP feature (VadCLIP precomputed,
ViT-B/16 512-d) -- a documented approximation of "CLIP embedding of the clip's
middle frame". e_motion = 6x8 mean-pooled grid of mean absolute frame
differences inside the snippet, computed at 48x64 (HxW) greyscale.

Causality (E0 6.2): clips are processed in wall-clock order across N=32 virtual
cameras (same round-robin construction as Experiment 2, single pass -- every
test video appears exactly once). For each clip we search ONLY clips already
seen. The top-1 neighbour and its similarity are recorded once; every theta in
the sweep is then evaluated offline from (top_sim, neighbour) pairs, which is
exact because the neighbour set is theta-independent.

Types: T1 same source video; T2 different video, same k-means scene cluster
(clustering is video-level, on mean e_scene; k in {20,50}); T3 different
cluster. The k-means proxy is documented in REPORT section 4.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


# ---------------------------------------------------------------- signatures

def build_scene_features(video_ids):
    """(n_videos, max_snips, 512) padded array + lens, L2-normalised rows."""
    rows = common.load_test_list()
    feats, lens = [], []
    for r in rows:
        f = np.load(r["feature_path"]).astype(np.float32)
        feats.append(f)
        lens.append(len(f))
    mx = max(lens)
    out = np.zeros((len(feats), mx, feats[0].shape[1]), dtype=np.float32)
    for i, f in enumerate(feats):
        out[i, :len(f)] = f
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.maximum(n, 1e-9)
    return out, lens


def build_motion_features(video_ids, lens, resize_wh=(64, 48), grid=(6, 8)):
    """(n_videos, max_snips, 48) motion grid per snippet. Cached to results/."""
    cache = os.path.join(common.RESULTS_DIR, "motion_features.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["motion"], d["lens"]
    import cv2
    from decord import VideoReader, cpu

    rows = common.load_test_list()
    mx = max(lens)
    out = np.zeros((len(video_ids), mx, grid[0] * grid[1]), dtype=np.float32)
    gh, gw = grid
    for i, r in enumerate(rows):
        vf = common.video_file_for(r["video_id"], r["category"])
        vr = VideoReader(vf, ctx=cpu(0), num_threads=2)
        n_snips = lens[i]
        prev = None
        diffs_acc = np.zeros((resize_wh[1], resize_wh[0]), dtype=np.float64)
        cnt = 0
        snip = 0
        for fi in range(len(vr)):
            if snip >= n_snips:
                break
            fr = vr[fi].asnumpy()
            g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, resize_wh, interpolation=cv2.INTER_AREA)
            if prev is not None:
                diffs_acc += np.abs(g.astype(np.float64) - prev.astype(np.float64))
                cnt += 1
            prev = g
            if (fi + 1) % common.SNIPPET_FRAMES == 0:
                d = diffs_acc / max(cnt, 1)
                cell = d.reshape(gh, resize_wh[1] // gh, gw, resize_wh[0] // gw)
                out[i, snip] = cell.mean(axis=(1, 3)).astype(np.float32).ravel()
                diffs_acc[:] = 0
                cnt = 0
                snip += 1
        if i % 50 == 0:
            print(f"motion: {i}/{len(rows)} videos")
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.maximum(n, 1e-9)
    np.savez(cache, motion=out, lens=np.array(lens))
    return out, lens


def snippet_labels(video_ids, lens):
    """Binary label per snippet from Temporal_Anomaly_Annotation.txt."""
    ann = common.load_temporal_annotations()
    labels = np.zeros((len(video_ids), max(lens)), dtype=np.int8)
    for i, vid in enumerate(video_ids):
        for a, b in ann.get(vid, []):
            s0 = max((a - 1) // 16, 0)
            s1 = min((b - 1) // 16, lens[i] - 1)
            labels[i, s0:s1 + 1] = 1
    return labels


def video_clusters(e_scene, lens, k, seed=0):
    """k-means over video-mean scene features -> cluster id per video."""
    from sklearn.cluster import KMeans
    means = np.stack([e_scene[i, :lens[i]].mean(axis=0) for i in range(len(lens))])
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-9)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(means)
    return km.labels_


# ------------------------------------------------------------- causal lookup

def build_timeline(video_ids, lens, n_streams):
    """Single pass, round-robin cameras; returns clip list in wall-clock order:
    each entry (video_idx, snippet_idx, t_seconds, camera)."""
    cameras = [[] for _ in range(n_streams)]
    for i in range(len(video_ids)):
        cameras[i % n_streams].append(i)
    spf = common.SNIPPET_FRAMES / common.UCF_FPS
    events = []
    for c, cam in enumerate(cameras):
        t = 0.0
        for vi in cam:
            for si in range(lens[vi]):
                events.append((t, c, vi, si))
                t += spf
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def causal_nn(keys, events, dim):
    """For every clip in time order, top-1 neighbour among strictly-earlier
    clips. Returns arrays (top_sim, nn_pos)."""
    import faiss
    index = faiss.IndexFlatIP(dim)
    n = len(events)
    top_sim = np.full(n, -1.0, dtype=np.float32)
    nn_pos = np.full(n, -1, dtype=np.int64)
    for pos in range(n):
        k = keys[pos:pos + 1]
        if index.ntotal > 0:
            sim, j = index.search(k, 1)
            top_sim[pos] = sim[0, 0]
            nn_pos[pos] = j[0, 0]
        index.add(k)
        if pos % 20000 == 0:
            print(f"causal nn: {pos}/{n}")
    return top_sim, nn_pos


def acausal_nn(keys):
    """Upper bound: NN over the whole dataset, excluding self. Reported
    separately and clearly labelled (E0 6.2)."""
    import faiss
    index = faiss.IndexFlatIP(keys.shape[1])
    index.add(keys)
    sim, idx = index.search(keys, 2)
    # first result is self (sim ~1.0); take the second
    return sim[:, 1], idx[:, 1]


# ---------------------------------------------------------------- evaluation

def evaluate_theta(top_sim, nn_pos, labels_ev, types_ev, thetas):
    """labels_ev/types_ev: per-event arrays. Returns per-theta metrics."""
    out = []
    lab = labels_ev
    valid = nn_pos >= 0
    nn_lab = np.where(valid, lab[np.clip(nn_pos, 0, None)], -1)
    for th in thetas:
        hit = valid & (top_sim >= th)
        n_hit = hit.sum()
        if n_hit == 0:
            out.append({"theta": float(th), "hit_rate": 0.0, "label_agreement": None})
            continue
        agree = (lab[hit] == nn_lab[hit]).mean()
        by_type = {}
        for t in ("T1", "T2", "T3"):
            m = hit & (types_ev == t)
            by_type[t] = float(m.sum() / len(hit))
        dangerous = (hit & (nn_lab == 0) & (lab == 1)).sum() / len(hit)
        out.append({"theta": float(th), "hit_rate": float(n_hit / len(hit)),
                    "label_agreement": float(agree), "by_type": by_type,
                    "dangerous_hit_rate": float(dangerous)})
    return out


def operating_point(curve, min_agreement=0.95):
    ok = [c for c in curve if c["label_agreement"] is not None
          and c["label_agreement"] >= min_agreement]
    if not ok:
        return None
    return max(ok, key=lambda c: c["hit_rate"])


def run_experiment(cfg, out_path):
    common.set_seed(cfg.get("seed", 0))
    rows = common.load_test_list()
    video_ids = [r["video_id"] for r in rows]

    e_scene, lens = build_scene_features(video_ids)
    e_motion, _ = build_motion_features(video_ids, lens)
    labels = snippet_labels(video_ids, lens)

    n_streams = cfg.get("n_streams", 32)
    events = build_timeline(video_ids, lens, n_streams)
    ev_vi = np.array([e[2] for e in events])
    ev_si = np.array([e[3] for e in events])
    ev_t = np.array([e[0] for e in events])
    labels_ev = labels[ev_vi, ev_si]
    print(f"total clips: {len(events)}; anomalous fraction: {labels_ev.mean():.4f}")

    k = cfg.get("scene_clusters", 20)
    clus = video_clusters(e_scene, lens, k, cfg.get("seed", 0))
    ev_clus = clus[ev_vi]

    thetas = np.round(np.arange(0.80, 0.991, 0.01), 2)
    alphas = cfg.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0])
    results_by_alpha = {}
    for alpha in alphas:
        keys = np.concatenate([alpha * e_scene[ev_vi, ev_si],
                               (1 - alpha) * e_motion[ev_vi, ev_si]], axis=1)
        keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-9)
        keys = keys.astype(np.float32)

        top_sim, nn_pos = causal_nn(keys, events, keys.shape[1])
        nn_vi = np.where(nn_pos >= 0, ev_vi[np.clip(nn_pos, 0, None)], -1)
        nn_clus = np.where(nn_pos >= 0, ev_clus[np.clip(nn_pos, 0, None)], -1)
        types_ev = np.where(nn_vi == ev_vi, "T1",
                            np.where(nn_clus == ev_clus, "T2", "T3"))

        curve = evaluate_theta(top_sim, nn_pos, labels_ev, types_ev, thetas)
        op = operating_point(curve)
        results_by_alpha[str(alpha)] = {"curve": curve, "operating_point": op}
        if op:
            print(f"alpha={alpha}: theta*={op['theta']} safe_hit_rate={op['hit_rate']:.3f}")

    # pick the alpha with the best safe hit rate
    best_alpha, best = None, None
    for a, r in results_by_alpha.items():
        if r["operating_point"] and (best is None or
                                     r["operating_point"]["hit_rate"] > best["hit_rate"]):
            best_alpha, best = a, r["operating_point"]

    # recompute details at best alpha for decay + acausal bound
    alpha = float(best_alpha)
    keys = np.concatenate([alpha * e_scene[ev_vi, ev_si],
                           (1 - alpha) * e_motion[ev_vi, ev_si]], axis=1)
    keys = (keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-9)
            ).astype(np.float32)
    top_sim, nn_pos = causal_nn(keys, events, keys.shape[1])
    nn_vi = np.where(nn_pos >= 0, ev_vi[np.clip(nn_pos, 0, None)], -1)
    nn_si = np.where(nn_pos >= 0, ev_si[np.clip(nn_pos, 0, None)], -1)
    th = best["theta"]
    hit = (nn_pos >= 0) & (top_sim >= th)

    # decay: T1 hits by video-internal time gap
    spf = common.SNIPPET_FRAMES / common.UCF_FPS
    t1 = hit & (nn_vi == ev_vi)
    gaps_min = (ev_si[t1] - nn_si[t1]) * spf / 60.0
    decay = {
        "lt_1min": float((gaps_min < 1).sum() / max(len(gaps_min), 1)),
        "1_5min": float(((gaps_min >= 1) & (gaps_min < 5)).sum() / max(len(gaps_min), 1)),
        "5_30min": float(((gaps_min >= 5) & (gaps_min < 30)).sum() / max(len(gaps_min), 1)),
        "gt_30min": float((gaps_min >= 30).sum() / max(len(gaps_min), 1)),
    }

    sim_ac, idx_ac = acausal_nn(keys)
    acausal_hit = (sim_ac >= th).mean()

    payload = {
        "alpha": alpha, "causal": True,
        "n_streams": n_streams, "scene_clusters_k": k,
        "n_clips": len(events),
        "alphas": results_by_alpha,
        "operating_point": best,
        "decay": decay,
        "acausal_upper_bound": float(acausal_hit),
        "note": "acausal_upper_bound is an optimistic whole-dataset lookup; "
                "all headline numbers are causal",
    }
    common.save_result(out_path, cfg, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_experiment(common.load_config(args.config), args.out)


if __name__ == "__main__":
    main()
