"""T3 — M1 sketch surrogate validity (decides P2-V3). No GPU.

Premise tested: a per-camera product-quantized count sketch over tier-1
pooled CLIP features (+ time-of-day bucket) predicts whether an event is
redundant with what the memory already stores for that camera.

Method (per work1.md §T3):
1. 749 logged events + narrations (L1.summary), bge embeddings E (384-d).
2. Post-hoc redundancy label per event: max embedding cosine of its
   narration to PRIOR same-camera (= same video) narrations; secondary
   label: tag-match rate vs priors. Strictly per-camera, never cross-camera.
3. Sketch count per event: PQ'd pooled tier-1 CLIP features
   (UCF: 10-crop snippet features, 16 frames @30fps; XD: 16-frame snippets
   @24fps; pooled = mean over event window) + time-of-day bucket
   (start_sec // 300 — no wall-clock timestamps exist; documented proxy).
4. Spearman rho between sketch count and redundancy.
5. Fallback: closed-taxonomy tag alone as predictor.

PASS = rho > 0.4 -> build P2-V3; else tag fallback if stronger; if both
fail, marginal-gain mechanism is dead.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

FLEETMEM = os.path.expanduser("~/e1/fleetmem")
sys.path.insert(0, FLEETMEM)
from querypath import load_memory, load_jsonl  # noqa: E402

E1 = os.path.expanduser("~/e1")
INGEST = os.path.join(E1, "runs/20260918_0111_writepath_fullcorpus")
TAGRUN = os.path.join(E1, "runs/20260918_0802_v01_tagging")
E0DATA = "/home/san/FleetVAD/research/e0/data"
OUT = os.path.join(E1, "t3_sketch_validity")

UCF_SPS = 16.0 / 30.0   # seconds per snippet (VadCLIP granularity)
XD_SPS = 16.0 / 24.0
PQ_M = 8                # subspaces
PQ_K = 64               # centroids per subspace
TOD_BUCKET_S = 300.0    # time-of-day proxy: 5-min bucket of start_sec


def load_tags():
    tags = {}
    for p in glob.glob(os.path.join(TAGRUN, "artifacts", "tags_*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            tags[r["event_id"]] = r.get("category_tag")
    return tags


def video_snippets(ds, vid):
    """Return (n_snips, 512) float32 CLIP features for a video."""
    if ds == "ucf":
        fs = sorted(glob.glob(os.path.join(
            E0DATA, "UCFClipFeatures", "*", f"{vid}__*.npy")))
        if not fs:
            return None
        arrs = [np.load(f).astype(np.float32) for f in fs]
        n = min(a.shape[0] for a in arrs)
        F = np.mean(np.stack([a[:n] for a in arrs]), axis=0)  # 10-crop mean
        return F
    p = os.path.join(E0DATA, "XDTestClipFeatures", f"{vid}.npy")
    if not os.path.exists(p):
        p = os.path.join(E0DATA, "XDTestClipFeatures_self", f"{vid}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def pooled_feature(F, start, end, sps):
    n = F.shape[0]
    i0 = min(max(0, int(start / sps)), n - 1)
    i1 = min(max(i0 + 1, int(np.ceil(end / sps))), n)
    v = F[i0:i1].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def pq_fit_encode(X, m=PQ_M, k=PQ_K, seed=0):
    from sklearn.cluster import KMeans
    n, d = X.shape
    assert d % m == 0
    sub = d // m
    codes = np.zeros((n, m), dtype=np.int64)
    for i in range(m):
        km = KMeans(n_clusters=k, n_init=4, random_state=seed)
        codes[:, i] = km.fit_predict(X[:, i * sub:(i + 1) * sub])
    return codes


def main():
    os.makedirs(os.path.join(OUT, "artifacts"), exist_ok=True)
    recs, E = load_memory(INGEST)
    tags = load_tags()
    n = len(recs)
    print(f"events: {n}", flush=True)

    # ---- pooled CLIP features per event
    cache = {}
    X = np.zeros((n, 512), dtype=np.float32)
    n_fallback = 0
    for i, r in enumerate(recs):
        L = r["L0"]
        key = (L["dataset"], L["video_id"])
        if key not in cache:
            cache[key] = video_snippets(*key)
        F = cache[key]
        if F is None:
            n_fallback += 1
            continue
        sps = UCF_SPS if L["dataset"] == "ucf" else XD_SPS
        X[i] = pooled_feature(F, L["start_sec"], L["end_sec"], sps)
        if i % 100 == 0:
            print(f"  pooled {i}/{n}", flush=True)
    print(f"events without features: {n_fallback}", flush=True)

    # ---- order events per camera (video), priors = earlier same-video events
    by_vid = {}
    for i, r in enumerate(recs):
        L = r["L0"]
        by_vid.setdefault((L["dataset"], L["video_id"]), []).append(i)
    for v in by_vid:
        by_vid[v].sort(key=lambda i: recs[i]["L0"]["start_sec"])

    # ---- PQ codes over all pooled features
    codes = pq_fit_encode(X)
    print("PQ codes done", flush=True)

    rows = []
    for v, idxs in by_vid.items():
        for pos, i in enumerate(idxs):
            priors = idxs[:pos]
            L = recs[i]["L0"]
            tag_i = tags.get(L["event_id"])
            row = {"event_id": L["event_id"], "dataset": L["dataset"],
                   "video_id": L["video_id"], "start_sec": L["start_sec"],
                   "tag": tag_i, "n_priors": len(priors)}
            if priors:
                sims = E[priors] @ E[i]
                j_best = int(np.argmax(sims))
                row["redundancy_cos"] = float(sims[j_best])
                pt = [tags.get(recs[j]["L0"]["event_id"]) for j in priors]
                row["tag_match_rate"] = float(np.mean(
                    [1.0 if t == tag_i else 0.0 for t in pt]))
                row["tag_any_match"] = float(
                    any(t == tag_i for t in pt))
                # sketch counts (prior same-camera events only)
                row["count_strict"] = float(sum(
                    int(np.array_equal(codes[j], codes[i])) for j in priors))
                tod = int(L["start_sec"] // TOD_BUCKET_S)
                row["count_strict_tod"] = float(sum(
                    int(np.array_equal(codes[j], codes[i]) and
                        int(recs[j]["L0"]["start_sec"] // TOD_BUCKET_S) == tod)
                    for j in priors))
                # per-subspace count, averaged over subspaces
                row["count_subspace_mean"] = float(np.mean(
                    [sum(1 for j in priors if codes[j, m] == codes[i, m])
                     for m in range(PQ_M)]))
                # oracle diagnostic: unquantized pooled-feature cosine
                cs = X[priors] @ X[i]
                row["oracle_clip_cos"] = float(cs.max())
            rows.append(row)

    with open(os.path.join(OUT, "artifacts", "per_event.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    el = [r for r in rows if r["n_priors"] > 0 and "redundancy_cos" in r]
    print(f"events with >=1 prior same-camera event: {len(el)} / {n}",
          flush=True)

    from scipy.stats import spearmanr
    y_cos = np.array([r["redundancy_cos"] for r in el])
    y_tag = np.array([r["tag_match_rate"] for r in el])
    preds = ["count_strict", "count_strict_tod", "count_subspace_mean",
             "oracle_clip_cos", "tag_any_match", "tag_match_rate"]
    corr = {}
    for p in preds:
        x = np.array([r[p] for r in el])
        rho_c, pv_c = spearmanr(x, y_cos)
        rho_t, pv_t = spearmanr(x, y_tag)
        corr[p] = {"vs_cosine_label": {"rho": float(rho_c), "p": float(pv_c)},
                   "vs_tagrate_label": {"rho": float(rho_t), "p": float(pv_t)}}
        print(f"{p:22s} rho_cos={rho_c:+.3f} (p={pv_c:.2e})  "
              f"rho_tag={rho_t:+.3f} (p={pv_t:.2e})", flush=True)

    # descriptive stats
    desc = {
        "n_events": n,
        "n_cameras": len(by_vid),
        "n_eligible": len(el),
        "events_per_camera_hist": {
            str(k): sum(1 for v in by_vid.values() if len(v) == k)
            for k in range(1, max(len(v) for v in by_vid.values()) + 1)},
        "redundancy_cos_quantiles": np.quantile(
            y_cos, [0, .25, .5, .75, 1]).round(3).tolist(),
        "count_subspace_mean_quantiles": np.quantile(
            [r["count_subspace_mean"] for r in el],
            [0, .25, .5, .75, 1]).round(3).tolist(),
    }

    metrics = {"labels": {"primary": "max bge cosine of narration vs prior "
                          "same-camera narrations",
                          "secondary": "tag-match rate vs priors"},
               "pq": {"M": PQ_M, "K": PQ_K, "trained_on":
                      "pooled features of all 749 events"},
               "tod_bucket_s": TOD_BUCKET_S,
               "spearman": corr, "descriptive": desc,
               "feature_fallbacks": n_fallback}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([r["count_subspace_mean"] for r in el])
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(x + rng.uniform(-0.03, 0.03, len(x)),
                  y_cos + rng.uniform(-0.005, 0.005, len(y_cos)),
                  s=14, alpha=0.6)
    ax[0].set_xlabel("PQ sketch count (subspace-mean, per-camera priors)")
    ax[0].set_ylabel("redundancy = max cosine to prior same-camera narration")
    ax[0].set_title(f"rho = {corr['count_subspace_mean']['vs_cosine_label']['rho']:.3f} "
                    f"(n={len(el)})")
    xs = np.array([r["count_strict"] for r in el])
    ax[1].scatter(xs + rng.uniform(-0.08, 0.08, len(xs)),
                  y_cos + rng.uniform(-0.005, 0.005, len(y_cos)),
                  s=14, alpha=0.6, c="tab:orange")
    ax[1].set_xlabel("PQ sketch count (strict full-code match)")
    ax[1].set_ylabel("redundancy = max cosine to prior same-camera narration")
    ax[1].set_title(f"rho = {corr['count_strict']['vs_cosine_label']['rho']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sketch_vs_redundancy.png"), dpi=150)

    # ---- run-dir metadata
    config = {"task": "T3 M1 sketch surrogate validity",
              "memory": INGEST, "tags": TAGRUN,
              "clip_features": {"ucf": "E0 data/UCFClipFeatures (10-crop, "
                                "16f@30fps snippets)", "xd": "E0 data/"
                                "XDTestClipFeatures (16f@24fps snippets)"},
              "pq": {"M": PQ_M, "K": PQ_K},
              "tod_bucket_s": TOD_BUCKET_S,
              "camera_definition": "one video = one camera/stream; strictly "
              "per-camera, never cross-camera",
              "no_gpu": True}
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    env = {"python": sys.version.split()[0],
           "conda_env": os.environ.get("CONDA_PREFIX", "unknown"),
           "date": time.strftime("%Y-%m-%d %H:%M %Z"), "no_gpu": True}
    import scipy, sklearn
    env["scipy"] = scipy.__version__
    env["sklearn"] = sklearn.__version__
    env["numpy"] = np.__version__
    with open(os.path.join(OUT, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    code_state = {"code_dir": FLEETMEM, "files": {}}
    for fn in os.listdir(FLEETMEM):
        if fn.endswith((".py", ".md")):
            p = os.path.join(FLEETMEM, fn)
            code_state["files"][fn] = {"bytes": os.path.getsize(p),
                                       "mtime": os.path.getmtime(p)}
    code_state["files"]["eviction/t23_tombstone_matrix.py"] = {
        "bytes": os.path.getsize(os.path.join(
            FLEETMEM, "eviction/t23_tombstone_matrix.py")),
        "mtime": os.path.getmtime(os.path.join(
            FLEETMEM, "eviction/t23_tombstone_matrix.py"))}
    with open(os.path.join(OUT, "code_state.json"), "w") as f:
        json.dump(code_state, f, indent=2)
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
