"""E0 Experiment 3 follow-up on ShanghaiTech (required by Gate C: T2 < 5% on
UCF-Crime, so cross-camera reuse must be re-measured where cameras genuinely
share a site).

ShanghaiTech test split: 107 videos, 13 fixed campus cameras. Here:
  T1  = neighbour from the same source video
  T1b = neighbour from a different video of the SAME camera
  T2  = neighbour from a DIFFERENT camera, same campus (exact, per E0 6.3)
  (no T3 -- one site)

e_scene: CLIP ViT-B/16 features extracted per frame and mean-pooled per
16-frame snippet, using the CLIP weights rebuilt from the VadCLIP checkpoint
(openaipublic is unreachable; no official precomputed SHT CLIP features exist
in VadCLIP format -- E0 3.3 permits extraction in that case).
e_motion: same 6x8 grid as the UCF run.
Causal constraint identical to Experiment 3 (wall-clock order, past only).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import redundancy as red  # reuse causal_nn, evaluate_theta, operating_point, acausal_nn

SHT_DIR = os.path.join(common.DATA_DIR, "shanghaitech", "shanghaitech", "testing")
SHT_FPS = 25.0
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_clip_model(device):
    import torch
    sys.path.insert(0, common.VADCLIP_SRC)
    from clip.clip import build_model
    state = torch.load(common.VADCLIP_CKPT, map_location="cpu")
    clip_state = {k[len("clipmodel."):]: v for k, v in state.items()
                  if k.startswith("clipmodel.")}
    model = build_model(clip_state).to(device).eval()
    return model


def preprocess(frame_rgb):
    import cv2
    img = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    return img.transpose(2, 0, 1)  # CHW


def list_videos():
    dirs = sorted(glob.glob(os.path.join(SHT_DIR, "frames", "*")))
    vids = []
    for d in dirs:
        name = os.path.basename(d)          # e.g. 01_0014
        cam = name.split("_")[0]
        n_frames = len(glob.glob(os.path.join(d, "*.jpg")))
        mask = np.load(os.path.join(SHT_DIR, "test_frame_mask", name + ".npy"))
        vids.append({"name": name, "cam": cam, "dir": d,
                     "n_frames": min(n_frames, len(mask)), "mask": mask})
    return vids


def extract_features(vids, device, batch=64):
    """Per video: CLIP feature + motion grid + label per 16-frame snippet."""
    import cv2
    import torch
    model = load_clip_model(device)
    F = common.SNIPPET_FRAMES
    all_scene, all_motion, all_labels, meta = [], [], [], []
    for vi, v in enumerate(vids):
        files = sorted(glob.glob(os.path.join(v["dir"], "*.jpg")))[:v["n_frames"]]
        frames_f, frames_g = [], []
        for f in files:
            bgr = cv2.imread(f)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_f.append(preprocess(rgb))
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            frames_g.append(cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA))
        n_snips = len(files) // F
        if n_snips == 0:
            continue
        # CLIP per frame, batched
        feats = []
        with torch.no_grad():
            for i in range(0, len(frames_f), batch):
                x = torch.tensor(np.stack(frames_f[i:i + batch]))
                x = x.to(device)
                if next(model.parameters()).dtype == torch.float16:
                    x = x.half()
                e = model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)
                feats.append(e.float().cpu().numpy())
        feats = np.concatenate(feats)[:n_snips * F]
        scene = feats.reshape(n_snips, F, -1).mean(axis=1)
        scene /= np.maximum(np.linalg.norm(scene, axis=1, keepdims=True), 1e-9)
        # motion
        motion = np.zeros((n_snips, 48), dtype=np.float32)
        for s in range(n_snips):
            acc = np.zeros((48, 64), dtype=np.float64)
            for i in range(s * F + 1, (s + 1) * F):
                acc += np.abs(frames_g[i].astype(np.float64)
                              - frames_g[i - 1].astype(np.float64))
            d = acc / (F - 1)
            cell = d.reshape(6, 8, 8, 8)
            motion[s] = cell.mean(axis=(1, 3)).ravel()
        motion /= np.maximum(np.linalg.norm(motion, axis=1, keepdims=True), 1e-9)
        # labels
        lab = np.array([v["mask"][s * F:(s + 1) * F].any() for s in range(n_snips)],
                       dtype=np.int8)
        all_scene.append(scene)
        all_motion.append(motion)
        all_labels.append(lab)
        meta.append({"video": v["name"], "cam": v["cam"], "n_snips": n_snips})
        print(f"{v['name']}: {n_snips} snippets, anom={lab.mean():.3f}", flush=True)
    return all_scene, all_motion, all_labels, meta


def run_experiment(cfg, out_path):
    import torch
    common.set_seed(cfg.get("seed", 0))
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    vids = list_videos()
    print(f"{len(vids)} test videos across "
          f"{len(set(v['cam'] for v in vids))} cameras")

    cache = os.path.join(common.RESULTS_DIR, "sht_features.npz")
    if os.path.exists(cache) and not cfg.get("force_features"):
        d = np.load(cache, allow_pickle=True)
        all_scene, all_motion, all_labels, meta = (
            list(d["scene"]), list(d["motion"]), list(d["labels"]),
            d["meta"].tolist())
    else:
        all_scene, all_motion, all_labels, meta = extract_features(vids, device)
        np.savez(cache, scene=np.array(all_scene, dtype=object),
                 motion=np.array(all_motion, dtype=object),
                 labels=np.array(all_labels, dtype=object),
                 meta=np.array(meta))

    # timeline: round-robin the 107 videos over n_streams virtual cameras,
    # single pass (mirrors the UCF construction)
    n_streams = cfg.get("n_streams", 32)
    spf = common.SNIPPET_FRAMES / SHT_FPS
    cams = [[] for _ in range(n_streams)]
    for i in range(len(meta)):
        cams[i % n_streams].append(i)
    events = []
    for c, cam in enumerate(cams):
        t = 0.0
        for vi in cam:
            for si in range(meta[vi]["n_snips"]):
                events.append((t, c, vi, si))
                t += spf
    events.sort(key=lambda e: (e[0], e[1]))
    ev_vi = np.array([e[2] for e in events])
    ev_si = np.array([e[3] for e in events])
    labels_cat = np.concatenate(all_labels)
    offs = np.cumsum([0] + [m["n_snips"] for m in meta])
    ev_lab = np.array([all_labels[vi][si] for vi, si in zip(ev_vi, ev_si)])
    print(f"total clips: {len(events)}, anomalous: {ev_lab.mean():.4f}")

    vid_of = np.array([m["video"] for m in meta])
    cam_of = np.array([m["cam"] for m in meta])

    thetas = np.round(np.arange(0.80, 0.991, 0.01), 2)
    alphas = cfg.get("alphas", [0.5, 0.75, 1.0])
    results_by_alpha = {}
    for alpha in alphas:
        keys = np.concatenate([alpha * np.concatenate(all_scene)[
                                   np.array([offs[vi] + si for vi, si in zip(ev_vi, ev_si)])],
                               (1 - alpha) * np.concatenate(all_motion)[
                                   np.array([offs[vi] + si for vi, si in zip(ev_vi, ev_si)])]],
                              axis=1).astype(np.float32)
        keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-9)
        top_sim, nn_pos = red.causal_nn(keys, events, keys.shape[1])
        nn_vi = np.where(nn_pos >= 0, ev_vi[np.clip(nn_pos, 0, None)], -1)
        same_vid = nn_vi == ev_vi
        same_cam = (nn_vi >= 0) & (cam_of[np.clip(nn_vi, 0, None)] == cam_of[ev_vi])
        types_ev = np.where(same_vid, "T1", np.where(same_cam, "T1b", "T2"))
        curve = red.evaluate_theta(top_sim, nn_pos, ev_lab, types_ev, thetas)
        # evaluate_theta reports T1/T2/T3 keys; remap T1b into by_type
        for c in curve:
            if "by_type" in c:
                th = c["theta"]
                hit = (nn_pos >= 0) & (top_sim >= th)
                c["by_type"] = {t: float(((types_ev == t) & hit).sum() / len(hit))
                                for t in ("T1", "T1b", "T2")}
        op = red.operating_point(curve)
        valid = [c for c in curve if c["label_agreement"] is not None]
        max_agree = max(valid, key=lambda c: c["label_agreement"]) if valid else None
        results_by_alpha[str(alpha)] = {"curve": curve, "operating_point": op,
                                        "max_agreement_point": max_agree}
        if op:
            print(f"alpha={alpha}: theta*={op['theta']} "
                  f"safe={op['hit_rate']:.3f} by={op['by_type']}", flush=True)
        else:
            print(f"alpha={alpha}: NO theta reaches 95% agreement; "
                  f"max agreement={max_agree['label_agreement']:.4f} "
                  f"at theta={max_agree['theta']} (hit_rate={max_agree['hit_rate']:.4f})",
                  flush=True)

    best_alpha, best = None, None
    for a, r in results_by_alpha.items():
        o = r["operating_point"]
        if o and (best is None or o["hit_rate"] > best["hit_rate"]):
            best_alpha, best = a, o

    payload = {
        "dataset": "ShanghaiTech test (107 videos, 12 cameras)",
        "alpha": float(best_alpha) if best_alpha else None, "causal": True,
        "n_streams": n_streams, "n_clips": len(events),
        "types": {"T1": "same video", "T1b": "same camera different video",
                  "T2": "different camera, same campus (exact)"},
        "alphas": results_by_alpha,
        "operating_point": best,
        "no_safe_operating_point": best is None,
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
