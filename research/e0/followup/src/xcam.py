"""Re-analysis of E0 Exp3 on ShanghaiTech using the cached features.

E0 recorded only the top-1 causal neighbour, so a same-video match (sim ~1.0 for
the adjacent snippet) MASKS any cross-camera match above theta. This script
re-runs the causal lookup with three separate candidate pools:

  ALL        : any earlier clip                    (= E0's measurement)
  XVID       : earlier clips from a different video (what a per-stream cache misses)
  XCAM       : earlier clips from a different CAMERA (the shared-cache premise)

and reports, per pool, hit rate AND that pool's own label agreement.
"""
import numpy as np, json, sys

d = np.load('/Users/24sf51025/Research/FleetVAD/research/e0/results/sht_features.npz', allow_pickle=True)
scene, motion, labels, meta = list(d['scene']), list(d['motion']), list(d['labels']), d['meta'].tolist()
SNIP, FPS = 16, 25.0
spf = SNIP/FPS

n_streams = 32
cams = [[] for _ in range(n_streams)]
for i in range(len(meta)): cams[i % n_streams].append(i)
events = []
for c, cam in enumerate(cams):
    t = 0.0
    for vi in cam:
        for si in range(meta[vi]['n_snips']):
            events.append((t, c, vi, si)); t += spf
events.sort(key=lambda e: (e[0], e[1]))
ev_vi = np.array([e[2] for e in events]); ev_si = np.array([e[3] for e in events])
ev_lab = np.array([labels[vi][si] for vi, si in zip(ev_vi, ev_si)])
cam_of = np.array([m['cam'] for m in meta])
ev_cam = cam_of[ev_vi]
n = len(events)
print(f"clips={n} anom={ev_lab.mean():.3f} cameras={len(set(ev_cam))}")

offs = np.cumsum([0]+[m['n_snips'] for m in meta])
flat = np.array([offs[vi]+si for vi, si in zip(ev_vi, ev_si)])
SC = np.concatenate(scene)[flat]; MO = np.concatenate(motion)[flat]

thetas = np.round(np.arange(0.80, 0.991, 0.01), 2)

def topk_masked(keys, mask_same):
    """causal top-1 within allowed pool. mask_same[i,j] True => j excluded for i."""
    S = keys @ keys.T
    iu = np.triu_indices(n, 0)
    S[iu[0], iu[1]] = -2.0          # only strictly earlier j < i
    S[mask_same] = -2.0
    j = S.argmax(axis=1); sim = S[np.arange(n), j]
    j[sim <= -2.0] = -1
    return sim, j

same_vid = ev_vi[:, None] == ev_vi[None, :]
same_cam = ev_cam[:, None] == ev_cam[None, :]

rows = []
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    K = np.concatenate([alpha*SC, (1-alpha)*MO], axis=1).astype(np.float32)
    K /= np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-9)
    for pool, mask in (('ALL', np.zeros((n, n), bool)), ('XVID', same_vid), ('XCAM', same_cam)):
        sim, j = topk_masked(K, mask)
        ok = j >= 0
        nn_lab = np.where(ok, ev_lab[np.clip(j, 0, None)], -1)
        for th in thetas:
            hit = ok & (sim >= th)
            h = hit.sum()
            if h == 0: continue
            agree = float((ev_lab[hit] == nn_lab[hit]).mean())
            dang = float((hit & (nn_lab == 0) & (ev_lab == 1)).sum()/n)
            rows.append(dict(alpha=alpha, pool=pool, theta=float(th),
                             hit_rate=float(h/n), agreement=agree, n_hits=int(h),
                             dangerous=dang))
json.dump(rows, open('/Users/24sf51025/Research/FleetVAD/research/e0/followup/results/xcam_sht.json','w'), indent=1)

print("\nBest hit rate at agreement >= 0.95, per (alpha, pool):")
print(f"{'alpha':>5} {'pool':>5} {'theta':>6} {'hit':>7} {'agree':>7} {'nhits':>6}")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for pool in ('ALL','XVID','XCAM'):
        cand = [r for r in rows if r['alpha']==alpha and r['pool']==pool and r['agreement']>=0.95]
        if not cand:
            best = max([r for r in rows if r['alpha']==alpha and r['pool']==pool], key=lambda r: r['agreement'])
            print(f"{alpha:>5} {pool:>5} {'--':>6} {'none':>7} {best['agreement']:>7.3f} {best['n_hits']:>6}  (max agree, theta={best['theta']})")
        else:
            b = max(cand, key=lambda r: r['hit_rate'])
            print(f"{alpha:>5} {pool:>5} {b['theta']:>6} {b['hit_rate']:>7.3f} {b['agreement']:>7.3f} {b['n_hits']:>6}")

print("\nXCAM detail, alpha=0.0 / 0.5 / 1.0 (agreement vs theta):")
for alpha in [0.0, 0.5, 1.0]:
    print(f" alpha={alpha}")
    for r in rows:
        if r['alpha']==alpha and r['pool']=='XCAM' and abs((r['theta']*100)%5)<1e-6:
            print(f"   th={r['theta']:.2f} hit={r['hit_rate']:.3f} agree={r['agreement']:.3f} n={r['n_hits']}")
