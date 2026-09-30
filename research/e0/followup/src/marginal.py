"""The cache's MARGINAL value: a cache only saves work on clips that would
otherwise reach the VLM, i.e. the ones the tier-1 gate escalates.

E0 measured hit rate over ALL clips. Recomputed here over ESCALATED clips only
(PerStreamGate, tau=0.5, margin=0.2604 -- E0's own 8.6%-rate calibration), and
restricted to the cache-eligible candidate pool (earlier escalated clips, since
only they have a stored VLM verdict).
"""
import numpy as np
E0='/Users/24sf51025/Research/FleetVAD/research/e0'; SNIP=16
d=np.load(f'{E0}/results/motion_features.npz'); motion,lens=d['motion'],d['lens']
t1=np.load(f'{E0}/results/tier1_scores.npz',allow_pickle=True); ids,scores=t1['video_ids'],t1['scores']
ann={}
for line in open(f'{E0}/third_party/VadCLIP/list/Temporal_Anomaly_Annotation.txt'):
    p=line.split()
    if not p: continue
    ann[p[0].replace('.mp4','')]=[(int(a),int(b)) for a,b in zip(p[2::2],p[3::2]) if int(a)>=0 and int(b)>=0]
labels=np.zeros((len(ids),lens.max()),dtype=np.int8)
for i,vid in enumerate(ids):
    for a,b in ann.get(vid,[]):
        labels[i,max((a-1)//SNIP,0):min((b-1)//SNIP,lens[i]-1)+1]=1
cams=[[] for _ in range(32)]
for i in range(len(ids)): cams[i%32].append(i)
ev=[]
for c,cam in enumerate(cams):
    t=0.0
    for vi in cam:
        for si in range(lens[vi]): ev.append((t,c,vi,si)); t+=SNIP/30.0
ev.sort(key=lambda e:(e[0],e[1]))
ev_vi=np.array([e[2] for e in ev]); ev_si=np.array([e[3] for e in ev]); n=len(ev)
ev_lab=labels[ev_vi,ev_si]; ev_s=scores[ev_vi,ev_si]
TAU,MARGIN=0.5,0.2603611499071121
esc=np.abs(ev_s-TAU)<MARGIN
print(f"clips={n}  escalated={esc.sum()} ({esc.mean():.4f})  anomalous among escalated={ev_lab[esc].mean():.3f}")
K=np.ascontiguousarray(motion[ev_vi,ev_si].astype(np.float32)); K/=np.maximum(np.linalg.norm(K,axis=1,keepdims=True),1e-9)

# causal top-1 among EARLIER ESCALATED clips only (the real cache contents)
idx=np.flatnonzero(esc); m=len(idx)
Ke=K[idx]; lab_e=ev_lab[idx]; vi_e=ev_vi[idx]
S=Ke@Ke.T; S[np.triu_indices(m,0)]=-2.0
nn=S.argmax(1); sim=S[np.arange(m),nn]; nn=np.where(sim>-2.0,nn,-1)
ok=nn>=0; nn_lab=np.where(ok,lab_e[np.clip(nn,0,None)],-1)
same_vid=ok&(vi_e[np.clip(nn,0,None)]==vi_e)
n_anom_e=(lab_e==1).sum()
print(f"\nCache over ESCALATED clips (pool = earlier escalated clips), motion-only:")
print(f"{'theta':>6} {'hit/esc':>8} {'agree':>7} {'agree|anom':>11} {'suppressed':>11} {'sameVid%':>9}")
for th in (0.80,0.90,0.95,0.96,0.98,0.99):
    hit=ok&(sim>=th)
    if hit.sum()==0: print(f"{th:>6.2f} {'0':>8}"); continue
    a=lab_e[hit]; b=nn_lab[hit]
    aa=a==1
    print(f"{th:>6.2f} {hit.mean():>8.3f} {(a==b).mean():>7.3f} "
          f"{((b[aa]==1).mean() if aa.any() else float('nan')):>11.3f} "
          f"{(hit&(lab_e==1)&(nn_lab==0)).sum()/max(n_anom_e,1):>11.3f} {same_vid[hit].mean():>9.3f}")
print("\nFor reference, E0's headline was a hit rate over ALL clips, where 91.4% of clips")
print("are never escalated and therefore never cost a VLM call in the first place.")
