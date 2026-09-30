"""Does the master plan's cache safety law rescue temporal verdict reuse?

Safety law (MASTER_PLAN 4.4): a hit is inadmissible when the tier-1 score is
rising, rise_i(t) = max(0, s(t) - s(t-w))/w > rho.  Applied here on top of the
validated UCF motion-only causal lookup, using the real VadCLIP tier-1 scores.

Reports, per (theta, rho): admitted hit rate and the fraction of ALL anomalous
clips whose anomaly a hit suppresses.
"""
import numpy as np
E0='/Users/24sf51025/Research/FleetVAD/research/e0'; SNIP=16
d=np.load(f'{E0}/results/motion_features.npz'); motion,lens=d['motion'],d['lens']
t1=np.load(f'{E0}/results/tier1_scores.npz',allow_pickle=True)
ids,scores=t1['video_ids'],t1['scores']
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
prev_s=np.where(ev_si>0, scores[ev_vi,np.maximum(ev_si-1,0)], ev_s)
rise=np.maximum(0.0, ev_s-prev_s)           # per-snippet (w = 1 snippet = 0.53 s)
K=np.ascontiguousarray(motion[ev_vi,ev_si].astype(np.float32)); K/=np.maximum(np.linalg.norm(K,axis=1,keepdims=True),1e-9)
sim=np.full(n,-2.0,np.float32); nn=np.full(n,-1,np.int64)
for a in range(0,n,2048):
    b=min(a+2048,n); bs=np.full(b-a,-2.0,np.float32); bj=np.full(b-a,-1,np.int64)
    for c0 in range(0,b,16384):
        c1=min(c0+16384,b); S=K[a:b]@K[c0:c1].T
        S[np.arange(c0,c1)[None,:]>=np.arange(a,b)[:,None]]=-2.0
        loc=S.argmax(1); v=S[np.arange(b-a),loc]; u=v>bs; bs[u]=v[u]; bj[u]=c0+loc[u]
    sim[a:b]=bs; nn[a:b]=np.where(bs>-2.0,bj,-1)
ok=nn>=0; nn_lab=np.where(ok,ev_lab[np.clip(nn,0,None)],-1)
n_anom=(ev_lab==1).sum()
print(f"clips={n}  anomalous={n_anom} ({n_anom/n:.3f})  rise>0 on {(rise>0).mean():.3f} of clips")
print(f"rise percentiles: p50={np.percentile(rise,50):.4f} p90={np.percentile(rise,90):.4f} p99={np.percentile(rise,99):.4f}")
print(f"\n{'theta':>6} {'rho':>7} {'hit_rate':>9} {'suppressed_anom':>16} {'agree':>7}")
for th in (0.90,0.96,0.98,0.99):
    for rho in (1e9, 0.20, 0.10, 0.05, 0.01, 0.0):
        hit=ok&(sim>=th)&(rise<=rho)
        if hit.sum()==0:
            print(f"{th:>6.2f} {rho:>7} {'0':>9}"); continue
        supp=((hit)&(ev_lab==1)&(nn_lab==0)).sum()/n_anom
        agree=(ev_lab[hit]==nn_lab[hit]).mean()
        lbl='none' if rho>1 else f"{rho:.2f}"
        print(f"{th:>6.2f} {lbl:>7} {hit.mean():>9.3f} {supp:>16.3f} {agree:>7.3f}")
