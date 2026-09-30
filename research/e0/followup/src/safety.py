"""Is `label_agreement >= 0.95` a meaningful safety bar?

Recompute at each theta: raw agreement, agreement conditional on the clip being
anomalous (= fraction of anomalies whose reused verdict is NOT suppressed),
Cohen's kappa, and the majority-class baseline. Both datasets, ALL pool.
"""
import numpy as np
E0='/Users/24sf51025/Research/FleetVAD/research/e0'
SNIP=16

def metrics(ev_lab, sim, nn, thetas, tag):
    ok=nn>=0; nn_lab=np.where(ok,ev_lab[np.clip(nn,0,None)],-1)
    p=ev_lab.mean()
    print(f"\n=== {tag} | anomaly base rate {p:.3f} | majority-class agreement {max(p,1-p):.3f} ===")
    print(f"{'theta':>6} {'hit':>6} {'agree':>7} {'agree|anom':>11} {'agree|norm':>11} {'kappa':>7} {'suppressed':>11}")
    for th in thetas:
        hit=ok&(sim>=th)
        if hit.sum()==0: continue
        a=ev_lab[hit]; b=nn_lab[hit]
        agree=(a==b).mean()
        aa=(a==1); an=(a==0)
        ag_a=(b[aa]==1).mean() if aa.any() else float('nan')
        ag_n=(b[an]==0).mean() if an.any() else float('nan')
        po=agree
        pa=a.mean(); pb=(b==1).mean()
        pe=pa*pb+(1-pa)*(1-pb)
        kappa=(po-pe)/(1-pe) if pe<1 else float('nan')
        # suppressed anomalies as a fraction of ALL anomalous clips in the workload
        supp=((hit)&(ev_lab==1)&(nn_lab==0)).sum()/max((ev_lab==1).sum(),1)
        print(f"{th:>6.2f} {hit.mean():>6.3f} {agree:>7.3f} {ag_a:>11.3f} {ag_n:>11.3f} {kappa:>7.3f} {supp:>11.3f}")

# ---------- UCF, motion-only (alpha=0), ALL pool ----------
d=np.load(f'{E0}/results/motion_features.npz'); motion,lens=d['motion'],d['lens']
ids=np.load(f'{E0}/results/tier1_scores.npz',allow_pickle=True)['video_ids']
ann={}
for line in open(f'{E0}/third_party/VadCLIP/list/Temporal_Anomaly_Annotation.txt'):
    p=line.split()
    if not p: continue
    segs=[(int(a),int(b)) for a,b in zip(p[2::2],p[3::2]) if int(a)>=0 and int(b)>=0]
    ann[p[0].replace('.mp4','')]=segs
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
ev_vi=np.array([e[2] for e in ev]); ev_si=np.array([e[3] for e in ev])
ev_lab=labels[ev_vi,ev_si]; n=len(ev)
K=np.ascontiguousarray(motion[ev_vi,ev_si].astype(np.float32)); K/=np.maximum(np.linalg.norm(K,axis=1,keepdims=True),1e-9)
sim=np.full(n,-2.0,np.float32); nn=np.full(n,-1,np.int64)
for a in range(0,n,2048):
    b=min(a+2048,n); bs=np.full(b-a,-2.0,np.float32); bj=np.full(b-a,-1,np.int64)
    for c0 in range(0,b,16384):
        c1=min(c0+16384,b); S=K[a:b]@K[c0:c1].T
        S[np.arange(c0,c1)[None,:]>=np.arange(a,b)[:,None]]=-2.0
        loc=S.argmax(1); v=S[np.arange(b-a),loc]; u=v>bs; bs[u]=v[u]; bj[u]=c0+loc[u]
    sim[a:b]=bs; nn[a:b]=np.where(bs>-2.0,bj,-1)
metrics(ev_lab,sim,nn,[0.80,0.90,0.95,0.96,0.98,0.99],'UCF-Crime  motion-only (alpha=0), any-neighbour pool')

# ---------- ShanghaiTech, alpha=1.0 scene (E0's reported best) ----------
z=np.load(f'{E0}/results/sht_features.npz',allow_pickle=True)
scene,motion2,labs,meta=list(z['scene']),list(z['motion']),list(z['labels']),z['meta'].tolist()
cams=[[] for _ in range(32)]
for i in range(len(meta)): cams[i%32].append(i)
ev=[]
for c,cam in enumerate(cams):
    t=0.0
    for vi in cam:
        for si in range(meta[vi]['n_snips']): ev.append((t,c,vi,si)); t+=SNIP/25.0
ev.sort(key=lambda e:(e[0],e[1]))
ev_vi=np.array([e[2] for e in ev]); ev_si=np.array([e[3] for e in ev])
ev_lab=np.array([labs[vi][si] for vi,si in zip(ev_vi,ev_si)])
offs=np.cumsum([0]+[m['n_snips'] for m in meta]); flat=np.array([offs[vi]+si for vi,si in zip(ev_vi,ev_si)])
SC=np.concatenate(scene)[flat]
n=len(ev); K=SC.astype(np.float32); K/=np.maximum(np.linalg.norm(K,axis=1,keepdims=True),1e-9)
S=K@K.T; S[np.triu_indices(n,0)]=-2.0
nn=S.argmax(1); sim=S[np.arange(n),nn]; nn=np.where(sim>-2.0,nn,-1)
metrics(ev_lab,sim,nn,[0.80,0.90,0.95,0.98,0.99],'ShanghaiTech  scene-only (alpha=1.0), any-neighbour pool')
