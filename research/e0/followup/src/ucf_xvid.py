"""UCF-Crime, motion-only signature (alpha=0), three candidate pools.

alpha=0 is the only UCF setting where E0 found non-trivial cross-video hit mass
(T2+T3 = 10.8% of clips at the aggregate-95% operating point). Motion features
are cached locally, so this reproduces E0's ALL curve as a check and adds the
XVID pool (same-video excluded) -- the shared-cache-vs-per-stream-cache question.
"""
import numpy as np, os, json
E0='/Users/24sf51025/Research/FleetVAD/research/e0'
d=np.load(f'{E0}/results/motion_features.npz')
motion, lens = d['motion'], d['lens']          # (290, 6750, 48)
ids=np.load(f'{E0}/results/tier1_scores.npz',allow_pickle=True)['video_ids']
SNIP, FPS = 16, 30.0

# per-snippet labels from the official temporal annotations (same as redundancy.py)
ann={}
for line in open(f'{E0}/third_party/VadCLIP/list/Temporal_Anomaly_Annotation.txt'):
    p=line.split()
    if not p: continue
    vid=p[0].replace('.mp4',''); segs=[]
    for a,b in zip(p[2::2],p[3::2]):
        a,b=int(a),int(b)
        if a>=0 and b>=0: segs.append((a,b))
    ann[vid]=segs
labels=np.zeros((len(ids), lens.max()), dtype=np.int8)
for i,vid in enumerate(ids):
    for a,b in ann.get(vid,[]):
        s0=max((a-1)//SNIP,0); s1=min((b-1)//SNIP, lens[i]-1)
        labels[i,s0:s1+1]=1

n_streams=32
cams=[[] for _ in range(n_streams)]
for i in range(len(ids)): cams[i%n_streams].append(i)
spf=SNIP/FPS; events=[]
for c,cam in enumerate(cams):
    t=0.0
    for vi in cam:
        for si in range(lens[vi]): events.append((t,c,vi,si)); t+=spf
events.sort(key=lambda e:(e[0],e[1]))
ev_vi=np.array([e[2] for e in events]); ev_si=np.array([e[3] for e in events])
ev_lab=labels[ev_vi,ev_si]; n=len(events)
K=np.ascontiguousarray(motion[ev_vi,ev_si].astype(np.float32))
K/=np.maximum(np.linalg.norm(K,axis=1,keepdims=True),1e-9)
print(f"clips={n} anom={ev_lab.mean():.4f}")

def causal_top1(exclude_same_video):
    sim=np.full(n,-2.0,dtype=np.float32); nn=np.full(n,-1,dtype=np.int64)
    QB=2048
    for a in range(0,n,QB):
        b=min(a+QB,n)
        if a==0 and b==1: continue
        bs=np.full(b-a,-2.0,dtype=np.float32); bj=np.full(b-a,-1,dtype=np.int64)
        CB=16384
        for c0 in range(0,b,CB):
            c1=min(c0+CB,b)
            S=K[a:b]@K[c0:c1].T
            qi=np.arange(a,b)[:,None]; ci=np.arange(c0,c1)[None,:]
            S[ci>=qi]=-2.0
            if exclude_same_video:
                S[ev_vi[a:b][:,None]==ev_vi[c0:c1][None,:]]=-2.0
            loc=S.argmax(axis=1); v=S[np.arange(b-a),loc]
            upd=v>bs; bs[upd]=v[upd]; bj[upd]=c0+loc[upd]
        sim[a:b]=bs; nn[a:b]=np.where(bs>-2.0,bj,-1)
    return sim,nn

thetas=np.round(np.arange(0.80,0.991,0.01),2)
out={}
for pool,excl in (('ALL',False),('XVID',True)):
    sim,nn=causal_top1(excl)
    ok=nn>=0; nn_lab=np.where(ok,ev_lab[np.clip(nn,0,None)],-1)
    rows=[]
    for th in thetas:
        hit=ok&(sim>=th); h=int(hit.sum())
        if h==0: continue
        rows.append(dict(theta=float(th),hit_rate=h/n,
                         agreement=float((ev_lab[hit]==nn_lab[hit]).mean()),n_hits=h,
                         xvid_frac=float((ev_vi[hit]!=ev_vi[np.clip(nn[hit],0,None)]).mean())))
    out[pool]=rows
    print(f"\n--- pool={pool} (motion-only, alpha=0) ---")
    print(f"{'theta':>6} {'hit':>7} {'agree':>7} {'nhits':>7}")
    for r in rows:
        if abs((r['theta']*100)%5)<1e-6 or r['theta'] in (0.96,0.99):
            print(f"{r['theta']:>6.2f} {r['hit_rate']:>7.3f} {r['agreement']:>7.3f} {r['n_hits']:>7}")
    safe=[r for r in rows if r['agreement']>=0.95]
    if safe:
        b=max(safe,key=lambda r:r['hit_rate'])
        print(f"  SAFE op point: theta={b['theta']:.2f} hit={b['hit_rate']:.3f} agree={b['agreement']:.3f}")
    else:
        b=max(rows,key=lambda r:r['agreement'])
        print(f"  NO safe point; max agree={b['agreement']:.3f} at theta={b['theta']:.2f} (hit={b['hit_rate']:.3f})")

p=ev_lab.mean()
print(f"\nbaselines: random-pair agreement={p*p+(1-p)**2:.3f}  majority-class={max(p,1-p):.3f}")
json.dump(out,open('/Users/24sf51025/Research/FleetVAD/research/e0/followup/results/ucf_xvid.json','w'),indent=1)
