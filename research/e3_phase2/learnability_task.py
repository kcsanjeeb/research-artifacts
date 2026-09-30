#!/usr/bin/env python3
"""Task 2: Clairvoyant learnability probe, DECAF E3-Phase2 oracle subset.

Labels: per-question set of policies judged correct by 72B (ties allowed).
Selector picks ONE policy per question; reward = 1 if picked policy is in the
correct set. Captured gain = selector reward - always-patch reward (best fixed
policy at 45.6%).
Features (available BEFORE answering): question type (derived heuristic
taxonomy from ego4d_oe.json annotations), query length (tokens/chars/words),
store/commit statistics from the deferred arm commit_log.jsonl joined on
video_id+question.
"""
import json
import re
import numpy as np

B = "/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/"
EGO = "/data3/zhuotaotian2_e2/code/ReKV/data/rvs/ego/ego4d_oe.json"
POLICIES = ["seg", "frame", "patch", "default"]
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------- data ----------------
def load_ordered(path):
    recs = json.load(open(path))["judged"]
    seen, out = {}, []
    for r in recs:
        k = (r["video_id"], r["question"])
        i = seen.get(k, 0)
        seen[k] = i + 1
        out.append(((k, i), 1 if r["judge_pred"] == "yes" else 0))
    return out

corr = {}
for pol in POLICIES:
    for k, c in load_ordered(B + f"judged_oracle_{pol}_72b.json"):
        corr.setdefault(k, {})[pol] = c
keys = sorted(corr)
assert all(len(corr[k]) == 4 for k in keys), "missing policy record"
n = len(keys)
print(f"oracle subset: {n} questions")

# ---------------- features ----------------
TYPE_RULES = [
    ("temporal-firstlast", re.compile(r"\b(first|last|earlier|beginning|end of|final)\b", re.I)),
    ("temporal-beforeafter", re.compile(r"\b(before|after|prior to|following)\b", re.I)),
    ("duration", re.compile(r"\bhow long\b|\bhow much time\b|\bhow many (seconds|minutes|hours)\b", re.I)),
    ("counting", re.compile(r"\bhow many\b", re.I)),
    ("task", re.compile(r"what task\b", re.I)),
    ("step", re.compile(r"what step\b", re.I)),
    ("overview", re.compile(r"overview|summar|describe the overall\b", re.I)),
    ("location", re.compile(r"\bwhere\b", re.I)),
    ("reason-why", re.compile(r"\bwhy\b", re.I)),
    ("action-doing", re.compile(r"doing|happen", re.I)),
]
def qtype(q):
    for name, rx in TYPE_RULES:
        if rx.search(q):
            return name
    return "other-what"

feat_raw = {}
for line in open(B + "answers_7b_b_deferred/commit_log.jsonl"):
    r = json.loads(line)
    k = (r["video_id"], r["question"])
    i = feat_raw.get(k + ("__occ",), 0)
    feat_raw[k + ("__occ",)] = i + 1
    kk = (k, i)
    if kk not in corr:
        continue
    p1 = r.get("pass1", {})
    sb = r.get("store_bytes", {})
    q = r["question"]
    feat_raw[kk] = dict(
        entropy=p1.get("mean_entropy", np.nan),
        n_tokens=p1.get("n_tokens", np.nan),
        committed_slots=p1.get("committed_slots", np.nan),
        n_committed_grains=p1.get("n_committed_grains", np.nan),
        store_bytes_total=sb.get("store_bytes_total", np.nan),
        store_bytes_kv=sb.get("store_bytes_kv", np.nan),
        n_segments=sb.get("n_segments", np.nan),
        pass1_policy=p1.get("policy", "NA"),
        pass1_level=p1.get("level", "NA"),
        uncertain=1.0 if r.get("uncertain") else 0.0,
        q_words=len(q.split()),
        q_chars=len(q),
        qtype=qtype(q),
    )

missing = [k for k in keys if k not in feat_raw]
print(f"features joined: {n - len(missing)}/{n} (missing {len(missing)})")
keys = [k for k in keys if k in feat_raw]
n = len(keys)

CONT = ["entropy", "n_tokens", "committed_slots", "n_committed_grains",
        "store_bytes_total", "store_bytes_kv", "n_segments", "uncertain",
        "q_words", "q_chars"]
CAT = ["qtype", "pass1_policy", "pass1_level"]

Xc = np.array([[feat_raw[k][f] for f in CONT] for k in keys], float)
types = np.array([feat_raw[k]["qtype"] for k in keys])
# one-hot question type (fixed taxonomy order)
type_levels = sorted(set(types))
Xt = np.column_stack([(types == t).astype(float) for t in type_levels])
X = np.column_stack([Xc, Xt])

Y = np.array([[corr[k][p] for p in POLICIES] for k in keys], float)  # reward matrix n x 4

def acc(idx, pol_i):
    return float(Y[idx, pol_i].mean())

print("\n--- label balance ---")
best_fixed = max(POLICIES, key=lambda p: acc(np.arange(n), POLICIES.index(p)))
for p in POLICIES:
    print(f"  {p:<8} acc={acc(np.arange(n), POLICIES.index(p))*100:5.1f}%  (#correct {int(Y[:,POLICIES.index(p)].sum())})")
set_sizes = Y.sum(1)
print(f"  correct-set size distribution: " +
      ", ".join(f"{int(s)}:{int((set_sizes==s).sum())}" for s in sorted(set(set_sizes))))
uniq = {}
for k in keys:
    correct = {p for p in POLICIES if corr[k][p] == 1}
    if len(correct) == 1:
        uniq[list(correct)[0]] = uniq.get(list(correct)[0], 0) + 1
print(f"  unique-correct counts: {uniq}  (of {n})")
none_correct = int((set_sizes == 0).sum())
print(f"  questions where ALL policies wrong: {none_correct}")
print(f"  best fixed policy: {best_fixed}")

# ---------------- selectors ----------------
def stump_fit_reward(Xtr, Ytr, col_names):
    """Decision stump maximizing train reward; categorical handled separately."""
    ntr = len(Ytr)
    base = Ytr.mean(0).max()
    best = ("const", None, base, None)
    for j in range(Xtr.shape[1]):
        vals = np.unique(Xtr[:, j])
        if len(vals) < 2:
            continue
        ths = (vals[:-1] + vals[1:]) / 2
        for th in ths:
            left = Xtr[:, j] <= th
            if left.sum() == 0 or left.sum() == ntr:
                continue
            rl = Ytr[left].mean(0).max()
            rr = Ytr[~left].mean(0).max()
            r = (left.mean() * rl + (~left).mean() * rr)
            if r > best[2]:
                pred_l = int(np.argmax(Ytr[left].mean(0)))
                pred_r = int(np.argmax(Ytr[~left].mean(0)))
                best = (f"{col_names[j]} <= {th:.4g}", (j, th, pred_l, pred_r), r, (rl, rr))
    return best

def stump_predict(Xte, model):
    if model[0] == "const":
        return np.full(len(Xte), int(model[3])) if model[3] is not None else np.full(len(Xte), -1)
    _, (j, th, pl, pr), _, _ = model
    return np.where(Xte[:, j] <= th, pl, pr)

def type_cond_fit(types_tr, Ytr):
    table = {}
    for t in np.unique(types_tr):
        idx = np.where(types_tr == t)[0]
        table[t] = int(np.argmax(Ytr[idx].mean(0)))
    return table

def type_cond_predict(types_te, table, fallback):
    return np.array([table.get(t, fallback) for t in types_te])

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def lr_fit(Xtr, Ytr, epochs=3000, lr=0.5, l2=1e-3):
    ntr, d = Xtr.shape
    K = Ytr.shape[1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    W = np.zeros((d, K)); b = np.zeros(K)
    for _ in range(epochs):
        P = softmax(Z @ W + b)
        G = (P - Ytr) / ntr
        W -= lr * (Z.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return (W, b, mu, sd)

def lr_predict(Xte, model):
    W, b, mu, sd = model
    return np.argmax(softmax((Xte - mu) / sd @ W + b), axis=1)

def reward_of(pred_idx, Yte):
    return float(Yte[np.arange(len(Yte)), pred_idx].mean())

# ---------------- protocol ----------------
idx = np.arange(n)
rng.shuffle(idx)
half = n // 2
tr, te = idx[:half], idx[half:]

patch_i = POLICIES.index("patch")
fallback = int(np.argmax(Y[tr].mean(0)))

# (i) decision stump: continuous features (incl. q one-hots via one-hot == value)
names = CONT + [f"type={t}" for t in type_levels]
stump = stump_fit_reward(X[tr], Y[tr], names)
pred_stump = stump_predict(X[te], stump)
# stump on categorical equality (qtype, pass1_policy, pass1_level)
cat_names = {0: types, 1: np.array([feat_raw[k]["pass1_policy"] for k in keys]),
             2: np.array([feat_raw[k]["pass1_level"] for k in keys])}
best_cat = None
for ci, (name, arr) in enumerate(cat_names.items()):
    for lev in np.unique(arr[tr]):
        pred = np.where(arr == lev, 1, 0)
        rl = Y[tr][pred[tr] == 1].mean(0).max() if (pred[tr] == 1).any() else 0
        rr = Y[tr][pred[tr] == 0].mean(0).max() if (pred[tr] == 0).any() else 0
        r = (pred[tr].mean() * rl + (1 - pred[tr].mean()) * rr)
        if best_cat is None or r > best_cat[2]:
            best_cat = (f"{['qtype','pass1_policy','pass1_level'][ci]}=={lev}", lev, r, int(np.argmax(Y[tr][pred[tr]==1].mean(0))), int(np.argmax(Y[tr][pred[tr]==0].mean(0))))

# (ii) logistic regression
lr = lr_fit(X[tr], Y[tr])
pred_lr = lr_predict(X[te], lr)

# (iii) question-type-conditioned
tct = type_cond_fit(types[tr], Y[tr])
pred_tc = type_cond_predict(types[te], tct, fallback)

# ---------------- evaluation ----------------
def boot_ci_gain(pred_idx_te, Bn=2000):
    g = np.empty(Bn)
    m = len(te)
    for b_ in range(Bn):
        s = np.random.default_rng(b_).integers(0, m, m)
        pb = Y[te][s, pred_idx_te[s]].mean()
        pp = Y[te][s, patch_i].mean()
        g[b_] = pb - pp
    return np.percentile(g, [2.5, 97.5])

print("\n--- held-out (seed 42, 50/50) ---")
res = {}
for name, pred in [("decision_stump", pred_stump), ("logistic_reg", pred_lr),
                   ("type_conditioned", pred_tc)]:
    r = reward_of(pred, Y[te]); p = acc(te, patch_i)
    gain = r - p
    lo, hi = boot_ci_gain(pred)
    res[name] = dict(heldout_reward=r, patch=p, gain=gain, ci=[lo, hi])
    print(f"  {name:<18} held-out {r*100:5.1f}%  patch {p*100:5.1f}%  gain {gain*100:+5.2f}pt  CI95 [{lo*100:+5.2f},{hi*100:+5.2f}]")
print(f"  stump rule: {stump[0]} (train reward {stump[2]:.3f})")
print(f"  best categorical stump: {best_cat[0]} (train reward {best_cat[2]:.3f})")
print(f"  type->policy table (train): {tct}")

# repeated splits for robustness
rep = []
for s in range(200):
    r2 = np.random.default_rng(1000 + s)
    idx2 = r2.permutation(n)
    tr2, te2 = idx2[:half], idx2[half:]
    lr2 = lr_fit(X[tr2], Y[tr2])
    g = reward_of(lr_predict(X[te2], lr2), Y[te2]) - acc(te2, patch_i)
    t2 = type_cond_fit(types[tr2], Y[tr2])
    g2 = reward_of(type_cond_predict(types[te2], t2, fallback), Y[te2]) - acc(te2, patch_i)
    st2 = stump_fit_reward(X[tr2], Y[tr2], names)
    g3 = reward_of(stump_predict(X[te2], st2), Y[te2]) - acc(te2, patch_i)
    rep.append((g, g2, g3))
rep = np.array(rep)
print("\n--- repeated splits (200x 50/50), mean held-out gain over patch ---")
for i, nm in enumerate(["logistic_reg", "type_conditioned", "decision_stump"]):
    m_, sd_ = rep[:, i].mean(), rep[:, i].std()
    print(f"  {nm:<18} {m_*100:+5.2f}pt +- {sd_*100:4.2f}pt  (frac positive: {(rep[:,i]>0).mean():.2f})")

# ---------------- oracle-selector upper bound (in-sample, same features) ----------------
lr_all = lr_fit(X, Y)
ins_lr = reward_of(lr_predict(X, lr_all), Y)
# memorization lookup: identical feature vector -> best policy on those rows
ins_nn = []
for i in range(n):
    same = np.where((X == X[i]).all(1))[0]
    ins_nn.append(np.argmax(Y[same].mean(0)))
ins_nn = reward_of(np.array(ins_nn), Y)
# in-sample stump
st_all = stump_fit_reward(X, Y, names)
ins_st = reward_of(stump_predict(X, st_all), Y)
print(f"\n--- oracle-selector upper bounds (fit on all {n}, in-sample) ---")
print(f"  memorization lookup: {ins_nn*100:.1f}%  (gain {ins_nn*100 - acc(np.arange(n), patch_i)*100:+.1f}pt)")
print(f"  logistic reg in-sample: {ins_lr*100:.1f}%  (gain {(ins_lr - acc(np.arange(n), patch_i))*100:+.1f}pt)")
print(f"  stump in-sample: {ins_st*100:.1f}%  (gain {(ins_st - acc(np.arange(n), patch_i))*100:+.1f}pt)")
print(f"  best-fixed patch in-sample: {acc(np.arange(n), patch_i)*100:.1f}%")

out = dict(n=n, seed=SEED,
           policy_acc={p: acc(np.arange(n), POLICIES.index(p)) for p in POLICIES},
           correct_set_size_dist={str(int(s)): int((set_sizes[:n] == s).sum()) for s in sorted(set(set_sizes))},
           unique_correct=uniq, all_wrong=none_correct,
           stump_rule=stump[0], type_table=tct,
           heldout={k: v for k, v in res.items()},
           repeated=dict(logistic_reg=[float(rep[:,0].mean()), float(rep[:,0].std())],
                         type_conditioned=[float(rep[:,1].mean()), float(rep[:,1].std())],
                         decision_stump=[float(rep[:,2].mean()), float(rep[:,2].std())]),
           insample_bounds=dict(memorization=ins_nn, logistic_reg=ins_lr, stump=ins_st))
json.dump(out, open("/data3/zhuotaotian2_e2/deliverables/e3_phase2/learnability_results.json", "w"), indent=2)
print("\nwrote learnability_results.json")
