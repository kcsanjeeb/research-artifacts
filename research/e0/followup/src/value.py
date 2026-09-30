"""E0.5 -- does the value function discriminate outcome-changing calls?

Population: ESCALATED clips only (PerStreamGate, tau=0.5,
margin=0.2603611499071121 -- E0's 8.6% calibration). Label 1 (primary): the VLM
call would FLIP the tier-1 decision, i.e. (s > tau) != gt. Label 2: GT anomaly.
Label 3: first snippet of a contiguous GT anomaly run.

Interpretations documented per E0.5 rule 8 style (no GT leakage into features):
- state machine is tier-1-only: NORMAL->SUSPECT on s>tau; SUSPECT->CONFIRMED
  after 3 consecutive snippets with s>tau; any->NORMAL after 3 consecutive with
  s<tau. g = {NORMAL:1.0, SUSPECT:1.5, CONFIRMED:0.1} (MASTER_PLAN 4.2/4.6).
- stale: delta = snippets since the stream's last ESCALATION (calls only happen
  there); stale = min(1, delta/32); stale starts at 1.
- u = clip(1-2|s-tau|,0,1) is a monotone transform of |s-tau|, so the
  "abs(s-tau) alone" floor (E0.5 5.3) IS the u row; stated, not duplicated.
- rankers are oriented so "higher = more worth serving".
- AUC: average-rank Mann-Whitney (tie-correct), pure numpy.
- Contention windows and the headroom sim regenerate the EXACT demand.simulate
  timeline (round-robin, looping playlists, horizon = max(longest playlist,
  900s), CLIP_PERIOD=1.0s) and verify it reproduces demand.json's N=32 stats.
"""
import json
import os

import numpy as np

E0 = '/Users/24sf51025/Research/FleetVAD/research/e0'
OUT = f'{E0}/followup/results'
FIG = f'{OUT}/figures'
SNIP = 16
FPS = 30.0
TAU, MARGIN = 0.5, 0.2603611499071121
SEED = 0

rng = np.random.RandomState(SEED)


# ------------------------------------------------------------------ artifacts

def load_all():
    t1 = np.load(f'{E0}/results/tier1_scores.npz', allow_pickle=True)
    ids = t1['video_ids'].tolist()
    scores, lens = t1['scores'], t1['lens']
    dem = json.load(open(f'{E0}/results/demand.json'))
    capacity = dem['capacity_clips_per_s']
    ann = {}
    for line in open(f'{E0}/third_party/VadCLIP/list/Temporal_Anomaly_Annotation.txt'):
        p = line.split()
        if not p:
            continue
        ann[p[0].replace('.mp4', '')] = [(int(a), int(b)) for a, b in
                                         zip(p[2::2], p[3::2])
                                         if int(a) >= 0 and int(b) >= 0]
    labels = np.zeros((len(ids), lens.max()), dtype=np.int8)
    for i, vid in enumerate(ids):
        for a, b in ann.get(vid, []):
            labels[i, max((a - 1) // SNIP, 0):min((b - 1) // SNIP, lens[i] - 1) + 1] = 1
    return ids, scores, lens, labels, capacity, dem


# --------------------------------------------------------------------- labels

def clip_labels(s, gt):
    """label1 = tier-1 decision wrong; label2 = gt; label3 = onset."""
    l1 = ((s > TAU).astype(np.int8) != gt).astype(np.int8)
    l3 = np.zeros_like(gt)
    prev = 0
    for t in range(len(gt)):
        if gt[t] == 1 and prev == 0:
            l3[t] = 1
        prev = gt[t]
    return l1, gt.astype(np.int8), l3


def state_machine(s):
    """Tier-1-only NORMAL/SUSPECT/CONFIRMED series -> g values."""
    gmap = {0: 1.0, 1: 1.5, 2: 0.1}
    state = 0
    run_hi, run_lo = 0, 0
    g = np.zeros(len(s))
    for t in range(len(s)):
        if s[t] > TAU:
            run_hi += 1
            run_lo = 0
        else:
            run_lo += 1
            run_hi = 0
        if state == 0 and s[t] > TAU:
            state = 1
        if state == 1 and run_hi >= 3:
            state = 2
        if run_lo >= 3:
            state = 0
        g[t] = gmap[state]
    return g


def value_terms(s, escalated_mask):
    """All terms for one video's snippet score series."""
    u = np.clip(1 - 2 * np.abs(s - TAU), 0, 1)
    g = state_machine(s)
    stale = np.zeros(len(s))
    last_esc = -32
    for t in range(len(s)):
        stale[t] = min(1.0, (t - last_esc) / 32.0)
        if escalated_mask[t]:
            last_esc = t
    terms = {'u': u, 'g_state': g, 'stale': stale}
    for w in (1, 2, 4):
        rise = np.zeros(len(s))
        for t in range(w, len(s)):
            rise[t] = max(0.0, (s[t] - s[t - w]) / w)
        terms[f'rise_w{w}'] = rise
    return terms


def combine_v(terms, w, lam):
    return terms['u'] * terms['g_state'] * (1 + lam * terms[f'rise_w{w}']) * terms['stale']


# -------------------------------------------------------------------- metrics

def auc_rank(score, label):
    """Tie-correct AUC via average ranks."""
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos = label == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_lift(score, label, ks=(0.10, 0.25, 0.50, 0.75)):
    n = len(score)
    base = label.mean()
    order = np.argsort(-score, kind='mergesort')
    out = {}
    for k in ks:
        m = max(1, int(round(k * n)))
        p = label[order[:m]].mean()
        out[f'precision@{int(k * 100)}'] = float(p)
        out[f'lift@{int(k * 100)}'] = float(p / base) if base > 0 else None
    return out


def evaluate(score, label):
    r = {'auc': auc_rank(score, label)}
    r.update(precision_lift(score, label))
    return r


# ------------------------------------------------------- population (B, sec5)

def build_population(ids, scores, lens, labels):
    """Escalated clips across all 290 videos, with terms and labels."""
    pop = {'s': [], 'gt': [], 'l1': [], 'l2': [], 'l3': [], 'vi': [], 'si': []}
    term_acc = {}
    for vi in range(len(ids)):
        s = scores[vi, :lens[vi]]
        gt = labels[vi, :lens[vi]]
        esc = np.abs(s - TAU) < MARGIN
        l1, l2, l3 = clip_labels(s, gt)
        terms = value_terms(s, esc)
        pop['s'].append(s[esc])
        pop['gt'].append(gt[esc])
        pop['l1'].append(l1[esc])
        pop['l2'].append(l2[esc])
        pop['l3'].append(l3[esc])
        pop['vi'] += [vi] * int(esc.sum())
        pop['si'] += list(np.flatnonzero(esc))
        for k, v in terms.items():
            term_acc.setdefault(k, []).append(v[esc])
    for k in ('s', 'gt', 'l1', 'l2', 'l3'):
        pop[k] = np.concatenate(pop[k])
    pop['vi'] = np.array(pop['vi'])
    pop['si'] = np.array(pop['si'])
    for k in term_acc:
        term_acc[k] = np.concatenate(term_acc[k])
    pop['terms'] = term_acc
    return pop


# ------------------------------------------------- simulation (C, D; sec 6-7)

def sim_events(n_streams, lens, period=1.0):
    """Exact replica of demand.simulate's timeline; returns escalation events
    (epoch, vi, si) and the demand bins. Verified against demand.json."""
    spf = SNIP / FPS
    cams = [[] for _ in range(n_streams)]
    for i in range(len(lens)):
        cams[i % n_streams].append(i)
    cam_tl, cam_dur = [], []
    for cam in cams:
        tl = [(vi, si) for vi in cam for si in range(lens[vi])]
        cam_tl.append(tl)
        cam_dur.append(len(tl) * spf)
    horizon = max(max(cam_dur), 900.0)
    n_bins = int(horizon) + 1
    demand = np.zeros(n_bins)
    events = []
    for tl, dur in zip(cam_tl, cam_dur):
        if not tl:
            continue
        t = 0.0
        while t < horizon:
            pos = t % dur
            vi, si = tl[min(int(pos / spf), len(tl) - 1)]
            events.append((int(t), vi, si))
            t += period
    return events, demand, horizon


def demand_bins(events, scores, n_bins):
    d = np.zeros(n_bins)
    for ep, vi, si in events:
        if abs(scores[vi, si] - TAU) < MARGIN:
            d[ep] += 1
    return d


# ------------------------------------------------------------------ headroom

def headroom(events, feats, capacity, queue_cap=30):
    """Simulate serving of escalation events under 5 policies.
    feats: dict per event index -> v, l1, l3. Budget: capacity per 1s epoch,
    accumulated (continuous server, use-it-or-lose-it within epoch)."""
    by_epoch = {}
    for k, (ep, vi, si) in enumerate(events):
        by_epoch.setdefault(ep, []).append(k)
    max_ep = max(by_epoch) + 1
    policies = {}
    for name in ('FIFO', 'Random', 'FIFO+Queue', 'ValueTopK', 'RawTopK', 'Oracle'):
        policies[name] = {'served': [], 'dropped': 0}
    for name in policies:
        budget = 0.0
        queue = []  # list of event indices waiting
        for ep in range(max_ep):
            budget += capacity
            arrivals = by_epoch.get(ep, [])
            if name in ('FIFO', 'Random'):
                cands = list(arrivals)
                if name == 'Random':
                    rng.shuffle(cands)
                k = int(min(len(cands), int(budget)))
                for idx in cands[:k]:
                    policies[name]['served'].append((idx, ep))
                    budget -= 1
                policies[name]['dropped'] += len(cands) - k
            else:
                for idx in arrivals:
                    if len(queue) < queue_cap:
                        queue.append(idx)
                    else:
                        # overflow: FIFO+Queue drops oldest; Value/Oracle drop
                        # the least deserving queued request
                        if name == 'FIFO+Queue':
                            queue.pop(0)
                        elif name == 'ValueTopK':
                            queue.pop(int(np.argmin([feats[q]['v'] for q in queue])))
                        elif name == 'RawTopK':
                            queue.pop(int(np.argmin([feats[q]['s'] for q in queue])))
                        else:
                            queue.pop(int(np.argmin([feats[q]['l1'] for q in queue])))
                        policies[name]['dropped'] += 1
                        queue.append(idx)
                    # arrivals beyond capacity stay queued
                key = (lambda q: -ep) if name == 'FIFO+Queue' else (
                    (lambda q: -feats[q]['v']) if name == 'ValueTopK' else (
                        (lambda q: -feats[q]['s']) if name == 'RawTopK'
                        else (lambda q: -feats[q]['l1'])))
                serve = min(len(queue), int(budget))
                if name == 'FIFO+Queue':
                    chosen = queue[:serve]
                    queue = queue[serve:]
                else:
                    queue.sort(key=key)
                    chosen = queue[:serve]
                    queue = queue[serve:]
                for idx in chosen:
                    policies[name]['served'].append((idx, ep))
                    budget -= 1
        policies[name]['dropped'] += len(queue)
    # metrics
    n_l1 = sum(1 for f in feats.values() if f['l1'] == 1)
    n_l3 = sum(1 for f in feats.values() if f['l3'] == 1)
    out = {}
    for name, p in policies.items():
        served_idx = {idx: ep_s for idx, ep_s in p['served']}
        l1_served = sum(1 for i in served_idx if feats[i]['l1'] == 1)
        l3_served = sum(1 for i in served_idx if feats[i]['l3'] == 1)
        delays = [ep_s - events[i][0] for i, ep_s in served_idx.items()]
        out[name] = {
            'frac_label1_served': l1_served / max(n_l1, 1),
            'frac_label3_served': l3_served / max(n_l3, 1),
            'frac_all_served': len(served_idx) / len(events),
            'delay_mean': float(np.mean(delays)) if delays else None,
            'delay_p95': float(np.percentile(delays, 95)) if delays else None,
            'dropped': p['dropped'],
        }
    return out


# ----------------------------------------------------------------------- main

def main():
    os.makedirs(FIG, exist_ok=True)
    ids, scores, lens, labels, capacity, dem = load_all()
    print(f"capacity={capacity:.4f}  tau={TAU}  margin={MARGIN:.4f}")

    # ---------- A/B: population + discrimination ----------
    pop = build_population(ids, scores, lens, labels)
    n = len(pop['s'])
    base = {'label1': float(pop['l1'].mean()), 'label2': float(pop['l2'].mean()),
            'label3': float(pop['l3'].mean())}
    print(f"\n[A] escalated clips: {n} ({n / lens.sum():.4f} of all)")
    print(f"    base rates: label1={base['label1']:.4f} label2={base['label2']:.4f} "
          f"label3={base['label3']:.4f}")

    terms = pop['terms']
    W, LAM = 2, 1.0
    v = combine_v(terms, W, LAM)
    rankers = {'v': v, 'u': terms['u'], 'rise': terms[f'rise_w{W}'],
               'stale': terms['stale'], 'g_state': terms['g_state'],
               'raw_s': pop['s'], 'oracle': pop['l1'].astype(float)}
    rand = rng.permutation(n).astype(float)
    rankers['random'] = rand

    disc = {}
    for lname in ('l1', 'l2', 'l3'):
        lab = pop[lname]
        disc[lname] = {r: evaluate(sc, lab) for r, sc in rankers.items()}
    print("\n[B] discrimination on Label 1 (escalated clips only):")
    for r, m in disc['l1'].items():
        print(f"    {r:>8}: auc={m['auc'] if m['auc'] is None else round(m['auc'], 4)} "
              f"lift@50={round(m['lift@50'], 3)} precision@50={round(m['precision@50'], 3)}")

    # lambda / w sweep for v on label1 (rule 3: sweep, don't tune)
    sweep = {}
    for w in (1, 2, 4):
        for lam in (0.0, 0.5, 1.0, 2.0):
            vv = combine_v(terms, w, lam)
            m = evaluate(vv, pop['l1'])
            sweep[f'w{w}_lam{lam}'] = {'auc': m['auc'], 'lift@50': m['lift@50']}
    print("\n[B] v sweep (label1):")
    for k, m in sweep.items():
        print(f"    {k}: auc={round(m['auc'], 4)} lift@50={round(m['lift@50'], 3)}")

    # ---------- C: contention-conditioned ----------
    contention = {}
    for N in (32, 64):
        events, _, horizon = sim_events(N, lens)
        n_bins = int(horizon) + 1
        d = demand_bins(events, scores, n_bins)
        # sanity vs demand.json
        ref = [c for c in dem['curve'] if c['N'] == N and c['clip_period_s'] == 1.0][0]
        print(f"\n[C] N={N}: regen demand mean={d.mean():.3f} p95={np.percentile(d, 95):.1f} "
              f"(demand.json: {ref['mean_demand']:.3f} / {ref['p95_demand']:.1f})")
        hot = d > capacity
        print(f"    contention epochs: {hot.sum()} / {len(d)} ({hot.mean():.3f})")
        # restrict escalated events to hot epochs
        sel_s, sel_l1, sel_l3 = [], [], []
        for ep, vi, si in events:
            s = scores[vi, si]
            if abs(s - TAU) < MARGIN and hot[ep]:
                sel_s.append((vi, si))
        # recompute terms per video for these clips
        idx_vi = np.array([x[0] for x in sel_s])
        idx_si = np.array([x[1] for x in sel_s])
        lab1 = np.array([1 if (scores[vi, si] > TAU) != bool(labels[vi, si]) else 0
                         for vi, si in sel_s])
        lab3 = np.array([clip_labels(scores[vi, :lens[vi]], labels[vi, :lens[vi]])[2][si]
                         for vi, si in sel_s])
        vv = np.zeros(len(sel_s))
        for j, (vi, si) in enumerate(sel_s):
            esc_v = np.abs(scores[vi, :lens[vi]] - TAU) < MARGIN
            t_j = value_terms(scores[vi, :lens[vi]], esc_v)
            vv[j] = combine_v(t_j, W, LAM)[si]
        m1 = evaluate(vv, lab1)
        m3 = evaluate(vv, lab3)
        contention[f'N{N}'] = {'label1': {'v': m1}, 'label3': {'v': m3},
                               'n_escalated_in_windows': len(sel_s),
                               'base_rate_label1': float(lab1.mean()) if len(lab1) else None}
        print(f"    in-window escalated: {len(sel_s)}, base_rate_l1={lab1.mean():.4f}")
        print(f"    v on label1: auc={round(m1['auc'], 4)} lift@50={round(m1['lift@50'], 3)}")

    # ---------- D: headroom ----------
    head = {}
    for N in (16, 32, 64):
        events, _, horizon = sim_events(N, lens)
        feats = {}
        esc_events = []
        for k, (ep, vi, si) in enumerate(events):
            s = scores[vi, si]
            if abs(s - TAU) < MARGIN:
                esc_v = np.abs(scores[vi, :lens[vi]] - TAU) < MARGIN
                t_j = value_terms(scores[vi, :lens[vi]], esc_v)
                feats[len(esc_events)] = {
                    'v': float(combine_v(t_j, W, LAM)[si]),
                    'l1': int((s > TAU) != bool(labels[vi, si])),
                    'l3': int(clip_labels(scores[vi, :lens[vi]], labels[vi, :lens[vi]])[2][si]),
                    's': float(s),
                }
                esc_events.append((ep, vi, si))
        res = headroom(esc_events, feats, capacity)
        head[f'N{N}'] = res
        print(f"\n[D] N={N}: escalated requests={len(esc_events)}")
        for name, m in res.items():
            print(f"    {name:>11}: L1 served={m['frac_label1_served']:.3f} "
                  f"L3 served={m['frac_label3_served']:.3f} "
                  f"p95 delay={m['delay_p95']} dropped={m['dropped']}")

    # ---------- verdict ----------
    m = disc['l1']['v']
    lift50, auc_v = m['lift@50'], m['auc']
    c_ok = all(contention[f'N{N}']['label1']['v']['lift@50'] >= 1.2 for N in (32, 64))
    if auc_v >= 0.70 and lift50 >= 1.5 and c_ok:
        verdict = 'STRONG'
    elif (0.60 <= auc_v < 0.70 or 1.2 <= lift50 < 1.5) and c_ok:
        verdict = 'MARGINAL'
    elif not c_ok:
        verdict = 'FAILS_IN_THE_REGIME_THAT_MATTERS'
    else:
        verdict = 'FAILS'
    beats_raw_s = disc['l1']['v']['auc'] > disc['l1']['raw_s']['auc']
    print(f"\n[VERDICT] {verdict}  (auc={auc_v:.4f}, lift@50={lift50:.3f}, "
          f"beats_raw_s={beats_raw_s})")

    # ---------- figures ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.hist(v, bins=100)
    plt.xlabel('v (u * g * (1+lambda*rise) * stale)')
    plt.ylabel('count')
    plt.title('E0.5: v over escalated clips')
    plt.tight_layout()
    plt.savefig(f'{FIG}/value_hist.png', dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    ks = [10, 25, 50, 75]
    for r in ('v', 'u', 'rise', 'stale', 'raw_s'):
        plt.plot(ks, [disc['l1'][r][f'lift@{k}'] for k in ks], 'o-', label=r)
    plt.axhline(1.0, color='k', ls='--', lw=1)
    plt.xlabel('top-k% of escalated clips served')
    plt.ylabel('lift (precision@k / base rate)')
    plt.title('E0.5: lift@k on Label 1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG}/value_lift.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    names = ['FIFO', 'Random', 'FIFO+Queue', 'ValueTopK', 'RawTopK', 'Oracle']
    Ns = [16, 32, 64]
    x = np.arange(len(Ns))
    wdt = 0.16
    for j, name in enumerate(names):
        plt.bar(x + (j - 2) * wdt,
                [head[f'N{N}'][name]['frac_label1_served'] for N in Ns],
                width=wdt, label=name)
    plt.xticks(x, [f'N={N}' for N in Ns])
    plt.ylabel('fraction of outcome-changing calls served')
    plt.title('E0.5: arbiter headroom vs baselines')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{FIG}/headroom.png', dpi=150)
    plt.close()

    # ---------- JSON ----------
    doc = {
        'config': {'tau': TAU, 'margin': MARGIN, 'lambda': LAM, 'w': W,
                   'seed': SEED, 'capacity': capacity,
                   'state_machine': 'tier-1-only: NORMAL->SUSPECT on s>tau; '
                                    'SUSPECT->CONFIRMED after 3 consecutive s>tau; '
                                    '->NORMAL after 3 consecutive s<tau',
                   'stale': 'min(1, snippets_since_last_escalation/32)',
                   'note': 'u is a monotone transform of abs(s-tau); that floor '
                           'is the u row'},
        'population': {'escalated_clips': int(n),
                       'base_rate_label1': base['label1'],
                       'base_rate_label2': base['label2'],
                       'base_rate_label3': base['label3']},
        'discrimination': {'label1': disc['l1'], 'label2': disc['l2'],
                           'label3': disc['l3']},
        'v_sweep_label1': sweep,
        'contention': contention,
        'headroom': head,
        'v_beats_raw_s': bool(beats_raw_s),
        'verdict': verdict,
    }
    with open(f'{OUT}/value.json', 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {OUT}/value.json")


if __name__ == '__main__':
    main()
