# DECAF Component B (Phase 2, 2026-09-28): multi-grain hedged store + debiased
# dual-signal write-time scoring + cross-grain consistency rerank + uncertainty-
# triggered deferred commitment. Built on corrected Component A (DECAF_FETCH_RANK).
# Everything here is a no-op unless DECAF_B=1.
#
# Grains per segment (MuKV's granularity set): a segment = SEG_FRAMES frames =
# SEG_TOKENS tokens (784 @ block_size 196). Grain units: 4 frame grains
# (196 tok) and 16 patch grains (49 tok); a "segment view" = its 4 frame slots
# committed together. Each grain is an addressable unit with a pre-RoPE content
# key (mean K), retained K/V (fp16, or 4-bit packed when DECAF_QUANT=1), and
# write-time importance from MuKV's dual signal (attention + FFT frequency),
# optionally position-detrended (DECAF_B_BIAS=debias, default) — the C1
# debiasing evidence.
#
# Storage accounting matches MuKV (bytes of retained K/V + keys + sidecar,
# reported as GB/h by the analysis script) so C3 comparisons hold.
import json
import math
import os

import torch

DECAF_B = os.environ.get('DECAF_B', '0') == '1'
DECAF_B_COMMIT = os.environ.get('DECAF_B_COMMIT', 'deferred')  # 'deferred' | 'writetime'
DECAF_B_TAX = float(os.environ.get('DECAF_B_TAX', '0.5'))      # per-segment token budget fraction (of SEG_TOKENS)
DECAF_B_BIAS = os.environ.get('DECAF_B_BIAS', 'debias')        # 'debias' | 'raw'
DECAF_B_GAMMA = float(os.environ.get('DECAF_B_GAMMA', '0.5'))  # cross-grain consistency weight
DECAF_B_ENTROPY = float(os.environ.get('DECAF_B_ENTROPY', '1.8'))  # uncertainty trigger (mean token nats)
DECAF_B_POLICY = os.environ.get('DECAF_B_POLICY', 'default')   # 'default'|'seg'|'frame'|'patch' (oracle policies)
DECAF_B_NOPASS2 = os.environ.get('DECAF_B_NOPASS2', '0') == '1'
DECAF_B_LOG = os.environ.get('DECAF_B_LOG', '')                # per-question commit log (jsonl)
DECAF_B_ANALYZE = os.environ.get('DECAF_B_ANALYZE', '')        # per-block bias-curve log (jsonl, encode-only)
DECAF_QUANT = os.environ.get('DECAF_QUANT', '0') == '1'        # 4-bit packed store for retained grains

SEG_FRAMES = 4
PATCH_TOKENS = 49
QB = 7.0  # int4 symmetric range

# Set from the host at model load (text_config.num_hidden_layers).
N_LAYERS = None

_STORE = None


def get_store(cm):
    global _STORE
    if _STORE is None:
        _STORE = MultiGrainStore(cm)
    return _STORE


def reset_store():
    global _STORE
    _STORE = None


def quantize(t):
    """fp16 (kh, T, dh) -> packed uint8 + fp16 per-(head, channel) scale.
    Per-channel absmax int4 (KIVI-style layout); W2.2 measured flat sensitivity,
    so 4-bit is near-lossless at the metric level (TB6 answers are the gate)."""
    if not DECAF_QUANT:
        return ('fp16', t.contiguous().cpu())
    t = t.float()
    scale = (t.abs().amax(dim=1, keepdim=True) / QB).clamp(min=1e-8).half()  # (kh,1,dh)
    q = torch.round(t / scale.float()).clamp(-QB, QB).to(torch.int8)
    packed = (q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)
    return ('q4', packed.contiguous().cpu(), scale.cpu())


def dequant(entry):
    kind = entry[0]
    if kind == 'fp16':
        return entry[1]
    _, packed, scale = entry
    lo = (packed & 0xF).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    lo = torch.where(lo > QB, lo - 16, lo)
    hi = torch.where(hi > QB, hi - 16, hi)
    q = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return (q.float() * scale.float()).half()


class GrainScorer:
    """Online positional detrending of per-block attention curves (the write-time
    debias). Keeps running per-block-relative-position sums of the RAW curve; each
    new block is debiased against the running positional mean (estimated from past
    blocks only — no leakage), then the debiased curve feeds the keep-lists and the
    raw curve feeds the bias-curve statistics (C1 evidence)."""

    def __init__(self, block_size):
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.n = 0
        self.pos_sum = torch.zeros(self.block_size)
        self.pos_sq = torch.zeros(self.block_size)
        self.deb_pos_sum = torch.zeros(self.block_size)
        self.deb_pos_sq = torch.zeros(self.block_size)
        self.curves_raw = []   # per-block raw curves (196,) for the analysis payload
        self.curves_deb = []

    def detrend(self, curve):
        if self.n < 2:
            return curve.clone()
        mu = self.pos_sum / self.n
        return curve - mu

    def update(self, curve_raw, curve_deb):
        self.curves_raw.append(curve_raw.cpu())
        self.curves_deb.append(curve_deb.cpu())
        self.pos_sum += curve_raw.cpu()
        self.pos_sq += curve_raw.cpu() ** 2
        self.deb_pos_sum += curve_deb.cpu()
        self.deb_pos_sq += curve_deb.cpu() ** 2
        self.n += 1

    def bias_stats(self):
        """mean |attention residual vs position| before (raw) / after (debias)."""
        if self.n == 0:
            return {}
        raw = torch.stack(self.curves_raw)            # (B, 196)
        deb = torch.stack(self.curves_deb)
        raw_res = (raw - raw.mean(dim=0, keepdim=True)).abs().mean().item()
        deb_res = (deb - deb.mean(dim=0, keepdim=True)).abs().mean().item()
        raw_curve = raw.mean(dim=0)
        deb_curve = deb.mean(dim=0)
        return {
            'n_blocks': self.n,
            'mean_abs_residual_raw': raw_res,
            'mean_abs_residual_debiased': deb_res,
            'pos_curve_raw': [round(float(v), 6) for v in raw_curve],
            'pos_curve_debiased': [round(float(v), 6) for v in deb_curve],
        }


class FrameRecord:
    __slots__ = ('token_start', 'kv', 'attn', 'deb_attn', 'freq')

    def __init__(self, token_start):
        self.token_start = token_start
        self.kv = {}            # layer_idx -> (k_entry, v_entry)
        self.attn = None        # (196, n_layers) set on layer 0
        self.deb_attn = None
        self.freq = None        # (196, n_layers)


class SegmentRecord:
    __slots__ = ('seg_id', 'token_start', 'frames', 'keys', 'grain_imp', 'grains',
                 'retained', 'bytes_kv', 'bytes_keys')

    def __init__(self, seg_id, token_start):
        self.seg_id = seg_id
        self.token_start = token_start
        self.frames = []
        self.keys = {}          # layer_idx -> {'frame': (4, kh*dh), 'patch': (16, kh*dh), 'seg': (kh*dh)}
        self.grain_imp = None   # dict of per-grain importance arrays
        self.grains = []        # grain dicts: {type, local idx, frame, patch, tokens, imp, imp_attn, imp_freq}
        self.retained = []      # retained grain dicts
        self.bytes_kv = 0
        self.bytes_keys = 0


class MultiGrainStore:
    """Assembles frames into segments at offload time, scores grains (dual signal,
    optionally debiased), applies the retention policy (writetime = single ranked
    keep-list; deferred = union of per-signal keep-lists under the same token tax),
    and accounts bytes exactly like the stock store for C3."""

    def __init__(self, cm):
        self.cm = cm
        self.block_size = cm.block_size
        self.seg_tokens = SEG_FRAMES * cm.block_size
        self.scorer = GrainScorer(cm.block_size)
        self.frame_records = []
        self.frames_by_start = {}  # token_start -> FrameRecord (pre-assembly)
        self.segments = []
        self.n_scored_layers = 0
        self.commit_plan = None   # cached per-question commit (planned on layer 0)
        self.pending_level = 'coarse'
        self.plan_log = None
        self._key_cache = None    # (n_segments, device, per-layer stacked keys)
        self.keys_kh = None
        self.keys_dh = None

    # ---- write-time assembly -------------------------------------------------
    def add_frame_layer(self, layer_idx, k, v, attn_vec, deb_attn_vec, freq_vec, token_start):
        # Frames are keyed by absolute token_start: each layer's manager offloads
        # ALL of its blocks during its own forward pass, so arrival order is
        # layer-major, not block-major. Assembly is greedy: as soon as 4
        # consecutive frames (one segment) have full layer coverage, the segment
        # is scored and finalized.
        fr = self.frames_by_start.get(token_start)
        if fr is None:
            fr = FrameRecord(token_start)
            fr.attn = torch.zeros(self.block_size, N_LAYERS)
            fr.deb_attn = torch.zeros(self.block_size, N_LAYERS)
            fr.freq = torch.zeros(self.block_size, N_LAYERS)
            self.frames_by_start[token_start] = fr
        fr.kv[layer_idx] = (quantize(k), quantize(v))  # per-layer CPU entries, (kh, bs, dh)
        fr.attn[:, layer_idx] = attn_vec.cpu()
        fr.deb_attn[:, layer_idx] = deb_attn_vec.cpu()
        fr.freq[:, layer_idx] = freq_vec.cpu()
        while True:
            starts = sorted(self.frames_by_start)
            if len(starts) < SEG_FRAMES:
                break
            head = starts[:SEG_FRAMES]
            if any(head[f] != head[0] + f * self.block_size for f in range(SEG_FRAMES)):
                break
            if not all(len(self.frames_by_start[s].kv) == N_LAYERS for s in head):
                break
            frames = [self.frames_by_start.pop(s) for s in head]
            self._assemble_segment(frames)

    def _assemble_segment(self, frames):
        seg = SegmentRecord(len(self.segments), frames[0].token_start)
        seg.frames = frames
        L = N_LAYERS
        # per-layer grain content keys (pre-RoPE mean K, from the stored grain
        # bytes — honest: with DECAF_QUANT the ranking keys come from 4-bit)
        seg.keys = {}
        seg.grains = []
        for li in range(L):
            ks = [dequant(fr.kv[li][0]) for fr in frames]  # (kh, 196, dh) each
            kseg = torch.cat(ks, dim=1)                    # (kh, 784, dh)
            kh, _, dh = kseg.shape
            self.keys_kh, self.keys_dh = kh, dh
            fkeys = torch.stack([k.mean(dim=1) for k in ks])           # (4, kh, dh)
            pkeys = torch.stack([kseg[:, p * PATCH_TOKENS:(p + 1) * PATCH_TOKENS, :].mean(dim=1)
                                 for p in range(self.block_size // PATCH_TOKENS * SEG_FRAMES)])  # (16, kh, dh)
            seg.keys[li] = {
                'frame': fkeys.reshape(4, kh * dh).half().cpu(),
                'patch': pkeys.reshape(16, kh * dh).half().cpu(),
                'seg': kseg.mean(dim=1).reshape(kh * dh).half().cpu(),
            }
        # importance signals, aggregated over layers
        attn_t = torch.stack([fr.attn.mean(dim=1) for fr in frames])       # (4, 196)
        deb_t = torch.stack([fr.deb_attn.mean(dim=1) for fr in frames])    # (4, 196)
        freq_t = torch.stack([fr.freq.mean(dim=1) for fr in frames])       # (4, 196)
        nt = self.block_size // PATCH_TOKENS  # patches per frame
        for f in range(SEG_FRAMES):
            sl = slice(f * self.block_size, (f + 1) * self.block_size)
            seg.grains.append({'type': 'frame', 'frame': f, 'patch': -1,
                               'tokens': self.block_size,
                               'imp_attn': float(attn_t.reshape(-1)[sl].mean()),
                               'imp_attn_deb': float(deb_t.reshape(-1)[sl].mean()),
                               'imp_freq': float(freq_t.reshape(-1)[sl].mean())})
            for p in range(nt):
                ps = slice(f * self.block_size + p * PATCH_TOKENS,
                           f * self.block_size + (p + 1) * PATCH_TOKENS)
                seg.grains.append({'type': 'patch', 'frame': f, 'patch': p,
                                   'tokens': PATCH_TOKENS,
                                   'imp_attn': float(attn_t.reshape(-1)[ps].mean()),
                                   'imp_attn_deb': float(deb_t.reshape(-1)[ps].mean()),
                                   'imp_freq': float(freq_t.reshape(-1)[ps].mean())})
        # dual-signal z-normed importance, per grain type (no size artifacts)
        for sig in ('imp_attn', 'imp_attn_deb', 'imp_freq'):
            for gtype in ('frame', 'patch'):
                idx = [i for i, g in enumerate(seg.grains) if g['type'] == gtype]
                vals = torch.tensor([seg.grains[i][sig] for i in idx])
                z = (vals - vals.mean()) / (vals.std() + 1e-6)
                for j, i in enumerate(idx):
                    seg.grains[i][sig + '_z'] = float(z[j])
        for g in seg.grains:
            g['imp_raw'] = g['imp_attn_z'] + g['imp_freq_z']        # entangled scoring
            g['imp_deb'] = g['imp_attn_deb_z'] + g['imp_freq_z']    # debiased scoring
        # retention under the token tax. The storage granularity is the whole
        # frame (patches ride on their frame's bytes), so the budget is charged
        # in realized frame bytes: a grain of an already-kept frame is free.
        budget = DECAF_B_TAX * self.seg_tokens
        use_sig = 'imp_deb' if DECAF_B_BIAS == 'debias' else 'imp_raw'

        def cost(frames_kept, g):
            return 0 if g['frame'] in frames_kept else self.block_size

        if DECAF_B_COMMIT == 'writetime':
            ranked = sorted(seg.grains, key=lambda g: -g[use_sig])
            keep_frames, used = set(), 0
            for g in ranked:
                c = cost(keep_frames, g)
                if used + c > budget + 1e-6:
                    continue
                keep_frames.add(g['frame'])
                used += c
            seg.retained = [g for g in ranked if g['frame'] in keep_frames]
        else:  # deferred: hedged superset = union of per-signal keep-lists, B/2 each
            kept_idx = set()
            keep_frames = set()
            for sig in ('imp_attn_deb' if DECAF_B_BIAS == 'debias' else 'imp_attn', 'imp_freq'):
                ranked = sorted(range(len(seg.grains)), key=lambda i: -seg.grains[i][sig])
                fr, u = set(), 0
                for i in ranked:
                    g = seg.grains[i]
                    c = cost(fr, g)
                    if i in kept_idx or u + c > budget / 2 + 1e-6:
                        continue
                    kept_idx.add(i)
                    fr.add(g['frame'])
                    u += c
                keep_frames |= fr
            seg.retained = [seg.grains[i] for i in sorted(kept_idx)]
        # free K/V of frames with no retained grain; realized store granularity
        # is the whole frame (patches ride on their frame's bytes)
        keep_frames = sorted(set(g['frame'] for g in seg.retained))
        drop = set(range(SEG_FRAMES)) - set(keep_frames)
        if drop:
            for li in range(L):
                for f in drop:
                    seg.frames[f].kv.pop(li, None)
        # bytes accounting (K+V of retained frames, per layer)
        per_tok = 0
        if keep_frames:
            e = seg.frames[keep_frames[0]].kv[0][0]
            if e[0] == 'fp16':
                per_tok = e[1].numel() / self.block_size * e[1].element_size() * 2  # K and V
            else:
                per_tok = e[1].numel() / self.block_size * e[1].element_size() * 2  # packed: already half-bytes
        seg.bytes_kv = int(len(keep_frames) * self.block_size * per_tok * L)
        seg.bytes_keys = int((4 + 16 + 1) * L * seg.keys[0]['seg'].numel() * 2)
        self.segments.append(seg)

    # ---- query-time commitment ------------------------------------------------
    def _gpu_key_cache(self, cms, device):
        """All grain content keys stacked per layer, uploaded once per segment-
        count (not per question): frame (S,4,kh,dh), patch (S,16,kh,dh)."""
        S = len(self.segments)
        if self._key_cache is not None and self._key_cache[0] == S \
                and self._key_cache[1] == device:
            return self._key_cache[2]
        kh = self.keys_kh
        dh = self.keys_dh
        keys = []
        for li in range(len(cms)):
            fr = torch.stack([s.keys[li]['frame'] for s in self.segments])
            pa = torch.stack([s.keys[li]['patch'] for s in self.segments])
            keys.append((
                fr.view(S, 4, kh, dh).to(device),
                pa.view(S, 16, kh, dh).to(device),
            ))
        self._key_cache = (S, device, keys)
        return keys

    def grain_similarities(self, cms, q_layers):
        """q_layers: list of per-layer question queries (heads, len_q, dh) on GPU.
        Batched: one einsum per layer over ALL segments/grains (the naive
        per-segment loop was O(S*L) small H2D copies per question — minutes at
        S=450). Returns per-segment dicts with per-grain query similarity
        (mean over full heads, stock GQA-expansion convention)."""
        L = len(cms)
        dev = q_layers[0].device
        kh, dh = self.keys_kh, self.keys_dh
        keys = self._gpu_key_cache(cms, dev)
        S = len(self.segments)
        fr_sims = torch.zeros(L, S, 4)
        pa_sims = torch.zeros(L, S, 16)
        for li, cm in enumerate(cms):
            ng = cm.num_heads // kh
            q = q_layers[li].float().mean(dim=1).view(kh, ng, dh)  # (kh, ng, dh)
            kf, kp = keys[li]
            sf = torch.einsum('ksnd,kgd->ksng', kf.float().permute(2, 0, 1, 3), q)
            sp = torch.einsum('ksnd,kgd->ksng', kp.float().permute(2, 0, 1, 3), q)
            fr_sims[li] = sf.mean(dim=(0, 3)).cpu()
            pa_sims[li] = sp.mean(dim=(0, 3)).cpu()
        fr_mean = fr_sims.mean(dim=0)  # (S, 4)
        pa_mean = pa_sims.mean(dim=0)  # (S, 16)
        seg_sims = []
        for si, seg in enumerate(self.segments):
            if not seg.retained:
                continue
            gsims = {}
            for g in seg.retained:
                if g['type'] == 'frame':
                    gsims[('frame', g['frame'], -1)] = float(fr_mean[si, g['frame']])
                else:
                    gsims[('patch', g['frame'], g['patch'])] = float(
                        pa_mean[si, g['frame'] * (self.block_size // PATCH_TOKENS) + g['patch']])
            seg_sims.append((seg, gsims))
        return seg_sims

    @staticmethod
    def _seg_score(seg, gsims, gamma):
        frames = [s for (t, f, p), s in gsims.items() if t == 'frame']
        patches = [s for (t, f, p), s in gsims.items() if t == 'patch']
        base = (sum(frames) / len(frames)) if frames else (
            sum(patches) / len(patches) if patches else -1e9)
        allv = sorted(gsims.values(), reverse=True)
        cons = sum(allv[:max(1, len(allv) // 2)]) / max(1, len(allv) // 2)
        return base + gamma * cons, base, cons

    def plan(self, cms, q_layers, budget_slots, level='coarse'):
        """Returns (commit, log): commit = list of (seg, grain) in slot-fill order,
        each grain occupying tokens/196 slots at a packed position. Slot layout is
        contiguous from n_init (rank-remap convention, corrected Component A)."""
        seg_sims = self.grain_similarities(cms, q_layers)
        scored = []
        for seg, gsims in seg_sims:
            sc, base, cons = self._seg_score(seg, gsims, DECAF_B_GAMMA)
            scored.append((sc, base, cons, seg, gsims))
        scored.sort(key=lambda x: -x[0])
        commit, slots = [], 0
        max_slots = budget_slots
        policy = DECAF_B_POLICY

        if policy == 'seg' or (policy == 'default' and level == 'coarse'):
            # coarse: whole-segment views of the top-ranked segments; patches
            # pack into their frame's slot, lone patch groups take their own slot
            for sc, base, cons, seg, gsims in scored:
                fr = sorted((g for g in seg.retained if g['type'] == 'frame'),
                            key=lambda g: -gsims.get(('frame', g['frame'], -1), -1e9))
                pa = sorted((g for g in seg.retained if g['type'] == 'patch'),
                            key=lambda g: -gsims.get(('patch', g['frame'], g['patch']), -1e9))
                if slots + len(fr) > max_slots and policy == 'default':
                    break
                for g in fr:
                    commit.append((seg, g))
                slots += len(fr)
                taken = {g['frame'] for g in fr}
                for g in pa:
                    if g['frame'] in taken:
                        commit.append((seg, g, 'pack'))
                    elif slots + 1 <= max_slots:
                        commit.append((seg, g))
                        slots += 1
                        taken.add(g['frame'])
        elif policy == 'frame':
            allf = []
            for sc, base, cons, seg, gsims in scored:
                for g in seg.retained:
                    if g['type'] == 'frame':
                        allf.append((gsims[('frame', g['frame'], -1)], seg, g))
            allf.sort(key=lambda x: -x[0])
            for s, seg, g in allf[:max_slots]:
                commit.append((seg, g))
                slots += 1
        elif policy == 'patch':
            allp = []
            for sc, base, cons, seg, gsims in scored:
                for g in seg.retained:
                    if g['type'] == 'patch':
                        allp.append((gsims[('patch', g['frame'], g['patch'])], seg, g))
            allp.sort(key=lambda x: -x[0])
            # pack up to 4 same-frame patches per slot
            groups = {}
            for s, seg, g in allp:
                groups.setdefault((id(seg), g['frame']), []).append((s, seg, g))
            for key, lst in sorted(groups.items(), key=lambda kv: -max(x[0] for x in kv[1])):
                if slots + 1 > max_slots:
                    break
                for s, seg, g in lst[:4]:
                    commit.append((seg, g, 'pack'))
                slots += 1
        else:  # deferred fine level / default pass2: global grain ranking,
            # coarse-first within equal similarity is implicit (frames listed
            # before patches per segment score order in `scored`)
            allg = []
            for sc, base, cons, seg, gsims in scored:
                for g in seg.retained:
                    allg.append((gsims[(g['type'], g['frame'], g['patch'])], seg, g))
            allg.sort(key=lambda x: -x[0])
            frames_taken = set()
            patch_groups = {}
            for s, seg, g in allg:
                key = (id(seg), g['frame'])
                if g['type'] == 'frame':
                    if slots + 1 <= max_slots:
                        commit.append((seg, g))
                        slots += 1
                        frames_taken.add(key)
                else:
                    if key in frames_taken or patch_groups.get(key, 0) >= 4:
                        continue
                    if patch_groups.get(key, 0) == 0:
                        if slots + 1 > max_slots:
                            continue
                        slots += 1
                    commit.append((seg, g, 'pack'))
                    patch_groups[key] = patch_groups.get(key, 0) + 1
        log = {
            'policy': policy, 'level': level,
            'seg_scores': [[int(seg.seg_id), round(float(sc), 4), round(float(base), 4),
                            round(float(cons), 4)] for sc, base, cons, seg, gsims in scored[:32]],
            'n_committed_grains': len(commit),
            'committed_slots': slots,
            'commit': [[int(seg.seg_id), g['type'], g['frame'], g['patch'],
                        'pack' if len(c) > 2 else 'slot'] for c in commit for seg, g, *rest in [c]],
        }
        return commit, log


    def store_bytes(self):
        """MuKV-style total-byte accounting of the hedged store (K/V + grain keys),
        all layers included."""
        kv = sum(s.bytes_kv for s in self.segments)
        keys = sum(s.bytes_keys for s in self.segments)
        return {'store_bytes_kv': int(kv), 'store_bytes_keys': int(keys),
                'store_bytes_total': int(kv + keys), 'n_segments': len(self.segments)}


def write_commit_log(rec):
    if DECAF_B_LOG:
        with open(DECAF_B_LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')


def write_analyze_log(rec):
    if DECAF_B_ANALYZE:
        with open(DECAF_B_ANALYZE, 'a') as f:
            f.write(json.dumps(rec) + '\n')
