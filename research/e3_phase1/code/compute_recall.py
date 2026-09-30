"""DECAF Component A: retrieval-recall measurement against the ego4d_oe temporal oracle.

Oracle definition (primary, STRICT): for a question with temporal window
[start_time, end_time] (seconds) in ego4d_oe.json, sampled at 0.5 fps, the
relevant block set is
    O = { b : start_time*0.5 <= b < end_time*0.5 }  (block b == sampled frame b)
restricted to blocks encoded at query time (b < frames_encoded; the stream has
only seen [0, end_time*0.5) frames, so this restriction is usually vacuous).
Recall@retrieve_size per question = |retrieved_layer0 ∩ O| / |O|; questions with
|O| == 0 are excluded and counted.

Relaxed variant (RELAXED_30S): the window is expanded by +-30 s on each side
(+-15 sampled frames at 0.5 fps) before intersecting with encoded blocks.

Metrics reported per jsonl log: macro/mean per-question recall, micro recall
(sum intersections / sum |O|), n_questions counted, n_skipped (empty oracle).

Usage: compute_recall.py <anno.json> <retrieval.jsonl> <out.json> [--layer L]
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('anno')
    ap.add_argument('jsonl')
    ap.add_argument('out')
    ap.add_argument('--layer', type=int, default=0, help='which layer\'s indices (default 0, as logged by W2.5 T1)')
    ap.add_argument('--sample_fps', type=float, default=0.5)
    args = ap.parse_args()

    anno = json.load(open(args.anno))
    win = {}
    for v in anno:
        for i, c in enumerate(v['conversations']):
            win[(v['video_id'], i)] = (c['start_time'], c['end_time'])
    q_order = {v['video_id']: [c['question'] for c in v['conversations']] for v in anno}

    per_q = []
    with open(args.jsonl) as f:
        for line in f:
            per_q.append(json.loads(line))

    def recall_for(rec, relaxed):
        vid = rec['video_id']
        # question index within the video's conversation list
        try:
            qi = q_order[vid].index(rec['question'])
        except ValueError:
            return None
        start, end = win[(vid, qi)]
        margin = 30.0 if relaxed else 0.0
        lo = (start - margin) * args.sample_fps
        hi = (end + margin) * args.sample_fps
        frames_encoded = (rec['encoded_length'] - 13) / 196.0
        oracle = {b for b in range(0, int(hi)) if b >= lo and 0 <= b < frames_encoded}
        if not oracle:
            return ('SKIP', 0, 0)
        idx = rec['retrieved_layer0']
        hit = len(set(idx) & oracle)
        return ('OK', hit, len(oracle))

    out = {'jsonl': args.jsonl, 'layer': args.layer}
    for tag, relaxed in [('strict', False), ('relaxed_pm30s', True)]:
        hits, tot, n, skipped = 0, 0, 0, 0
        per_q_recall = []
        for rec in per_q:
            r = recall_for(rec, relaxed)
            if r is None:
                continue
            status, hit, osize = r
            if status == 'SKIP':
                skipped += 1
                continue
            n += 1
            hits += hit
            tot += osize
            per_q_recall.append(hit / osize)
        out[tag] = {
            'n_questions': n,
            'n_skipped_empty_oracle': skipped,
            'micro_recall_pct': round(100.0 * hits / tot, 2) if tot else None,
            'macro_recall_pct': round(100.0 * sum(per_q_recall) / len(per_q_recall), 2) if per_q_recall else None,
        }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
