"""Self-extract XD-Violence test CLIP features with the verified E0.8 spec:
CLIP ViT-B/16 (weights from model_xd.pth clipmodel.* keys), standard CLIP
transform (resize-short-side 224 bicubic + center crop + CLIP normalize),
one forward per frame, raw (unnormalized) outputs, mean-pooled per 16-frame
snippet, saved as fp16 npy per video. One worker per GPU; skips existing.
"""
import argparse
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
import common  # noqa: E402

OUT_DIR = os.path.join(E0, 'data', 'XDTestClipFeatures_self')
VIDS = os.path.join(E0, 'data', 'xd_test_videos')
SNIP = 16


def run_worker(shard, n_shards):
    import torch
    sys.path.insert(0, common.VADCLIP_SRC)
    from clip.clip import build_model, _transform
    from decord import VideoReader, cpu
    from PIL import Image

    state = torch.load(os.path.join(E0, 'data', 'model_xd.pth'),
                       map_location='cpu')
    clip_state = {k[len('clipmodel.'):]: v for k, v in state.items()
                  if k.startswith('clipmodel.')}
    model = build_model(clip_state).cuda().eval()
    tf = _transform(224)

    import pandas as pd
    df = pd.read_csv(os.path.join(common.VADCLIP_LIST, 'xd_CLIP_rgbtest.csv'))
    vids = [os.path.basename(p).replace('__0.npy', '') for p in df['path']]
    mine = [v for i, v in enumerate(vids) if i % n_shards == shard]
    todo = [v for v in mine
            if not os.path.exists(os.path.join(OUT_DIR, v + '.npy'))]
    print(f"[xtract {shard}] {len(todo)} videos", flush=True)

    for vi, v in enumerate(todo):
        vf = os.path.join(VIDS, v + '.mp4')
        try:
            vr = VideoReader(vf, ctx=cpu(0), num_threads=1)
        except Exception:
            print(f"[xtract {shard}] SKIP {v} (unreadable)", flush=True)
            continue
        feats = []
        buf = []
        for fi in range(len(vr)):
            img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
            buf.append(tf(img))
            if len(buf) == 64:
                x = torch.stack(buf).cuda()
                if next(model.parameters()).dtype == torch.float16:
                    x = x.half()
                with torch.no_grad():
                    e = model.encode_image(x)
                feats.append(e.float().cpu().numpy())
                buf = []
        if buf:
            x = torch.stack(buf).cuda()
            if next(model.parameters()).dtype == torch.float16:
                x = x.half()
            with torch.no_grad():
                e = model.encode_image(x)
            feats.append(e.float().cpu().numpy())
        feats = np.concatenate(feats)
        n_snip = len(feats) // SNIP
        snip = feats[:n_snip * SNIP].reshape(n_snip, SNIP, -1).mean(axis=1)
        np.save(os.path.join(OUT_DIR, v + '.npy'), snip.astype(np.float16))
        if vi % 50 == 0:
            print(f"[xtract {shard}] {vi}/{len(todo)}", flush=True)
    print(f"[xtract {shard}] DONE", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    run_worker(args.shard, args.n_shards)
