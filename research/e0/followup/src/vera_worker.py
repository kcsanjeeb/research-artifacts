"""VERA (InternVL2-8B + verbalized-learning prompt) on two E0.7 cells:
band s0 and top-K s0. E0.8 section 6 -- answers the "you used one small VLM"
rebuttal. Protocol faithful to VERA's generate_initial_score.py:
  - prompt: VERA_learner_instruct.txt with $Data -> Frame1..Frame12 <image>,
    and the 5 guiding questions (new_qs) inserted before the position_flag
    line, exactly as upstream
  - score_c: VERA's own parse (split 'Output'; '0' present -> 0 else 1)
  - score_b: P("1") at the model's chosen label position, logprob from
    output_scores at the first {0,1}-token generated after the 'Output'
    section -- one generation per clip, no re-prompting
Clips: the same frozen lists and the same 12-frame 448x448 JPEG caches as
E0.7 (VERA natively uses 8 frames over a 10s window; we hold the grid's
clip spec constant across models and note the difference).
Model: OpenGVLab/InternVL2-8B, fp16 + eager (sm_70 patches, as HolmesVAU).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
import common  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
VERA_DIR = os.path.join(common.THIRD_PARTY, 'VERA')
MODEL_PATH = os.path.join(common.DATA_DIR, 'InternVL2-8B')
SEED = 0
SNIP = 16

NEW_QS = ("1. Are there any people in the video who are not in their typical "
          "positions or engaging in activities that are not consistent with "
          "their usual behavior?\n"
          "2. Are there any vehicles in the video that are not in their "
          "typical positions or being used in a way that is not consistent "
          "with their usual function?\n"
          "3. Are there any objects in the video that are not in their "
          "typical positions or being used in a way that is not consistent "
          "with their usual function?\n"
          "4. Is there any visible damage or unusual movement in the video "
          "that indicates an anomaly?\n"
          "5. Are there any unusual sounds or noises in the video that "
          "suggest an anomaly?\n")
POSITION_FLAG = ("Based on the analysis above, please conclude your answer "
                 "to 'Is there any anomaly in the video?' in 'Yes, there is "
                 "an anomaly' or 'No, there is no anomaly'.")


def vera_prompt(n_frames):
    instruct = open(os.path.join(VERA_DIR, 'VERA_learner_instruct.txt')).read()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(n_frames)])
    q = instruct.replace('$Data', video_prefix)
    i = q.find(POSITION_FLAG)
    if i != -1:
        q = q[:i] + NEW_QS + q[i:]
    return q


def cache_dir(gate):
    return (os.path.join(E0, 'data', 'clip_cache') if gate == 'band'
            else os.path.join(E0, 'data', 'clip_cache_topk_s0'))


def list_path(gate):
    return os.path.join(OUT, 'escalated_clips.npz' if gate == 'band'
                        else 'escalated_topk.npz')


def run_worker(gate, shard, n_shards, batch):
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.join(E0, 'src'))
    from capacity import load_holmes  # generic InternVLChatModel loader (sm_70)
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import build_transform

    common.set_seed(SEED)
    device = torch.device("cuda:0")
    model, tok, load_s = load_holmes(MODEL_PATH, device)
    print(f"[vera {gate} shard{shard}] loaded in {load_s:.1f}s", flush=True)
    transform = build_transform(input_size=448)
    sys.path.insert(0, MODEL_PATH)
    from conversation import get_conv_template

    id0 = tok.encode("0", add_special_tokens=False)[0]
    id1 = tok.encode("1", add_special_tokens=False)[0]
    print(f"[vera {gate} shard{shard}] label ids: 0->{id0} 1->{id1}", flush=True)

    d = np.load(list_path(gate), allow_pickle=True)
    vis, sis = d['vi'], d['si']
    cdir = cache_dir(gate)
    mine = list(range(shard, len(vis), n_shards))
    print(f"[vera {gate} shard{shard}] {len(mine)} clips", flush=True)

    def load_clip(k):
        return torch.stack([transform(Image.open(
            os.path.join(cdir, f"{k:05d}_f{j:02d}.jpg")).convert('RGB'))
            for j in range(12)])

    out_path = os.path.join(OUT, f"vera_scores_{gate}_s0_shard{shard}.jsonl")
    with open(out_path, 'w') as fout:
        for b0 in range(0, len(mine), batch):
            ks = mine[b0:b0 + batch]
            pv = torch.cat([load_clip(k) for k in ks]).to(torch.float16).to(device)
            npl = [1] * pv.shape[0]
            questions = [vera_prompt(12) for _ in ks]
            queries, pos = [], 0
            IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
            model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CTX)
            for q in questions:
                template = get_conv_template(model.template)
                template.system_message = model.system_message
                template.append_message(template.roles[0], q)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()
                for _ in range(query.count('<image>')):
                    image_tokens = (IMG_START + IMG_CTX * model.num_image_token
                                    * npl[pos] + IMG_END)
                    pos += 1
                    query = query.replace('<image>', image_tokens, 1)
                queries.append(query)
            tok.padding_side = 'left'
            inputs = tok(queries, return_tensors='pt', padding=True)
            gen_out = model.generate(
                pixel_values=pv,
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_new_tokens=128, do_sample=False,
                eos_token_id=tok.convert_tokens_to_ids(template.sep),
                output_scores=True, return_dict_in_generate=True)
            texts = tok.batch_decode(gen_out.sequences, skip_special_tokens=True)
            seqs = gen_out.sequences
            for i, k in enumerate(ks):
                text = texts[i].split(template.sep)[0].strip()
                # score_c: VERA's own parse
                parts = text.split('Output')
                score_c = 0 if (len(parts) > 1 and '0' in parts[-1]) else 1
                # score_b: P(1) at the first {0,1} token after the 'Output'
                # marker in the generated sequence
                prompt_len = seqs.shape[1] - len(gen_out.scores)
                label_pos, p1 = None, None
                for t in range(len(gen_out.scores)):
                    tok_id = seqs[i, prompt_len + t].item()
                    if tok_id in (id0, id1):
                        # ensure we're past the Output section in the text
                        prefix_text = tok.decode(
                            seqs[i, prompt_len:prompt_len + t].tolist(),
                            skip_special_tokens=True)
                        if 'Output' in prefix_text:
                            logits = gen_out.scores[t][i].float()
                            p1 = torch.softmax(logits[[id0, id1]], dim=0)[1].item()
                            label_pos = t
                            break
                fout.write(json.dumps({
                    'k': int(k), 'vi': int(vis[k]), 'si': int(sis[k]),
                    'score_b': p1, 'score_c': score_c,
                    'label_pos': label_pos, 'text': text[:400]}) + '\n')
            fout.flush()
            if (b0 + len(ks)) % 50 < batch:
                print(f"[vera {gate} shard{shard}] {b0+len(ks)}/{len(mine)}",
                      flush=True)
    print(f"[vera {gate} shard{shard}] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=4)
    ap.add_argument('--batch', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run_worker(args.gate, args.shard, args.n_shards, args.batch)


if __name__ == '__main__':
    main()
