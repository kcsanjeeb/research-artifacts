"""HolmesVAU-2B on XD escalated sets (band s0, top-K s0) -- E0.6/E0.7 protocol
on XD-Violence: same prompt, same fp16/eager config, same score extraction
(P(Yes) first-token logprob + hard parse + ATS). Output per-shard JSONL.
"""
import argparse
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
import common  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
SEED = 0
PROMPT = ("Does this video contain an anomaly? Answer Yes or No first, "
          "then explain briefly.")


def run_worker(gate, shard, n_shards, batch):
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.join(E0, 'src'))
    from capacity import load_holmes, CLIP_SPEC
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import build_transform
    from holmesvau.ATS.anomaly_scorer import URDMU

    common.set_seed(SEED)
    device = torch.device("cuda:0")
    model, tok, load_s = load_holmes(common.HOLMES_MODEL_PATH, device)
    scorer = URDMU().to(device)
    scorer.load_state_dict(torch.load(
        os.path.join(common.HOLMES_DIR, 'holmesvau', 'ATS', 'anomaly_scorer.pth'),
        map_location=device))
    scorer.eval()
    transform = build_transform(input_size=448)
    yn = {w: tok.encode(w, add_special_tokens=False)[0]
          for w in ("Yes", "No", "yes", "no")}
    sys.path.insert(0, common.HOLMES_MODEL_PATH)
    from conversation import get_conv_template

    d = np.load(os.path.join(OUT, f'xd_escalated_{gate}.npz'), allow_pickle=True)
    vis, sis = d['vi'], d['si']
    cdir = os.path.join(E0, 'data', f'xd_clip_cache_{gate}')
    mine = list(range(shard, len(vis), n_shards))
    print(f"[xd-vlm {gate} shard{shard}] {len(mine)} clips", flush=True)

    def load_clip(k):
        return torch.stack([transform(Image.open(
            os.path.join(cdir, f"{k:05d}_f{j:02d}.jpg")).convert('RGB'))
            for j in range(12)])

    out_path = os.path.join(OUT, f"xd_vlm_scores_{gate}_shard{shard}.jsonl")
    with open(out_path, 'w') as fout:
        for b0 in range(0, len(mine), batch):
            ks = mine[b0:b0 + batch]
            pv = torch.cat([load_clip(k) for k in ks]).to(torch.float16).to(device)
            with torch.no_grad():
                vit = model.vision_model(pixel_values=pv, output_hidden_states=False,
                                         return_dict=True).last_hidden_state
                cls = vit[:, 0, :].to(torch.float32).unsqueeze(0)
                ats = scorer(cls)['anomaly_scores'][0]
                ats = ats.detach().cpu().numpy().reshape(len(ks), 12).mean(axis=1)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n'
                                    for i in range(CLIP_SPEC["frames"])])
            questions = [video_prefix + PROMPT for _ in ks]
            queries, pos = [], 0
            IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
            model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CTX)
            npl = [1] * pv.shape[0]
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
                max_new_tokens=64, do_sample=False,
                eos_token_id=tok.convert_tokens_to_ids(template.sep),
                output_scores=True, return_dict_in_generate=True)
            first_logits = gen_out.scores[0]
            texts = tok.batch_decode(gen_out.sequences, skip_special_tokens=True)
            for i, k in enumerate(ks):
                logits = first_logits[i].float()
                yes = torch.logsumexp(logits[[yn["Yes"], yn["yes"]]], dim=0)
                no = torch.logsumexp(logits[[yn["No"], yn["no"]]], dim=0)
                p_yes = torch.softmax(torch.stack([no, yes]), dim=0)[1].item()
                text = texts[i].split(template.sep)[0].strip()
                fw = text.split()[0].strip('.,!').lower() if text.split() else ''
                fout.write(json.dumps({
                    'k': int(k), 'vi': int(vis[k]), 'si': int(sis[k]),
                    'score_b': p_yes, 'score_c': 1 if fw == 'yes' else 0,
                    'score_a': float(ats[i]), 'text': text[:400]}) + '\n')
            fout.flush()
            if (b0 + len(ks)) % 100 < batch:
                print(f"[xd-vlm {gate} shard{shard}] {b0+len(ks)}/{len(mine)}",
                      flush=True)
    print(f"[xd-vlm {gate} shard{shard}] DONE", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run_worker(args.gate, args.shard, args.n_shards, args.batch)
