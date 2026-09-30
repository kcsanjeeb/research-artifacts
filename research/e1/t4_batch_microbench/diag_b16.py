import sys, glob, json
import torch

sys.path.insert(0, "/home/san/e1/t4_batch_microbench")
import v100_patches
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

p = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
v100_patches.apply_patches(model)
proc = AutoProcessor.from_pretrained(p)
proc.tokenizer.padding_side = "left"

SCHEMA = open("/dev/null").read() if False else (
    "These 8 frames are sampled uniformly from a single surveillance video "
    "segment. Describe only what is directly visible in the frames. Respond "
    "with a single strict JSON object and no other text, with exactly these "
    'keys: "actors", "actions", "objects", "location_cues", "summary".')
fs = sorted(glob.glob("/home/san/e1/narration_spike/frames/e*/f*.jpg"))[:8]
imgs = [Image.open(f).convert("RGB").resize((448, 336)) for f in fs]
bs = 16
all_imgs, texts = [], []
for _ in range(bs):
    all_imgs.extend(imgs)
    content = [{"type": "image"} for _ in range(8)] + [{"type": "text", "text": SCHEMA}]
    texts.append(proc.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
inputs = proc(text=texts, images=all_imgs, padding=True, return_tensors="pt").to("cuda")
pl = inputs["input_ids"].shape[1]
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                         temperature=0.2, top_p=0.9)
eos = proc.tokenizer.eos_token_id
print("prompt_len", pl, "eos_id", eos)
for i in range(bs):
    row = out[i]
    gen = row[pl:]
    n_eos = (gen == eos).sum().item()
    # true length = first eos (if any)
    idx = (gen == eos).nonzero()
    true_len = idx[0].item() if len(idx) else len(gen)
    txt = proc.decode(gen[:min(true_len + 1, len(gen))], skip_special_tokens=True)
    print("row %2d gen_len=%3d true_len=%3d n_eos=%d tail=%r" %
          (i, len(gen), true_len, n_eos, txt[-70:]))
