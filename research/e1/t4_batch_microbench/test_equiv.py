import sys, glob, time
import torch

sys.path.insert(0, "/home/san/e1/t4_batch_microbench")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

p = "Qwen/Qwen2.5-VL-7B-Instruct"
proc = AutoProcessor.from_pretrained(p)

fs = sorted(glob.glob("/home/san/e1/narration_spike/frames/e0000/f*.jpg"))
imgs = [Image.open(f).convert("RGB").resize((448, 336)) for f in fs]
instr = ("These 8 frames are sampled uniformly from a single surveillance video "
         "segment. Describe only what is directly visible. Respond with a single "
         "strict JSON object with keys actors, actions, objects, location_cues, summary.")
content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": instr}]
text = proc.apply_chat_template([{"role": "user", "content": content}],
                                tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")

# --- unpatched ---
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
with torch.no_grad():
    out_orig = model(**inputs)
    gen_orig = model.generate(**inputs, max_new_tokens=64, do_sample=False)
logits_orig = out_orig.logits[0, -1].float().cpu()
del model
torch.cuda.empty_cache()

# --- patched ---
import v100_patches
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
v100_patches.apply_patches(model)
with torch.no_grad():
    t0 = time.time()
    out_patch = model(**inputs)
    fwd_s = time.time() - t0
    gen_patch = model.generate(**inputs, max_new_tokens=64, do_sample=False)
logits_patch = out_patch.logits[0, -1].float().cpu()

diff = (logits_orig - logits_patch).abs().max().item()
print("fwd patched: %.2fs, logits shape %s vs %s" %
      (fwd_s, tuple(out_patch.logits.shape), tuple(out_orig.logits.shape)))
print("max |logits diff| = %.5f" % diff)
same = (gen_orig[0, inputs["input_ids"].shape[1]:]
        == gen_patch[0, inputs["input_ids"].shape[1]:]).all().item()
print("greedy 64-token outputs identical:", bool(same))
print("patched out:", proc.decode(gen_patch[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[:120])
