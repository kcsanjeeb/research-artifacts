import time, torch, glob, sys
import torch.backends.cuda as bc

# force mem-efficient sdpa (handles attn_mask in O(N)); math fallback is the O(N^2) killer
bc.enable_flash_sdp(False)
bc.enable_math_sdp(False)
bc.enable_mem_efficient_sdp(True)
print("sdpa flags: flash=%s math=%s mem_eff=%s" % (bc.flash_sdp_enabled(), bc.math_sdp_enabled(), bc.mem_efficient_sdp_enabled()), flush=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

bs = int(sys.argv[1])
gen = int(sys.argv[2]) if len(sys.argv) > 2 else 64

p = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
proc = AutoProcessor.from_pretrained(p)
proc.tokenizer.padding_side = "left"
fs = sorted(glob.glob("/home/san/e1/narration_spike/frames/e0000/f*.jpg"))
def mk():
    return [Image.open(f).convert("RGB").resize((448, 336)) for f in fs]
instr = "Describe these frames briefly."
imgs, texts = [], []
for _ in range(bs):
    imgs.extend(mk())
    content = [{"type": "image"} for _ in range(8)] + [{"type": "text", "text": instr}]
    texts.append(proc.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
inputs = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
seq = inputs["input_ids"].shape[1]
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
t0 = time.time()
try:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=gen, do_sample=False)
    torch.cuda.synchronize()
    print("RESULT bs=%d seq=%d gen=%d gen_s=%.2f peak_mb=%.0f"
          % (bs, seq, gen, time.time() - t0, torch.cuda.max_memory_allocated() / 2**20), flush=True)
except RuntimeError as e:
    print("RESULT bs=%d RuntimeError: %s" % (bs, str(e)[:200]), flush=True)
except torch.cuda.OutOfMemoryError:
    print("RESULT bs=%d OOM" % bs, flush=True)
