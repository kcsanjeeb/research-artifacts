import sys, glob, time
import torch

sys.path.insert(0, "/home/san/e1/t4_batch_microbench")
import v100_patches
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

bs = int(sys.argv[1])
gen = int(sys.argv[2]) if len(sys.argv) > 2 else 64
nf = int(sys.argv[3]) if len(sys.argv) > 3 else 8
padding = sys.argv[4] != "nopad" if len(sys.argv) > 4 else True

p = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
v100_patches.apply_patches(model)
proc = AutoProcessor.from_pretrained(p)
if padding:
    proc.tokenizer.padding_side = "left"

fs = sorted(glob.glob("/home/san/e1/narration_spike/frames/e0000/f*.jpg"))
sel = fs[:nf] if nf == 8 else [fs[0], fs[2], fs[4], fs[6]]
def mk():
    return [Image.open(f).convert("RGB").resize((448, 336)) for f in sel]
instr = "Describe these frames briefly."
imgs, texts = [], []
for _ in range(bs):
    imgs.extend(mk())
    content = [{"type": "image"} for _ in range(nf)] + [{"type": "text", "text": instr}]
    texts.append(proc.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
inputs = proc(text=texts, images=imgs, padding=padding, return_tensors="pt").to("cuda")
seq = inputs["input_ids"].shape[1]
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
t0 = time.time()
try:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=gen, do_sample=False)
    torch.cuda.synchronize()
    print("RESULT bs=%d nf=%d pad=%s seq=%d gen=%d gen_s=%.2f peak_mb=%.0f out_len=%d"
          % (bs, nf, padding, seq, gen, time.time() - t0,
             torch.cuda.max_memory_allocated() / 2**20, out.shape[1]), flush=True)
except torch.cuda.OutOfMemoryError:
    print("RESULT bs=%d nf=%d OOM" % (bs, nf), flush=True)
