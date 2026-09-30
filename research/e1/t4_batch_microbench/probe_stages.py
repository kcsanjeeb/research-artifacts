import time, torch, glob, sys
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 8

def mb():
    return torch.cuda.memory_allocated() / 2**20

p = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
proc = AutoProcessor.from_pretrained(p)
proc.tokenizer.padding_side = "left"
print("after load: %.0f MB" % mb(), flush=True)

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
inputs = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
print("after proc: %.0f MB (seq=%d, n_img=%d)" % (mb(), inputs["input_ids"].shape[1], inputs["image_grid_thw"].shape[0]), flush=True)

torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    out = model(**inputs, use_cache=True)
torch.cuda.synchronize()
print("after 1 forward: %.0f MB (%.1fs), logits shape %s" % (mb(), time.time() - t0, tuple(out.logits.shape)), flush=True)
del out
torch.cuda.synchronize()
print("after del logits: %.0f MB" % mb(), flush=True)
