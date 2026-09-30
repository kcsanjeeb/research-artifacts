import time, torch, gc, glob
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

p = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    p, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
proc = AutoProcessor.from_pretrained(p)
proc.tokenizer.padding_side = "left"
fs = sorted(glob.glob("/home/san/e1/narration_spike/frames/e000*/f0.jpg"))

def mk(nf):
    return [Image.open(f).convert("RGB").resize((448, 336)) for f in fs[:nf]]

instr = "Describe these frames briefly."
for bs in (1, 2, 4, 8, 16):
    imgs, texts = [], []
    for _ in range(bs):
        nf = 8
        imgs.extend(mk(nf))
        content = [{"type": "image"} for _ in range(nf)] + [{"type": "text", "text": instr}]
        texts.append(proc.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    inputs = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
    seq = inputs["input_ids"].shape[1]
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        print("bs=%d seq=%d gen_s=%.2f peak_mb=%.0f"
              % (bs, seq, time.time() - t0, torch.cuda.max_memory_allocated() / 2**20), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("bs=%d OOM" % bs, flush=True)
        break
    del inputs, out
    gc.collect()
    torch.cuda.empty_cache()
