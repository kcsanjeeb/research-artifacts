"""Crop-variant XD feature test: VadCLIP's csv uses crop index 5 in a
ten-crop layout = [tl, tr, bl, br, c] x [orig, flip] -> crop 5 is
TOP-LEFT + horizontal flip (resize shorter side 256, crop 224 at corner).
Extracts N_TEST videos, scores, prints AP2. Compare: center-crop AP2=0.8251.
"""
import os
import sys

import numpy as np
import torch

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
sys.path.insert(0, os.path.join(E0, 'third_party', 'VadCLIP', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xd_replication as xr  # noqa: E402
from clip.clip import build_model  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402
from model import CLIPVAD  # noqa: E402
from utils.tools import get_batch_mask, get_prompt_text  # noqa: E402
from vlmvalue import auc_rank, ap_score  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

N_TEST = int(os.environ.get('FLIP_N', '80'))
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

state = torch.load(os.path.join(E0, 'data', 'model_xd.pth'), map_location='cpu')
clip_state = {k[len('clipmodel.'):]: v for k, v in state.items()
              if k.startswith('clipmodel.')}
model = build_model(clip_state).cuda().eval()

tf_resize = T.Resize(256, interpolation=InterpolationMode.BICUBIC)
tf_to_tensor = T.ToTensor()
tf_norm = T.Normalize(mean=MEAN, std=STD)


def tl_flip(img):
    img = img.convert('RGB')
    img = tf_resize(img)
    img = F.crop(img, 0, 0, 224, 224)
    img = F.hflip(img)
    return tf_norm(tf_to_tensor(img))


rows = xr.xd_rows()[:N_TEST]
SNIP = 16
feats = []
for r in rows:
    vf = os.path.join(E0, 'data', 'xd_test_videos', r['video_id'] + '.mp4')
    vr = VideoReader(vf, ctx=cpu(0), num_threads=2)
    buf, fes = [], []
    for fi in range(len(vr)):
        buf.append(tl_flip(Image.fromarray(vr[fi].asnumpy())))
        if len(buf) == 64:
            x = torch.stack(buf).cuda()
            if next(model.parameters()).dtype == torch.float16:
                x = x.half()
            with torch.no_grad():
                e = model.encode_image(x)
            fes.append(e.float().cpu().numpy())
            buf = []
    if buf:
        x = torch.stack(buf).cuda()
        if next(model.parameters()).dtype == torch.float16:
            x = x.half()
        with torch.no_grad():
            e = model.encode_image(x)
        fes.append(e.float().cpu().numpy())
    fes = np.concatenate(fes)
    n = len(fes) // SNIP
    feats.append(fes[:n * SNIP].reshape(n, SNIP, -1).mean(axis=1)
                 .astype(np.float16))

m = CLIPVAD(7, 512, 256, 512, 1, 1, 64, 10, 10, 'cuda')
m.load_state_dict(state)
m.to('cuda').eval()
pt = get_prompt_text(xr.XD_LABEL_MAP)
gt_all = xr.build_xd_labels(rows)
s_all = {}
with torch.no_grad():
    for i, f in enumerate(feats):
        length = len(f)
        pad = (-length) % 256
        if pad:
            f = np.concatenate([f, np.zeros((pad, f.shape[1]), dtype=f.dtype)])
        v = torch.tensor(f).float().cuda().reshape(-1, 256, 512)
        nseg = (length + 255) // 256
        lengths = torch.tensor([min(256, length - j * 256)
                                for j in range(nseg)], dtype=torch.int)
        pm = get_batch_mask(lengths, 256).cuda()
        _, l1, l2 = m(v, pm, pt, lengths)
        l1 = l1.reshape(-1, l1.shape[2])
        l2 = l2.reshape(-1, l2.shape[2])
        s_all[i] = (1 - l2[0:length].softmax(dim=-1)[:, 0]).cpu().numpy()
gt = np.concatenate(gt_all)
sc = np.concatenate([np.repeat(s_all[i], 16) for i in range(len(rows))])
n = min(len(sc), len(gt))
print(f"TLFLIP-{N_TEST}videos: AUC2={auc_rank(sc[:n], gt[:n]):.4f} "
      f"AP2={ap_score(sc[:n], gt[:n]):.4f} "
      f"(center-crop full: AP2=0.8251, target 84.51)", flush=True)
