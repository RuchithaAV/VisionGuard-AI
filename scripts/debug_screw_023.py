"""Compare PatchCore heatmap vs ground-truth mask on screw/test/manipulated_front/023.png."""

from pathlib import Path

import cv2
import numpy as np

from src.data.preprocessing import load_and_preprocess_image, load_and_preprocess_mask
from src.evaluation.metrics import compute_pixel_auroc, compute_pixel_iou
from src.inference.decision_engine import DecisionEngine
from src.models.efficientnet import MultiScaleFeatureExtractor
from src.models.feature_memory import PatchCoreMemoryBank
from src.utils.config import load_config

cfg = load_config("configs/model.yaml")
pc_cfg = cfg["patchcore"]

img_path = Path("data/raw/screw/test/manipulated_front/023.png")
mask_path = Path("data/raw/screw/ground_truth/manipulated_front/023_mask.png")

img = load_and_preprocess_image(img_path, target_size=(256, 256), normalize=True)
gt_mask = load_and_preprocess_mask(mask_path, target_size=(256, 256), binarize=True)[..., 0]

extractor = MultiScaleFeatureExtractor(
    input_shape=tuple(pc_cfg["input_shape"]),
    layer_names=pc_cfg["layer_names"],
    pool_size=pc_cfg["pool_size"],
)
bank, meta = PatchCoreMemoryBank.load("models/patchcore/screw", feature_extractor=extractor)
engine = DecisionEngine.from_dict(meta["decision_engine"])

raw_score, heatmap = bank.score_image(
    img,
    gaussian_kernel=pc_cfg["gaussian_kernel"],
    sigma=pc_cfg["gaussian_sigma"],
)

fg = gt_mask > 0.5
bg = ~fg
peak_yx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
gt_center_yx = np.mean(np.argwhere(fg), axis=0)

print(f"Image score      : {raw_score:.2f} (PASS<={engine.pass_threshold:.2f}, FAIL>{engine.fail_threshold:.2f})")
print(f"GT defect pixels : {int(fg.sum())} / {fg.size} ({fg.mean()*100:.2f}%)")
print(f"GT defect center : (y={gt_center_yx[0]:.1f}, x={gt_center_yx[1]:.1f})")
print(f"Heatmap peak     : (y={peak_yx[0]}, x={peak_yx[1]}) | val={heatmap[peak_yx]:.2f}")
print(f"Inside GT mask   : mean={heatmap[fg].mean():.2f}, max={heatmap[fg].max():.2f}")
print(f"Outside GT mask  : mean={heatmap[bg].mean():.2f}, p95={np.percentile(heatmap[bg], 95):.2f}, max={heatmap[bg].max():.2f}")
print(f"Single-img AUROC : {compute_pixel_auroc(gt_mask[None], heatmap[None])*100:.2f}%")
print(f"IoU @ pass_thresh: {compute_pixel_iou(gt_mask[None], heatmap[None], engine.pass_threshold)*100:.2f}%")
print(f"IoU @ 80% peak   : {compute_pixel_iou(gt_mask[None], heatmap[None], 0.8 * heatmap.max())*100:.2f}%")

orig_u8 = (img * 255).astype(np.uint8)
gt_vis = orig_u8.copy()
gt_vis[fg] = [0, 255, 0]

h_norm = np.clip((heatmap - engine.val_mean) / (heatmap.max() - engine.val_mean + 1e-8), 0, 1)
hm_color = cv2.applyColorMap((h_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
pred_vis = cv2.addWeighted(orig_u8, 0.55, hm_color, 0.45, 0)

out = Path("artifacts/screw_023_debug.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), cv2.cvtColor(np.hstack([orig_u8, gt_vis, pred_vis]), cv2.COLOR_RGB2BGR))
print(f"Saved visual comparison to {out}")
