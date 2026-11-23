import sys
from pathlib import Path
from src.inference.predictor import VisionGuardPredictor

p = VisionGuardPredictor()
engine = p.engines["screw"]
print(f"SCREW: PASS <= {engine.pass_threshold:.2f} | REVIEW ({engine.pass_threshold:.2f}, {engine.fail_threshold:.2f}] | FAIL > {engine.fail_threshold:.2f}", flush=True)

test_dir = Path("data/raw/screw/test")

samples = [
    "good/000.png",
    "good/005.png",
    "scratch_head/000.png",
    "scratch_head/005.png",
    "thread_side/000.png",
    "thread_side/002.png",
    "manipulated_front/000.png",
    "manipulated_front/023.png",
    "thread_top/000.png",
    "scratch_neck/000.png",
]

for s in samples:
    pth = test_dir / s
    if pth.exists():
        res = p.predict(pth.read_bytes(), category="screw")
        print(f"{s:<30} -> {res['decision']:<7} (score: {res['raw_score']:.2f}, area: {res['defect_area_pct']:.2f}%)", flush=True)
