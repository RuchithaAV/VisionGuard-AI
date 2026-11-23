from pathlib import Path
from src.inference.predictor import VisionGuardPredictor

p = VisionGuardPredictor()

for cat in ["metal_nut", "capsule", "screw"]:
    engine = p.engines[cat]
    print(f"\n=================== {cat.upper()} ===================")
    print(f"PASS <= {engine.pass_threshold:.2f} | REVIEW ({engine.pass_threshold:.2f} to {engine.fail_threshold:.2f}] | FAIL > {engine.fail_threshold:.2f}")
    
    test_dir = Path(f"data/raw/{cat}/test")
    found_review = []
    
    for pth in sorted(test_dir.rglob("*.png")):
        rel = pth.relative_to(test_dir)
        res = p.predict(pth.read_bytes(), category=cat)
        if res["decision"] == "REVIEW":
            found_review.append((str(rel), res["raw_score"], res["defect_area_pct"]))
            print(f"  FOUND REVIEW: {rel} | score={res['raw_score']:.2f} | area={res['defect_area_pct']:.2f}%", flush=True)
            if len(found_review) >= 3:
                break
