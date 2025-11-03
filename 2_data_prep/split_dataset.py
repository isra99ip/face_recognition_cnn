from pathlib import Path
import shutil, random, os
random.seed(42)
SRC="data/raw_crops"; DST="data"
for split in ["train","val","test"]:
    for c in Path(SRC).glob("*"):
        (Path(DST)/split/c.name).mkdir(parents=True, exist_ok=True)
for cls_dir in Path(SRC).glob("*"):
    imgs = list(cls_dir.glob("*.jpg"))
    random.shuffle(imgs)
    n=len(imgs); a=int(0.7*n); b=int(0.85*n)
    parts={"train":imgs[:a],"val":imgs[a:b],"test":imgs[b:]}
    for sp,files in parts.items():
        for f in files:
            shutil.copy(f, Path(DST)/sp/cls_dir.name/f.name)
print("OK")
