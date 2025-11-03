import cv2, os, glob
from pathlib import Path

SRC = "1_data_collection/samples"
DST = "data/raw_crops"
SIZE = 224
os.makedirs(DST, exist_ok=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for person_dir in sorted(Path(SRC).glob("*")):
    if not person_dir.is_dir(): 
        continue
    out_dir = Path(DST)/person_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for imgp in glob.glob(str(person_dir/"*.jpg")):
        img = cv2.imread(imgp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces)==0:
            h,w = gray.shape; s=min(h,w); y=(h-s)//2; x=(w-s)//2; crop=img[y:y+s,x:x+s]
        else:
            x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
            pad = int(0.2*max(w,h))
            x0 = max(0,x-pad); y0=max(0,y-pad)
            x1 = min(img.shape[1], x+w+pad); y1=min(img.shape[0], y+h+pad)
            crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, (SIZE,SIZE))
        cv2.imwrite(str(out_dir/Path(imgp).name), crop)
print("OK")
