import cv2, os, time, argparse, uuid
ap = argparse.ArgumentParser()
ap.add_argument("--person", required=True)
ap.add_argument("--out", default="1_data_collection/samples")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--fps", type=float, default=5)
args = ap.parse_args()
os.makedirs(f"{args.out}/{args.person}", exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("No hay cámara disponible")
interval = 1.0/args.fps
t0 = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    t = time.time()
    if t - t0 >= interval and args.n > 0:
        fname = f"{args.out}/{args.person}/{uuid.uuid4().hex}.jpg"
        cv2.imwrite(fname, frame)
        args.n -= 1; t0 = t
        print("saved", fname)
    cv2.imshow("captura", frame)
    if cv2.waitKey(1) & 0xFF == 27 or args.n==0: break
cap.release(); cv2.destroyAllWindows()
