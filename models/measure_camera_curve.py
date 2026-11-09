# measure_camera_curve.py OUT.npz SLIDES_ROOT : camera tone curve, measured from raw vs the embedded JPEG of
# 60 random slides (models/camera_curve.npz, used by color_fix.camera_curve)
import sys, glob, random, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, cv2, rawpy
from concurrent.futures import ProcessPoolExecutor
import slide_crop as sc, color_fix as cf
from upright import folders_rows
BINS = np.linspace(-9, 0.5, 400)                  # log2 of the normalised linear value
def job(a):
    folder, r = a
    with rawpy.imread(os.path.join(folder, r["file"])) as raw:
        lin = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=16, gamma=(1, 1), no_auto_bright=True)
        jpg = cv2.imdecode(np.frombuffer(raw.extract_thumb().data, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
    jpg = cv2.cvtColor(cv2.copyMakeBorder(jpg, 4, 4, 4, 4, cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)
    s = lin.shape[1] / float(r["img_w"])
    x = sc.crop_image(lin, r, 24, s).astype(np.float32) / 65535
    y = sc.crop_image(jpg, r, 24, jpg.shape[1] / float(r["img_w"])).astype(np.float32) / 255
    h, w = min(x.shape[0], y.shape[0]), min(x.shape[1], y.shape[1]); x, y = x[:h, :w], y[:h, :w]
    x = x / max(np.percentile(cf.luminance(x[::4, ::4]), 99.5), 1e-6)   # same scaling as correct(colour=0)
    x = cv2.GaussianBlur(x, (0, 0), 1.5); y = cv2.GaussianBlur(y, (0, 0), 1.5)   # alignment / JPEG noise
    idx = np.random.default_rng(0).choice(h * w, 60000, replace=False)
    return np.log2(np.clip(x.reshape(-1, 3)[idx], 1e-4, None)), y.reshape(-1, 3)[idx]
if __name__ == "__main__":
    rows = [(fo, r) for fo, rs in folders_rows(glob.glob(sys.argv[2] + "/*/")) for r in rs.values() if not sc.rejected(r)]
    random.seed(5); pick = random.sample(rows, 60)
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(job, pick))
    X = np.vstack([a for a, _ in res]); Y = np.vstack([b for _, b in res])
    curves = []
    for c in range(3):
        b = np.digitize(X[:, c], BINS)
        med = np.array([np.median(Y[b == i, c]) if (b == i).sum() > 50 else np.nan for i in range(1, len(BINS))])
        curves.append(med)
    curves = np.array(curves); centers = (BINS[1:] + BINS[:-1]) / 2
    mean = np.nanmean(curves, 0)
    ok = ~np.isnan(mean)
    lut = np.maximum.accumulate(np.interp(centers, centers[ok], mean[ok]))      # monotonic
    np.savez(sys.argv[1], log2x=centers, lut=lut, per_channel=curves)
    for v in (0.01, 0.03, 0.1, 0.18, 0.3, 0.5, 0.8, 1.0):
        print(f"linear {v:5.2f}  camera {np.interp(np.log2(v), centers, lut):.3f}")
