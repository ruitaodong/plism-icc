import sys, os, pickle, numpy as np, pandas as pd, time
from PIL import Image
from skimage.registration import phase_cross_correlation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); import align_score as AS
def dec(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def enc(l): l = np.clip(l, 0, 1); return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)
B = 8
df = pd.read_csv("PLISM_sm_en.csv"); grp = df[df.device == "GT450"][["stain", "coordinate", "tissue_type"]].reset_index(drop=True)
lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = []; t0 = time.time()
for i, r in grp.iloc[lo:hi].iterrows():
    a = np.asarray(Image.open(f"psm/{r.stain}/GT450/{r.coordinate}.png").convert("RGB")); b = np.asarray(Image.open(f"psm/{r.stain}/P/{r.coordinate}.png").convert("RGB"))
    F, M = AS.lightness(a), AS.lightness(b)
    (dy, dx), _, _ = phase_cross_correlation(F, M, upsample_factor=1, normalization=None); dy, dx = int(dy), int(dx)
    if max(abs(dy), abs(dx)) > 24: dy = dx = 0                                    # implausible -> keep as stored
    A = a[max(dy, 0):512 + min(dy, 0), max(dx, 0):512 + min(dx, 0)]; Bm = b[max(-dy, 0):512 + min(-dy, 0), max(-dx, 0):512 + min(-dx, 0)]
    h, w = (A.shape[0] // 64) * 64, (A.shape[1] // 64) * 64; A, Bm = A[:h, :w], Bm[:h, :w]
    bm = lambda z: np.round(enc(dec(z / 255.0).reshape(h // B, B, w // B, B, 3).mean((1, 3))) * 65535).astype(np.uint16)
    FA, FB_ = AS.lightness(A), AS.lightness(Bm)
    out.append(dict(stain=r.stain, tissue=r.tissue_type, coord=r.coordinate, dy=dy, dx=dx, r=AS.pearson(FA, FB_), X=bm(A), Y=bm(Bm)))
pickle.dump(out, open(f"sm_pairs_{lo:05d}.pkl", "wb")); print("done", lo, hi, len(out), f"{time.time()-t0:.0f}s", flush=True)
