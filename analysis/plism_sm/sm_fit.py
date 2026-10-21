import sys, os, glob, pickle, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); import plism_icc as P, fit_blur as FB
def dec(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def enc(l): l = np.clip(l, 0, 1); return np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)
G = sum((pickle.load(open(f, "rb")) for f in sorted(glob.glob("sm_pairs_*.pkl"))), [])
rng = np.random.default_rng(0); test = set()
for st in sorted({g["stain"] for g in G}):
    idx = [i for i, g in enumerate(G) if g["stain"] == st]; test |= set(rng.choice(idx, int(round(0.15 * len(idx))), replace=False).tolist())
train = [g for i, g in enumerate(G) if i not in test]; testg = [g for i, g in enumerate(G) if i in test]
if not os.path.exists("sm_bins.npz"):
    parts = []
    for k in range(0, len(train), 400):
        x = np.concatenate([g["X"].reshape(-1, 3) for g in train[k:k + 400]]).astype(np.float64) / 65535
        y = np.concatenate([g["Y"].reshape(-1, 3) for g in train[k:k + 400]]).astype(np.float64) / 65535
        parts.append(FB.joint_bins(x, y, 7))
    u, cnt, sx, sy = FB.reduce_bins(parts); np.savez("sm_bins.npz", u=u, cnt=cnt, sx=sx, sy=sy)
    print(f"train {len(train)} groups, test {len(testg)} groups ({len({g['tissue'] for g in testg})} tissues); {len(u):,} joint bins", flush=True)
z = np.load("sm_bins.npz"); b = (z["u"], z["cnt"], z["sx"], z["sy"])
def evaluate(lut, groups):
    rp, rt, cm, Lm, Lt = [], [], [], [], []
    for g in groups:
        x = g["X"].astype(np.float64) / 65535; y = g["Y"].astype(np.float64) / 65535; h, w = x.shape[:2]
        m = x.reshape(-1, 3) if lut is None else P.apply_lut(lut, x.reshape(-1, 3)); m = m.reshape(x.shape)
        reg = lambda a: enc(dec(a).reshape(h // 8, 8, w // 8, 8, 3).mean((1, 3))).reshape(-1, 3)
        rp.append(reg(m)); rt.append(reg(y))
        lm, lt = P.srgb_to_lab_d50(m)[..., 0], P.srgb_to_lab_d50(y)[..., 0]
        sm, st_ = lm.reshape(h // 8, 8, w // 8, 8).std((1, 3)).ravel(), lt.reshape(h // 8, 8, w // 8, 8).std((1, 3)).ravel(); ok = st_ > 2
        cm.append(sm[ok] / st_[ok]); Lm.append(lm.ravel()); Lt.append(lt.ravel())
    de = P.delta_e00(np.concatenate(rp), np.concatenate(rt)); Lm, Lt = np.concatenate(Lm), np.concatenate(Lt)
    per = np.array([len(r) for r in rp]); split = np.split(de, np.cumsum(per)[:-1])
    return dict(region_dE00=float(de.mean()), p95=float(np.percentile(de, 95)), contrast=float(np.median(np.concatenate(cm))),
                dark1=float(np.percentile(Lm, 1) - np.percentile(Lt, 1)), bright99=float(np.percentile(Lm, 99) - np.percentile(Lt, 99))), [float(s.mean()) for s in split]
c, lam = float(sys.argv[1]), float(sys.argv[2]); t0 = time.time()
if c < 0:
    res, per = evaluate(None, testg); lut = None; rej = 0.0
else:
    lut, w, hist = FB.robust_fit(b, 33, lam, c, 4); rej = hist[-1][1] if hist else 0.0
    res, per = evaluate(lut, testg); np.save(f"sm_lut_c{c:g}_lam{lam:g}.npy", lut)
res.update(c=c, lam=lam, rejected=rej, secs=round(time.time() - t0))
json.dump(dict(res=res, per_group=per, groups=[(g["stain"], g["tissue"], g["coord"]) for g in testg]), open(f"sm_res_c{c:g}_lam{lam:g}.json", "w"))
print(json.dumps(res), flush=True)
