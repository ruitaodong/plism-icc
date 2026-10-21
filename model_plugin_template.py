"""
Model plugin for fit_pyr.py --model-plugin (template; best guess of a typical proprietary CNN).

Edit the marked lines for your models. Only three things are required:
  TILE, load(), predict(model, tiles) -> {"seg": (N,C,h,w) probs, "cls": (N,K) probs, "feat": (N,D)}
Any key may be omitted (e.g. no "feat"). Probabilities, not logits.
Tiles arrive as (N, TILE, TILE, 3) uint8 RGB at the hold-out evaluation level; choose
fit_pyr.py --eval-level so that level matches your model's magnification (MPP below).
"""
import numpy as np

TILE = 512          # <- model input size, px
MPP = 0.5           # <- model resolution, um/px (informational; pick --eval-level to match)
BATCH = 16
DEVICE = "cuda"     # <- or "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # <- your normalisation (ImageNet guess)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_feat = {}


def load():
    import torch
    seg = torch.jit.load("seg_model.pt", map_location=DEVICE).eval()     # <- your segmentation model
    cls = torch.jit.load("cls_model.pt", map_location=DEVICE).eval()     # <- your classifier (or None if one net)
    # features: global-average-pool an internal layer via a forward hook (name is a guess)
    try:
        layer = dict(cls.named_modules())["encoder.layer4"]              # <- bottleneck / penultimate layer
        layer.register_forward_hook(lambda m, i, o: _feat.__setitem__("x", o))
    except Exception:
        pass
    return {"seg": seg, "cls": cls}


def predict(model, tiles):
    import torch
    x = (tiles.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(DEVICE)
    out = {}
    with torch.no_grad():
        seg_logits = model["seg"](x)                                     # <- (N, C, h, w)
        out["seg"] = torch.softmax(seg_logits, 1).float().cpu().numpy()  #    sigmoid instead if multi-label
        if model["cls"] is not None:
            cls_logits = model["cls"](x)                                 # <- (N, K)
            out["cls"] = torch.softmax(cls_logits, 1).float().cpu().numpy()
            if "x" in _feat:
                f = _feat["x"]
                out["feat"] = (f.mean((2, 3)) if f.ndim == 4 else f).float().cpu().numpy()
    return out


# ---- ONNX alternative (no PyTorch): replace load/predict with e.g.
# import onnxruntime as ort
# def load():  return ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# def predict(sess, tiles):
#     x = ((tiles.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)
#     seg_logits, cls_logits = sess.run(None, {sess.get_inputs()[0].name: x})
#     e = lambda z, ax: np.exp(z - z.max(ax, keepdims=True)) / np.exp(z - z.max(ax, keepdims=True)).sum(ax, keepdims=True)
#     return {"seg": e(seg_logits, 1), "cls": e(cls_logits, 1)}
