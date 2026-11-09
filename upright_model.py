"""Slide orientation guess with the pretrained model of github.com/ternaus/check_orientation
(ResNeXt50, 4 classes: class k = upright image turned k x 90 deg counter-clockwise).
Weights: models/2020-11-16_resnext50_32x4d.pth"""
import os
import cv2
import numpy as np
import timm
import torch

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "2020-11-16_resnext50_32x4d.pth")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model():
    model = timm.create_model("resnext50_32x4d", pretrained=False, num_classes=4)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.eval().to(dev), dev


def prepare(rgb):
    img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return (img - MEAN) / STD


@torch.no_grad()
def predict(model, dev, rgbs):
    """rgbs: list of RGB uint8 images. Returns (N, 4) probabilities that the image is
    the upright one turned k x 90 deg counter-clockwise, averaged over the 4 test rotations."""
    batch = []
    for rgb in rgbs:
        for j in range(4):
            batch.append(prepare(np.ascontiguousarray(np.rot90(rgb, j))))
    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(dev)
    p = torch.softmax(model(x), 1).cpu().numpy().reshape(len(rgbs), 4, 4)
    out = np.zeros((len(rgbs), 4))
    for j in range(4):                 # rotating the input by j shifts the class by j
        out += np.roll(p[:, j, :], -j, axis=1)
    return out / 4
