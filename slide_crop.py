#!/usr/bin/env python3
"""
Crop photographed slides (Dias) out of their mount/frame.

Every NEF is analysed on its own (slides shift between shots, so no shared crop):
the visible film window is a bright rectangle bounded by four sharp edges.
  1. coarse: exhaustive search for the landscape/portrait box whose borders best
     match dark->bright edges (small image)
  2. fine:   each border is fitted as a sub-pixel line (half-size image); a border
     misled by picture content falls back to the common angle; a strip of bare
     film between mount and picture is skipped; the final crop is the largest
     rectangle inside all four borders
  3. export: take the camera's own rendering (camera JPG, else the JPEG embedded in
     the NEF, as Quick Look shows it) or the correction from color_fix.py,
     rotate/crop, inset by a safety margin, save as JPEG with the NEF's EXIF/XMP
     (without MakerNote), orientation and date taken

Two steps, so the crops can be reused later with any converter:

    python slide_crop.py detect  FOLDER [FOLDER ...]   -> FOLDER/crops.csv + review sheets
    python slide_crop.py export  FOLDER [FOLDER ...]   -> FOLDER/cropped/*.jpg
    python slide_crop.py run     FOLDER [FOLDER ...]   -> both

Both steps skip work already done (use --redo / --overwrite to repeat), so an
interrupted run simply continues.

crops.csv coordinates are in pixels of the full-size developed raw image
(rawpy / LibRaw default, e.g. 6032 x 4032), origin top-left:
    cx, cy        centre of the film window
    w, h          size of the window (edge to edge, before the safety margin)
    angle         rotation in degrees (positive = window rotated clockwise)
    status        ok / check: <reason> / manual
    flag          blank / blurry / double (quality.py), duplicate (dupes.py); blank,
                  blurry and duplicate rows get no JPEG (rejected()); old `color`
                  flags from the first colour pass remain as data
The remaining columns are diagnostics. To fix a crop by hand, edit its row and
set status to "manual"; detect never overwrites manual rows.
"""

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import rawpy
from PIL import ExifTags, Image
from tqdm import tqdm

CSV_NAME = "crops.csv"
FIELDS = ["file", "orientation", "cx", "cy", "w", "h", "angle", "status", "img_w", "img_h",
          # diagnostics
          "orient_margin", "straight", "contrast", "unevenness", "film_edge", "angle_fixed",
          "search",
          # upright display (upright.py): EXIF orientation 1-8, how it was decided, confidence
          "upright", "upright_by", "upright_conf", "upright_checked",
          # upright.py textcheck: "<exif> <score> <second> <words>" of the best text reading
          "upright_text",
          # image quality (quality.py): metrics, and flag = blank / blurry / color / double
          "q_std", "q_sharp", "q_ratio", "q_castm", "q_ev", "q_double", "q_warm", "q_face", "flag",
          # dupes.py: rack/file of the slide this one duplicates (flag duplicate)
          "duplicate_of",
          # correction from the NEF (color_fix.py): color_fix auto = colour corrected (full),
          # mild = half strength, off,
          # shadows = shadow lift in % (0/35), fixed = what the JPEG currently has
          "color_fix", "shadows", "exposure", "fixed",
          # pipeline.py: crop the preview in _preview/ was made from (cx,cy,w,h,angle)
          "preview"]

# film formats: name -> (aspect range long/short side, long side vertical,
#                        window long side / image width)
FORMATS = {"landscape": ((1.38, 1.62), False, (0.40, 0.75)),    # 35 mm film, 3:2
           "portrait": ((1.38, 1.62), True, (0.40, 0.75)),
           "square": ((0.95, 1.08), False, (0.22, 0.55))}       # 126 / Instamatic, 28 x 28 mm
# Square slides are only searched in racks that need it (see cmd_detect): searching
# square everywhere lets wrong boxes compete and spoils the 3:2 results.
FORMATS_35MM = ("landscape", "portrait")
FORMATS_ALL = ("landscape", "portrait", "square")
COARSE_WIDTH = 200              # px, image width for the coarse search
DEBUG = bool(os.environ.get("SLIDE_CROP_DEBUG"))


# ---------------------------------------------------------------- detection

def gradients(gray, sigma):
    g = np.log(gray.astype(np.float32) + 4.0)     # contrast invariant
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    return gx, gy


def box_scores(cx, cy, w, h):
    """Score map for all boxes of size w x h: mean of +gx on left edge,
    -gx on right, +gy on top, -gy on bottom. cx/cy are cumulative sums."""
    H, W = cy.shape[0], cx.shape[1]
    ny, nx = H - h, W - w
    if ny <= 0 or nx <= 0:
        return None
    V = cx[h:h + ny, :] - cx[:ny, :]
    Hh = cy[:, w:w + nx] - cy[:, :nx]
    return ((V[:, :nx] - V[:, w:w + nx]) / h + (Hh[:ny, :] - Hh[h:h + ny, :]) / w) / 4


def coarse_search(gx, gy, formats=FORMATS_35MM):
    """Brute-force axis-aligned box search.
    Returns {format: (score, x, y, w, h, format)}."""
    c = 0.06  # clip: long straight edges must win over short strong content edges
    gx, gy = np.clip(gx, -c, c) / c, np.clip(gy, -c, c) / c
    H, W = gx.shape
    cx = np.vstack([np.zeros((1, W), np.float32), np.cumsum(gx, 0)])
    cy = np.hstack([np.zeros((H, 1), np.float32), np.cumsum(gy, 1)])
    best = {}
    for orient in formats:
        (amin, amax), vertical, (lmin, lmax) = FORMATS[orient]
        b = (-1e9,)
        for L in range(int(lmin * W), int(lmax * W) + 1):
            for S in range(int(L / amax), int(L / amin) + 1):
                w, h = (S, L) if vertical else (L, S)
                sc = box_scores(cx, cy, w, h)
                if sc is None:
                    continue
                i = int(np.argmax(sc))
                y, x = divmod(i, sc.shape[1])
                if sc[y, x] > b[0]:
                    b = (float(sc[y, x]), x, y, w, h, orient)
        best[orient] = b
    return best


def edge_points(gx, gy, side, line_pos, a, b, half, sign):
    """Sub-pixel edge positions along one border.
    side 'v': vertical border near x=line_pos(t) for rows t in [a,b];
    side 'h': horizontal border near y=line_pos(t) for cols t in [a,b]."""
    pts = []
    step = 6
    for t in range(int(a), int(b), step):
        p = line_pos(t)
        lo, hi = int(round(p - half)), int(round(p + half)) + 1
        if side == "v":
            if lo < 1 or hi >= gx.shape[1] - 1:
                continue
            prof = sign * gx[t:t + step, lo:hi].mean(0)
        else:
            if lo < 1 or hi >= gy.shape[0] - 1:
                continue
            prof = sign * gy[lo:hi, t:t + step].mean(1)
        k = int(np.argmax(prof))
        if prof[k] <= 0 or k == 0 or k == len(prof) - 1:
            continue
        y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
        d = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / d if d != 0 else 0.0
        pts.append((t + step / 2, lo + k + off, prof[k]))
    return np.array(pts).reshape(-1, 3)


def robust_line(pts, tol=1.0):
    """Fit pos = m*t + q, dropping outliers.
    Returns dict(m, q, inl=inlier fraction, strength=median edge gradient, res=inlier residuals)."""
    if len(pts) < 8:
        return None
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        m, q = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
        r = np.abs(pts[:, 1] - (m * pts[:, 0] + q))
        thr = max(tol, 2.5 * np.median(r[keep]))
        new = r < thr
        if new.sum() < 8 or (new == keep).all():
            break
        keep = new
    res = pts[keep, 1] - (m * pts[keep, 0] + q)
    return dict(m=m, q=q, inl=keep.mean(), strength=float(np.median(pts[keep, 2])), res=res)


def robust_offset(pts, m, q, tol=1.5):
    """Like robust_line, but with the slope m fixed."""
    if len(pts) < 8:
        return None
    r = pts[:, 1] - (m * pts[:, 0] + q)
    off = np.median(r)
    for _ in range(4):
        keep = np.abs(r - off) < max(tol, 2.5 * np.median(np.abs(r - off)))
        off = np.median(r[keep])
    q += off
    res = pts[keep, 1] - (m * pts[keep, 0] + q)
    return dict(m=m, q=q, inl=keep.mean(), strength=float(np.median(pts[keep, 2])), res=res)


def angle_of(name, m):
    """Border slope -> rotation angle in degrees (same sense for all four borders)."""
    return np.degrees(np.arctan(m)) * (1 if name in ("top", "bottom") else -1)


def support(gx, gy, side, sign, m, q, a, b):
    """How well a line lies on a real edge: median over the border of the strongest
    edge gradient within +-2 px of the line."""
    vals = []
    for t in np.arange(a, b, 4):
        p = int(round(m * t + q))
        ti = int(t)
        if side == "v":
            if 2 <= p < gx.shape[1] - 3 and 0 <= ti < gx.shape[0]:
                vals.append((sign * gx[ti, p - 2:p + 3]).max())
        else:
            if 2 <= p < gy.shape[0] - 3 and 0 <= ti < gy.shape[1]:
                vals.append((sign * gy[p - 2:p + 3, ti]).max())
    return float(np.median(vals)) if vals else 0.0


# name, orientation of the line, inward direction (+1: towards larger x/y)
EDGES = (("left", "v", +1), ("right", "v", -1), ("top", "h", +1), ("bottom", "h", -1))


def fit_edge(gx, gy, side, sign, m, q, a, b, half):
    span = b - a
    # wide search band: stay clear of the neighbouring borders; narrow band: go close to
    # the corners, mount edges are often bent there
    skip = 0.08 * span if half > 8 else 12
    pts = edge_points(gx, gy, side, lambda t: m * t + q, a + skip, b - skip, half, sign)
    return robust_line(pts)


def edge_spans(lines):
    """Extent of each border = between its two neighbouring borders (at the box centre)."""
    c, w, h, _ = lines_to_rect(lines)
    return {"left": (c[1] - h / 2, c[1] + h / 2), "right": (c[1] - h / 2, c[1] + h / 2),
            "top": (c[0] - w / 2, c[0] + w / 2), "bottom": (c[0] - w / 2, c[0] + w / 2)}


def film_edge_inside(lg, side, sign, m, q, a, b, depth):
    """Look inward from a mount border for a strip of bare film (film rebate):
    a featureless band followed by a second straight, parallel dark->bright edge.
    Returns (inward distance of that edge or 0, its gradient sign seen inward)."""
    t = np.arange(a + 0.1 * (b - a), b - 0.1 * (b - a), 2, dtype=np.float32)
    d = np.arange(0, depth, dtype=np.float32)
    T, D = np.meshgrid(t, d, indexing="ij")
    pos = m * T + q + sign * D
    mx, my = (pos, T) if side == "v" else (T, pos)
    S = cv2.remap(lg, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
    G = np.gradient(S, axis=1)
    G_med = np.median(G, axis=0)          # straight edges along the whole border survive the median
    lo = 8
    k = lo + int(np.argmax(np.abs(G_med[lo:-2])))
    e = G_med[k]                          # the strip may be brighter or darker than the picture
    if abs(e) < 0.08:
        return 0.0, 0
    if np.mean(np.sign(e) * G[:, k] > 0.3 * abs(e)) < 0.7:   # present along most of the border
        return 0.0, 0
    band = S[:, 4:k - 3]
    if band.shape[1] < 2:
        return 0.0, 0
    # bare film is uniform along the border; picture content is not
    if np.median(band.std(axis=0)) > 0.15:
        return 0.0, 0
    return float(k), int(np.sign(e))


def refine(gray, gx, gy, x, y, w, h):
    """Refine an axis-aligned box (in gx/gy pixel units) to four sub-pixel lines.
    Returns (lines, info)."""
    init = {"left": x, "right": x + w, "top": y, "bottom": y + h}
    spans = {"left": (y, y + h), "right": (y, y + h), "top": (x, x + w), "bottom": (x, x + w)}
    lines = {n: dict(m=0.0, q=init[n]) for n, _, _ in EDGES}
    for half in (max(w, h) * 0.03, 4):          # wide pass, then narrow pass
        for name, side, sign in EDGES:
            fit = None
            # a soft edge (light fall-off) can put the coarse box off by more than
            # the band: widen it
            for widen in ((1, 2, 3) if half > 8 else (1,)):
                fit = fit_edge(gx, gy, side, sign, lines[name]["m"], lines[name]["q"],
                               *spans[name], half * widen)
                if fit is not None:
                    break
            if fit is None:
                return None, {}
            lines[name] = fit
        spans = edge_spans(lines)

    # a border whose angle disagrees with the others was misled by picture content
    # (happens mostly on the short borders): keep its position, use the common angle
    angles = {n: angle_of(n, l["m"]) for n, l in lines.items()}
    med = float(np.median(list(angles.values())))
    fixed = []
    for name, side, sign in EDGES:
        if abs(angles[name] - med) <= 0.8:
            continue
        m = np.tan(np.radians(med)) * (1 if side == "h" else -1)
        a, b = spans[name]
        tm = (a + b) / 2
        q = lines[name]["m"] * tm + lines[name]["q"] - m * tm      # same position mid-border
        for half in (max(w, h) * 0.03, 4):
            skip = 0.08 * (b - a) if half > 8 else 12
            pts = edge_points(gx, gy, side, lambda t: m * t + q, a + skip, b - skip, half, sign)
            fit = robust_offset(pts, m, q)
            if fit is None:
                break
            q = fit["q"]
        if fit is None:
            continue
        # keep the common angle only if it is at least about as well supported by a
        # real edge (a slide can sit genuinely skewed in its mount)
        s_orig = support(gx, gy, side, sign, lines[name]["m"], lines[name]["q"], a + 12, b - 12)
        s_fix = support(gx, gy, side, sign, fit["m"], fit["q"], a + 12, b - 12)
        if DEBUG:
            print(f"  {name}: angle {angles[name]:+.2f} vs {med:+.2f}  support {s_orig:.3f} -> {s_fix:.3f}")
        if s_fix < 0.8 * s_orig:
            continue
        # near each end, if the original line is the better supported one there, make
        # sure we stay inside it
        L = 0.25 * (b - a)
        push = 0.0
        for e0, e1 in ((a + 12, a + 12 + L), (b - 12 - L, b - 12)):
            so = support(gx, gy, side, sign, lines[name]["m"], lines[name]["q"], e0, e1)
            sf = support(gx, gy, side, sign, fit["m"], fit["q"], e0, e1)
            if DEBUG:
                print(f"     end support orig {so:.3f} fixed {sf:.3f}")
            if so > 1.5 * sf and so > 0.05:
                for te in (e0, e1):
                    d = sign * ((lines[name]["m"] * te + lines[name]["q"]) - (fit["m"] * te + fit["q"]))
                    push = max(push, d)
        if DEBUG:
            print(f"     push inward {push:.1f}")
        fit["q"] += sign * push
        lines[name] = fit
        fixed.append(name)
    spans = edge_spans(lines)

    # bare film strip between mount and picture -> move border to the picture edge
    lg = cv2.GaussianBlur(np.log(gray.astype(np.float32) + 4.0), (0, 0), 1.0)
    moved = []
    for name, side, sign in EDGES:
        size = w if side == "v" else h
        dk, esign = film_edge_inside(lg, side, sign, lines[name]["m"], lines[name]["q"],
                                     *spans[name], int(0.07 * size))
        if dk:
            fit = fit_edge(gx, gy, side, sign * esign, lines[name]["m"],
                           lines[name]["q"] + sign * dk, *spans[name], 4)
            if fit is not None:
                lines[name] = fit
                moved.append(name)

    # uneven mount edges: shift each border inward so (almost) no frame pixels remain
    shift = 0.0
    for name, side, sign in EDGES:
        s = max(0.0, float(np.percentile(sign * lines[name]["res"], 98)))
        if name in moved:   # soft picture edge behind a film strip: keep clear of it
            s += 3
        lines[name]["q"] += sign * s
        shift = max(shift, s)
    info = dict(inl=min(l["inl"] for l in lines.values()),
                strength=min(l["strength"] for l in lines.values()),
                shift=shift, film_edge=" ".join(moved), angle_fixed=" ".join(fixed))
    return lines, info


def lines_to_rect(lines):
    """Intersect the four lines -> centre, size, angle (degrees)."""
    def v(name):  # x = m*y + q
        return lines[name]["m"], lines[name]["q"]

    hz = v  # y = m*x + q

    def cross(vn, hn):
        mv, qv = v(vn)
        mh, qh = hz(hn)
        yy = (mh * qv + qh) / (1 - mh * mv)
        return np.array([mv * yy + qv, yy])

    tl, tr = cross("left", "top"), cross("right", "top")
    bl, br = cross("left", "bottom"), cross("right", "bottom")
    # angle: horizontal borders slope +m, vertical borders slope -m (x = m y)
    ang = np.degrees(np.mean([np.arctan(lines["top"]["m"]), np.arctan(lines["bottom"]["m"]),
                              -np.arctan(lines["left"]["m"]), -np.arctan(lines["right"]["m"])]))
    # the four borders are not exactly a rectangle (keystone, lens distortion, film
    # skewed in the mount): take the largest rectangle at that angle that lies inside
    t = np.radians(ang)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # rotate by -angle
    TL, TR, BR, BL = (R @ p for p in (tl, tr, br, bl))
    left, right = max(TL[0], BL[0]), min(TR[0], BR[0])
    top, bottom = max(TL[1], TR[1]), min(BL[1], BR[1])
    c = R.T @ np.array([(left + right) / 2, (top + bottom) / 2])
    return c, right - left, bottom - top, ang


def detect_one(path, square=False):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    full_h, full_w = rgb.shape[0] * 2, rgb.shape[1] * 2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # coarse: small image, exhaustive search over size and position, both orientations
    k = COARSE_WIDTH / gray.shape[1]
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    best = coarse_search(*gradients(small, 0.7), FORMATS_ALL if square else FORMATS_35MM)
    orient = max(best, key=lambda o: best[o][0])
    # how clearly the format won: 3:2 against the other orientation, square against 3:2
    rivals = {"landscape": ["portrait"], "portrait": ["landscape"],
              "square": ["landscape", "portrait"]}[orient]
    margin = best[orient][0] - max(best[o][0] for o in rivals)
    _, x, y, w, h, _ = best[orient]

    # fine: sub-pixel borders on the half-size image
    gx, gy = gradients(gray, 1.0)
    lines, info = refine(gray, gx, gy, x / k, y / k, w / k, h / k)
    row = {"file": os.path.basename(path), "orientation": orient,
           "img_w": full_w, "img_h": full_h,
           "orient_margin": round(margin, 3), "search": "all" if square else "35mm"}
    if lines is None:
        c = np.array([(x + w / 2) / k, (y + h / 2) / k])
        ww, hh, ang = w / k, h / k, 0.0
    else:
        c, ww, hh, ang = lines_to_rect(lines)
        row.update(straight=round(info["inl"], 3), contrast=round(info["strength"], 3),
                   unevenness=round(info["shift"] * 2, 1), film_edge=info["film_edge"],
                   angle_fixed=info["angle_fixed"])
    # half-size pixel centres -> full-size coordinates
    row.update(cx=round((c[0] + 0.5) * 2 - 0.5, 2), cy=round((c[1] + 0.5) * 2 - 0.5, 2),
               w=round(ww * 2, 2), h=round(hh * 2, 2), angle=round(float(ang), 3))
    return row


# ---------------------------------------------------------------- rack-guided search
# Second opinion for flagged slides: search again for a box of the rack's typical
# window size (from its ok slides), on a finer coarse grid. Checked by eye on the 91
# flagged slides of the collection: 79 correct; it fails where a mount's window is
# smaller than the rack's typical one. Used by review.py as the suggestion.
GUIDED_WIDTH = 400              # px, coarse grid of the guided search
SQUARE_PER_SHORT = 1.165        # 126 window side / 35 mm short side (measured, 1971)


def rack_typical_sizes(rows):
    """Median (w, h) per format from a rack's ok/manual slides, completed by symmetry."""
    t = {}
    for o in ("landscape", "portrait", "square"):
        ok = [r for r in rows.values() if r["status"] in ("ok", "manual") and r["orientation"] == o]
        if len(ok) >= 3:
            t[o] = (float(np.median([float(r["w"]) for r in ok])),
                    float(np.median([float(r["h"]) for r in ok])))
    if "landscape" in t and "portrait" not in t:
        t["portrait"] = t["landscape"][::-1]
    if "portrait" in t and "landscape" not in t:
        t["landscape"] = t["portrait"][::-1]
    if "square" not in t and "landscape" in t:
        s = min(t["landscape"]) * SQUARE_PER_SHORT
        t["square"] = (s, s)
    return t


def detect_guided(path, typical):
    """Like detect_one, but only boxes within +-4 % of the rack's typical sizes."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = GUIDED_WIDTH / gray.shape[1]
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gx, gy = gradients(small, 0.7)
    c = 0.06
    gx, gy = np.clip(gx, -c, c) / c, np.clip(gy, -c, c) / c
    H, W = gx.shape
    cx = np.vstack([np.zeros((1, W), np.float32), np.cumsum(gx, 0)])
    cy = np.hstack([np.zeros((H, 1), np.float32), np.cumsum(gy, 1)])
    best = {}
    for o, (tw, th) in typical.items():
        w0, h0 = tw / 2 * k, th / 2 * k                       # full size -> coarse grid
        b = (-1e9,)
        for w in range(int(w0 * 0.96), int(w0 * 1.04) + 2):
            for h in range(int(h0 * 0.96), int(h0 * 1.04) + 2):
                s = box_scores(cx, cy, w, h)
                if s is None:
                    continue
                i = int(np.argmax(s))
                y, x = divmod(i, s.shape[1])
                if s[y, x] > b[0]:
                    b = (float(s[y, x]), x, y, w, h)
        best[o] = b
    orient = max(best, key=lambda o: best[o][0])
    rest = [best[o][0] for o in best if o != orient]
    margin = best[orient][0] - (max(rest) if rest else 0)
    _, x, y, w, h = best[orient]
    GX, GY = gradients(gray, 1.0)
    lines, info = refine(gray, GX, GY, x / k, y / k, w / k, h / k)
    row = {"file": os.path.basename(path), "orientation": orient,
           "img_w": gray.shape[1] * 2, "img_h": gray.shape[0] * 2,
           "orient_margin": round(margin, 3), "search": "rack-guided"}
    if lines is None:
        cc = np.array([(x + w / 2) / k, (y + h / 2) / k])
        ww, hh, ang = w / k, h / k, 0.0
    else:
        cc, ww, hh, ang = lines_to_rect(lines)
        row.update(straight=round(info["inl"], 3), contrast=round(info["strength"], 3),
                   unevenness=round(info["shift"] * 2, 1), film_edge=info["film_edge"],
                   angle_fixed=info["angle_fixed"])
    row.update(cx=round((cc[0] + 0.5) * 2 - 0.5, 2), cy=round((cc[1] + 0.5) * 2 - 0.5, 2),
               w=round(ww * 2, 2), h=round(hh * 2, 2), angle=round(float(ang), 3))
    return row


def assess(row, typical_short=None):
    """'ok' or 'check: <reasons>'. typical_short: median short side of the folder's
    slides of the same format family (None: not enough to compare)."""
    if row.get("straight", "") == "":
        return "check: no edges found"
    w, h = float(row["w"]), float(row["h"])
    short, long_ = min(w, h), max(w, h)
    # empty projector slot: hardly any edge contrast (and no clear format)
    contrast = float(row["contrast"])
    if contrast < 0.005 or (contrast < 0.012 and float(row["orient_margin"]) < 0.1):
        return "check: no slide?"
    why = []
    if float(row["orient_margin"]) < 0.06:
        why.append("orientation unsure")
    if float(row["straight"]) < 0.45:
        why.append("border not straight")
    if abs(float(row["angle"])) > 2.5:
        why.append("large tilt")
    if float(row["unevenness"]) > 20:
        why.append("uneven border")
    if not (1.38 <= long_ / short <= 1.60 or long_ / short <= 1.08):
        why.append("unusual aspect")
    if typical_short and abs(short / typical_short - 1) > 0.04:
        why.append("unusual size")
    return "check: " + ", ".join(why) if why else "ok"


# ---------------------------------------------------------------- csv / review

def read_csv(folder):
    p = os.path.join(folder, CSV_NAME)
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def write_csv(folder, rows):
    p = os.path.join(folder, CSV_NAME)
    with open(p + ".tmp", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for name in sorted(rows):
            wr.writerow({k: rows[name].get(k, "") for k in FIELDS})
    os.replace(p + ".tmp", p)


def corners(r, margin=0.0):
    cx, cy, w, h, a = (float(r[k]) for k in ("cx", "cy", "w", "h", "angle"))
    w, h = w - 2 * margin, h - 2 * margin
    t = np.radians(a)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return pts @ R.T + [cx, cy]


def review_tile(folder, r, width=600):
    with rawpy.imread(os.path.join(folder, r["file"])) as raw:
        thumb = raw.extract_thumb()
    img = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
    k = width / float(r["img_w"])
    img = cv2.resize(img, (width, int(round(float(r["img_h"]) * k))), interpolation=cv2.INTER_AREA)
    st = r["status"]
    col = (0, 255, 0) if st == "ok" else (0, 0, 255) if st.startswith("check") else (255, 160, 0)
    cv2.polylines(img, [np.round(corners(r) * k).astype(np.int32)], True, col, 1, cv2.LINE_AA)
    cv2.putText(img, f'{r["file"]}  {st}', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return img


def _review_tile(args):
    return review_tile(*args)


def write_review(folder, rows, jobs=None, per_sheet=24, cols=4):
    out = os.path.join(folder, "cropped", "_review")
    os.makedirs(out, exist_ok=True)
    for old in glob.glob(os.path.join(out, "*.jpg")):
        os.remove(old)
    names = sorted(rows)
    with ProcessPoolExecutor(jobs) as ex:
        all_tiles = list(tqdm(ex.map(_review_tile, [(folder, rows[n]) for n in names], chunksize=4),
                              total=len(names), desc="review sheets", unit="img"))
    for s in range(0, len(names), per_sheet):
        tiles = all_tiles[s:s + per_sheet]
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(out, f"review_{s // per_sheet + 1:03d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


# ---------------------------------------------------------------- export

def camera_jpg(path):
    """The camera JPG shot next to the NEF (RAW+JPEG), if it exists."""
    base = os.path.splitext(path)[0]
    for ext in (".JPG", ".jpg", ".JPEG", ".jpeg"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def load_image(path):
    """RGB image in the camera look, in the pixel grid of crops.csv, plus a note if it
    had to be developed from the raw instead. Taken from the camera JPG next to the NEF
    if there is one (highest JPEG quality, 12-14 MB), otherwise from the JPEG embedded
    in the NEF (same look, ~1-2 MB, stronger compression). Both lack the 8 px border
    LibRaw keeps, so they are padded back (the two are aligned within 0.5 px)."""
    with rawpy.imread(path) as raw:
        H, W = raw.sizes.height, raw.sizes.width
        try:
            jpg = camera_jpg(path)
            img = cv2.imread(jpg, cv2.IMREAD_COLOR) if jpg else None
            if img is None:
                th = raw.extract_thumb()
                if th.format == rawpy.ThumbFormat.JPEG:
                    img = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                dy, dx = H - img.shape[0], W - img.shape[1]
                if 0 <= dx <= 64 and 0 <= dy <= 64:   # full size, centred on the sensor
                    img = cv2.copyMakeBorder(img, dy // 2, dy - dy // 2, dx // 2, dx - dx // 2,
                                             cv2.BORDER_REPLICATE)
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            pass
        return (raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8),
                "no full-size camera preview, developed from raw")


# IFD0 tags that describe the NEF's embedded 160x120 preview, not the photo
NEF_STRUCTURE_TAGS = (254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 330, 532, 700)


def nef_metadata(path, width, height, orientation=1, taken=None):
    """EXIF (all of it, except the Nikon MakerNote, which is ~200 KB and does not fit
    into a JPEG's 64 KB EXIF segment) and XMP of the NEF, adapted to the cropped size.
    orientation: EXIF orientation (1-8) to display the slide upright.
    taken: (EXIF date, year) from dates.slide_datetime, or None to keep the scan date."""
    try:
        with Image.open(path) as nef:
            exif = nef.getexif()
            xmp = nef.info.get("xmp")
            for k in NEF_STRUCTURE_TAGS:
                exif.pop(k, None)
            sub = exif.get_ifd(ExifTags.IFD.Exif)
            sub.pop(0x927C, None)                       # MakerNote
            sub[0xA002], sub[0xA003] = width, height    # PixelX/YDimension
            exif.get_ifd(ExifTags.IFD.GPSInfo)          # load, so it is written too
            exif[0x0112] = orientation
            if taken:
                import dates
                dates.set_exif_date(exif, taken[0])
                xmp = dates.set_xmp_dates(xmp, taken[0], taken[1]) if xmp else xmp
            return exif.tobytes(), xmp
    except Exception:
        return None, None


XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"


def set_jpeg_orientation(path, orientation):
    """Set the EXIF orientation of an existing JPEG without re-encoding it."""
    with Image.open(path) as im:
        exif = im.getexif()
    if exif.get(0x0112, 1) == orientation:
        return False
    exif[0x0112] = orientation
    rewrite_jpeg_metadata(path, exif)
    return True


def rewrite_jpeg_metadata(path, exif, xmp=None):
    """Replace the EXIF (and, if given, the XMP) APP1 segment of an existing JPEG
    without re-encoding it: the image data stays byte-identical, file date kept."""
    data = open(path, "rb").read()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"not a JPEG: {path}")
    seg = lambda body: b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body
    raw = exif.tobytes()
    new = {"exif": seg(raw if raw.startswith(b"Exif") else b"Exif\x00\x00" + raw)}
    if xmp is not None:
        new["xmp"] = seg(XMP_ID + (xmp if isinstance(xmp, bytes) else xmp.encode("utf-8")))
    out, i, done = bytearray(data[:2]), 2, set()
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"unexpected JPEG structure: {path}")
        marker = data[i + 1]
        if marker == 0xDA:                                   # start of scan: rest unchanged
            for k in new:
                if k not in done:
                    out += new[k]
            out += data[i:]
            break
        n = int.from_bytes(data[i + 2:i + 4], "big")
        part = data[i:i + 2 + n]
        kind = ("exif" if part[4:10] == b"Exif\x00\x00" else
                "xmp" if part[4:4 + len(XMP_ID)] == XMP_ID else None) if marker == 0xE1 else None
        if kind in new and kind not in done:
            out += new[kind]
            done.add(kind)
        else:
            out += part
        i += 2 + n
    st = os.stat(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
    os.utime(path, (st.st_atime, st.st_mtime))
    return True


def crop_image(img, r, margin, scale=1.0):
    """cut the (rotated) crop of row r out of img; scale = img size / crops.csv grid"""
    cx, cy, w, h, a = (float(r[k]) for k in ("cx", "cy", "w", "h", "angle"))
    W, H = int(round((w - 2 * margin) * scale)), int(round((h - 2 * margin) * scale))
    # destination pixel (u,v) -> source: centre + R(angle) * (u - W/2, v - H/2)
    t = np.radians(a)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    off = (np.array([cx, cy]) + 0.5) * scale - 0.5 - R @ np.array([(W - 1) / 2, (H - 1) / 2])
    M = np.hstack([R, off[:, None]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def export_one(folder, r, out_dir, margin, quality):
    import color_fix
    colour, shadows = color_fix.wanted(r)
    if color_fix.needs_fix(r):                  # corrected (color_fix.py)
        camera_rgb = None
        dst = os.path.join(out_dir, os.path.splitext(r["file"])[0] + ".jpg")
        if colour == 0.5 and not shadows and os.path.exists(dst) and (r.get("fixed") or "-0") == "-0":
            # mild = tint removed on the camera look: the exported JPEG is that look in the
            # best quality left (made from the camera JPG, which may be deleted by now)
            camera_rgb = cv2.cvtColor(cv2.imread(dst, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
                                      cv2.COLOR_BGR2RGB)
        crop = color_fix.develop(os.path.join(folder, r["file"]), r, margin, colour, shadows,
                                 camera_rgb=camera_rgb)
        note = None
    else:
        rgb, note = load_image(os.path.join(folder, r["file"]))
        crop = crop_image(rgb, r, margin)
    H, W = crop.shape[:2]
    dst = os.path.join(out_dir, os.path.splitext(r["file"])[0] + ".jpg")
    import dates
    taken = dates.slide_datetime(folder, r["file"], read_csv(folder))
    exif, xmp = nef_metadata(os.path.join(folder, r["file"]), W, H, int(r.get("upright") or 1), taken)
    meta = {k: v for k, v in (("exif", exif), ("xmp", xmp)) if v}
    if exif is None:
        note = (note + "; " if note else "") + "EXIF could not be copied"
    Image.fromarray(crop).save(dst, quality=quality, **meta)
    st = os.stat(os.path.join(folder, r["file"]))
    os.utime(dst, (st.st_atime, st.st_mtime))
    return r["file"], note


# ---------------------------------------------------------------- commands

def detect_many(folder, rows, files, square, jobs, desc):
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(detect_one, os.path.join(folder, f), square): f for f in files}
        for i, fu in enumerate(tqdm(as_completed(futs), total=len(futs), desc=desc, unit="img"), 1):
            f = futs[fu]
            try:
                yield f, fu.result()
            except Exception as e:  # unreadable file etc.
                tqdm.write(f"  ERROR {f}: {e}")
                continue
            if i % 20 == 0:
                write_csv(folder, rows)  # checkpoint, so an interrupted run resumes


def assess_all(rows):
    # typical size per format family (35 mm vs square), from at least 3 slides
    shorts = {}
    for r in rows.values():
        if r.get("straight", "") != "":
            shorts.setdefault(r["orientation"] == "square", []).append(min(float(r["w"]), float(r["h"])))
    typical = {fam: float(np.median(v)) for fam, v in shorts.items() if len(v) >= 3}
    for r in rows.values():
        if r.get("status") != "manual":
            r["status"] = assess(r, typical.get(r["orientation"] == "square"))


def cmd_detect(folder, args):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".nef") and not f.startswith("._"))
    if not files:                               # not a rack (models/, system folders, ...)
        return
    rows = {f: r for f, r in read_csv(folder).items() if f in files}   # drop deleted NEFs
    todo = [f for f in files if args.redo and rows.get(f, {}).get("status") != "manual"
            or f not in rows]
    print(f"{folder}: {len(files)} NEF, {len(todo)} to detect")
    for f, row in detect_many(folder, rows, todo, False, args.jobs, "detect"):
        rows[f] = row
    assess_all(rows)

    # square slides (126 / Instamatic): searching them for every slide spoils 3:2
    # results, so the first pass is 3:2 only. If many slides of a rack are flagged
    # (>= 10 %, empty slots not counted), the rack probably contains square slides:
    # redo it with all formats (that leaves its 3:2 results unchanged). Retrying single
    # flagged slides as square was tried and rejected: it produced wrong squares.
    untried = lambda r: r.get("search", "35mm") == "35mm"     # square search not tried yet
    flagged_now = [f for f, r in rows.items() if r["status"].startswith("check")
                   and r["status"] != "check: no slide?" and untried(r)]
    if len(flagged_now) >= max(2, 0.10 * len(rows)):
        redo = [f for f, r in rows.items() if r.get("status") != "manual" and untried(r)]
        first = {f: dict(rows[f]) for f in redo}
        for f, row in detect_many(folder, rows, redo, True, args.jobs, "many flags, all formats"):
            rows[f] = row
        assess_all(rows)
        # keep it only where it found real squares: a slide becomes square only if
        # that crop is ok, and the rack needs at least 3 of them (squares come in
        # groups; single "squares" in 3:2 racks were wrong)
        good = [f for f in redo if rows[f]["orientation"] == "square" and rows[f]["status"] == "ok"]
        for f in redo:
            if len(good) < 3 or (rows[f]["orientation"] == "square" and f not in good):
                rows[f] = dict(first[f], search="35mm, square rejected")
        assess_all(rows)

    write_csv(folder, rows)
    flagged = [n for n in sorted(rows) if rows[n]["status"].startswith("check")]
    squares = sum(r["orientation"] == "square" for r in rows.values())
    out = write_review(folder, rows, args.jobs)
    print(f"  crops: {os.path.join(folder, CSV_NAME)}")
    print(f"  review sheets: {out}")
    if squares:
        print(f"  {squares} square slides")
    if flagged:
        print(f"  {len(flagged)} flagged for checking: {', '.join(flagged)}")


def rejected(r):
    """flagged blank or blurry (quality.py) or duplicate (dupes.py): no JPEG"""
    return bool({"blank", "blurry", "duplicate"} & set((r.get("flag") or "").split(",")))


def cmd_export(folder, args):
    rows = read_csv(folder)
    if not rows:
        print(f"{folder}: no {CSV_NAME}, run detect first")
        return
    out_dir = os.path.join(folder, "cropped")
    os.makedirs(out_dir, exist_ok=True)
    jpg = lambda r: os.path.join(out_dir, os.path.splitext(r["file"])[0] + ".jpg")
    # cropped/ follows crops.csv: slides flagged blank/blurry/duplicate get no JPEG,
    # an existing one is removed (the NEF stays; the flag says why)
    gone = [r for r in rows.values() if rejected(r) and os.path.exists(jpg(r))]
    for r in gone:
        os.remove(jpg(r))
        print(f"  removed {os.path.basename(jpg(r))} ({r['flag']})")
    todo = [r for r in rows.values() if not rejected(r) and (args.overwrite or not os.path.exists(jpg(r)))]
    print(f"{folder}: exporting {len(todo)} of {len(rows)}" + (f", {len(gone)} flagged JPEGs removed" if gone else ""))
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(export_one, folder, r, out_dir, args.margin, args.quality)
                for r in todo]
        done = []
        for fu in tqdm(as_completed(futs), total=len(futs), desc="export", unit="img"):
            try:
                f, note = fu.result()
                done.append(f)
                if note:
                    tqdm.write(f"  {f}: {note}")
            except Exception as e:
                tqdm.write(f"  ERROR: {e}")
    if done:                                    # record which corrections the JPEGs have
        rows = read_csv(folder)
        for f in done:
            r = rows[f]
            import color_fix
            r["fixed"] = color_fix.state_of(r)
        write_csv(folder, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["detect", "export", "run"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--redo", action="store_true", help="detect again (except rows marked manual)")
    p.add_argument("--overwrite", action="store_true", help="re-export existing JPEGs")
    p.add_argument("--margin", type=float, default=24,
                   help="safety inset in px per side, covers the blurred mount edge (default 24)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    args = p.parse_args()
    for folder in args.folders:
        if args.command in ("detect", "run"):
            cmd_detect(folder, args)
        if args.command in ("export", "run"):
            cmd_export(folder, args)


if __name__ == "__main__":
    sys.exit(main())
