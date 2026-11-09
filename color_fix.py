#!/usr/bin/env python3
"""
Correct colour casts (faded dyes, lamp light) and dark faces, per slide.

    python color_fix.py review FOLDER [FOLDER ...]         page: camera look vs corrected, choose
    python color_fix.py apply  FOLDER [FOLDER ...] [--all] re-export the slides whose choice changed
    (proposals without review: pipeline.py decide FOLDER --auto)

Per slide (crops.csv): color_fix = off / mild / auto (full), shadows = 0 / 35 %,
exposure = EV for overexposed slides (by hand; develop());
fixed = what the JPEG currently has ("c35" = full colour + shadows 35 %, "m0" = mild,
"-0" = camera look). The export (slide_crop.py) does the correction:
  mild, no shadows: the camera look with MILD_TINT of the midtone cast removed (Lab
    a*/b* shift), so contrast and brightness stay the camera's (remove_tint)
  everything else, from the raw data (16 bit, linear, camera white balance):
    1. levels per channel (0.1 / 99.7 %; half way towards the common levels for mild),
       without colour: white point = 99.5 % of the brightness
    2. shadows (35 %): local tone mapping, the large-scale brightness (edge-
       preserving guided filter, no halos) pulled towards the brightest areas
    3. the camera's tone curve (camera_curve, measured: models/camera_curve.npz)
    4. colour: per-channel midtone gamma bringing the channel medians together
       (MID_STRENGTH, x 0.5 for mild, damped for green-dominated pictures), then the
       median brightness restored
    5. with shadows: the remaining midtone tint removed (remove_tint): lifted shadows
       show the cast hidden in them (pink/violet veil)
  A shadow lift always comes with at least mild colour (wanted()).

Proposals (candidates, proposal; thresholds in quality.REF): full colour for midtone
cast q_castm >= REF["q_castm_full"] or flag `color`; mild for q_castm >= REF["q_castm"]
unless CLIP sees a warm scene (q_warm: sunset, fire, candle light); shadows 35 % for a
face darker than FACE_DARK (q_face). 55 % was tried: faces brighter, but flat and hazy.

review: local page (http://127.0.0.1:8768): candidates and chosen slides, each twice,
    upright (left camera look, right with the settings below it; orange = proposal,
    green = chosen). Click an image for full size (space = other side, c = colour
    off/mild/full, 0/3 = shadows, arrows, Esc). Saved at once.
apply: re-exports every slide whose choice differs from `fixed` (--all: every corrected
    slide, after a change of the method).
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rawpy
from tqdm import tqdm

import slide_crop as sc

MID_STRENGTH = 0.8
LEVELS = (0.1, 99.7)
COLOUR = {"auto": 1.0, "mild": 0.5}   # color_fix -> strength of the colour correction (off = 0)
MILD_TINT = 0.7      # mild without shadows: share of the midtone cast removed on the camera look
CURVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "camera_curve.npz")
_curve = None


def camera_curve(x):
    """the camera's tone curve (linear, white = 99.5 % brightness -> display 0-1), measured
    on 60 slides: raw data vs the camera's JPEG (models/measure_camera_curve.py)"""
    global _curve
    if _curve is None:
        _curve = dict(np.load(CURVE_FILE))
    return np.interp(np.log2(np.clip(x, 1e-4, None)), _curve["log2x"], _curve["lut"]).astype(np.float32)


def remove_tint(rgb, k=MILD_TINT):
    """uint8 RGB in the camera look -> same with k of the midtone colour cast removed
    (Lab a*, b* shifted; full shift in the midtones, less towards black and white, so
    contrast and brightness stay the camera's)"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    mid = (L > 60) & (L < 200)
    if not mid.any():
        return rgb
    da, db = lab[..., 1][mid].mean() - 128, lab[..., 2][mid].mean() - 128
    w = np.clip(1 - np.abs(L - 128) / 128, 0, 1) ** 0.5
    lab[..., 1] -= k * da * w
    lab[..., 2] -= k * db * w
    return cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
FACE_DARK = 0.25     # face brightness (quality q_face) below this -> shadows 35 %
SHADOWS = (0, 35)


def luminance(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def guided(I, r, eps):
    """self-guided filter (He et al.): edge-preserving smoothing"""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    m = box(I)
    var = box(I * I) - m * m
    a = var / (var + eps)
    b = m - a * m
    return box(a) * I + box(b)


def lift_shadows(lin, amount):
    """local tone mapping of linear RGB: compress the large-scale log brightness
    towards the brightest areas by `amount` (0-1), keep the detail"""
    L = (luminance(lin) + 1e-4).astype(np.float32)
    logL = np.log2(L)
    k = 1000 / max(L.shape)                      # base layer at ~1000 px: same look at any size
    small = cv2.resize(logL, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    base = cv2.resize(guided(small, 24, 0.6), (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    top = np.percentile(base[::4, ::4], 99)
    detail = (logL - base) * 1.15
    newL = np.power(2.0, top + (base - top) * (1 - amount) + detail)
    return np.clip(lin * (newL / L)[..., None], 0, 1)


def green_share(srgb_img):
    """0..1: how much the picture is dominated by green (meadows, forest). The midtone
    balance assumes "grey on average" and turns such pictures violet, so it is damped
    (0 below 25 % green pixels, full damping from 60 %)."""
    hsv = cv2.cvtColor((srgb_img[::4, ::4] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)).mean()
    return float(np.clip((green - 0.25) / 0.35, 0, 1))


def correct(lin, colour=1.0, shadows=0, ev=0.0):
    """linear RGB float (0-1) -> display RGB float (0-1), tone by the camera's curve.
    colour = strength 0..1 (0.5 = mild: levels and midtone balance half way between the
    camera white balance and grey); ev = exposure correction for overexposed slides:
    levels with headroom (white at 99.98 % instead of 99.7 %), then 2**ev"""
    k = float(colour)
    levels = (LEVELS[0], 99.98) if ev else LEVELS
    sample = lin[::4, ::4]
    out = np.empty_like(lin)
    if k:
        glo, ghi = np.percentile(luminance(sample), levels)
        for c in range(3):
            a, b = np.percentile(sample[..., c], levels)
            a, b = glo + (a - glo) * k, ghi + (b - ghi) * k      # k < 1: per channel only partly
            out[..., c] = np.clip((lin[..., c] - a) / max(b - a, 1e-6), 0, 1)
    else:
        out = np.clip(lin / max(np.percentile(luminance(sample), levels[1] if ev else 99.5), 1e-6), 0, 1)
    if ev:
        out = np.clip(out * 2.0 ** ev, 0, 1)
    if shadows:
        out = lift_shadows(out, shadows / 100)
    out = camera_curve(out)                     # brightness and contrast of the camera look
    if not k:
        return out
    before = np.median(luminance(out[::4, ::4]))
    med = np.array([np.median(out[::4, ::4, c]) for c in range(3)]).clip(0.02, 0.98)
    target = np.exp(np.log(med).mean())          # geometric mean of the channel medians
    g = (np.log(target) / np.log(med)) ** (MID_STRENGTH * k * (1 - 0.6 * green_share(out)))
    for c in range(3):
        out[..., c] = np.power(out[..., c], g[c])
    after = np.median(luminance(out[::4, ::4]))
    if 0.02 < after < 0.98 and 0.02 < before < 0.98:
        out = np.power(out, np.log(before) / np.log(after))
    return out


def develop(nef, r, margin, colour, shadows, half=False, camera_rgb=None):
    """crop of row r, corrected; RGB uint8. Mild without shadows: the camera look with
    its tint removed (camera_rgb = that crop if already at hand, e.g. the exported JPEG);
    otherwise developed from the raw data with the camera's tone curve. Column
    `exposure` (EV, e.g. -1) develops from the raw data with highlight headroom
    (LibRaw highlight blending), for overexposed slides"""
    ev = exposure_of(r)
    if float(colour) == 0.5 and not shadows and not ev:
        if camera_rgb is None:
            rgb, _ = sc.load_image(nef)
            camera_rgb = sc.crop_image(rgb, r, margin)
            if half:
                camera_rgb = cv2.resize(camera_rgb, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        return remove_tint(camera_rgb)
    with rawpy.imread(nef) as raw:
        rgb = raw.postprocess(half_size=half, use_camera_wb=True, output_bps=16, gamma=(1, 1),
                              no_auto_bright=True,
                              highlight_mode=rawpy.HighlightMode.Blend if ev else rawpy.HighlightMode.Clip)
    scale = rgb.shape[1] / float(r["img_w"]) if r.get("img_w") else (0.5 if half else 1.0)
    crop = sc.crop_image(rgb, r, margin, scale).astype(np.float32) / 65535
    out = (correct(crop, colour, shadows, ev) * 255 + 0.5).astype(np.uint8)
    return remove_tint(out) if shadows else out


def wanted(r):
    """(colour strength 0/0.5/1, shadows %) chosen for row r"""
    k, s = COLOUR.get(r.get("color_fix"), 0.0), int(r.get("shadows") or 0)
    return (max(k, 0.5) if s else k), s          # a shadow lift always with at least mild colour


def colour_value(k):
    """strength -> color_fix value"""
    return {1.0: "auto", 0.5: "mild"}.get(float(k), "off")


def exposure_of(r):
    """exposure correction in EV for row r (column `exposure`, set by hand), 0 if none"""
    try:
        return float(r.get("exposure") or 0)
    except ValueError:
        return 0.0


def state(colour, shadows, ev=0.0):
    return f"{ {1.0: 'c', 0.5: 'm'}.get(float(colour), '-') }{shadows}" + (f"e{ev:g}" if ev else "")


def state_of(r):
    """what row r should look like, e.g. "c0e-1" = full colour, no shadows, -1 EV"""
    return state(*wanted(r), exposure_of(r))


def needs_fix(r):
    """True if the row is developed by color_fix (colour, shadows or exposure)"""
    return bool(any(wanted(r)) or exposure_of(r))


def current(r):
    """what the JPEG has now (empty `fixed` = camera look)"""
    return r.get("fixed") or "-0"


def orient(img, o):
    from upright import orient as _orient
    return _orient(img, o)


def proposal(tags, r=None):
    """(colour strength, shadows) proposed for a candidate with these tags"""
    k = 1.0 if "colour" in tags else 0.5 if "mild" in tags else 0.0
    shadows = 35 if "dark face" in tags else 0
    return (max(k, 0.5) if shadows else k), shadows


def candidates(folders):
    """[(folder, nef, row, tags)] for the review page (thresholds: quality.REF):
    colour = strong midtone cast (full correction); mild = moderate cast and no warm
    scene; dark face = a face (macOS Vision) darker than FACE_DARK"""
    from quality import REF
    from upright import folders_rows, img_path
    out = []
    for folder, rows in folders_rows(folders):
        for f, r in sorted(rows.items()):
            if sc.rejected(r) or not img_path(folder, f) or not r.get("q_castm"):
                continue
            t = []
            castm, warm = float(r["q_castm"]), float(r.get("q_warm") or 0)
            if castm >= REF["q_castm_full"] or "color" in (r.get("flag") or ""):
                t.append("colour")
            elif castm >= REF["q_castm"] and warm <= REF["q_warm"]:
                t.append("mild")
            if (r.get("q_face") or "none") != "none" and float(r["q_face"]) < FACE_DARK:
                t.append("dark face")
            if t or r.get("color_fix") or r.get("shadows"):
                out.append((folder, f, r, t))
    return out


# ---------------------------------------------------------------- apply

def export_job(a):
    folder, r = a
    return sc.export_one(folder, r, os.path.join(folder, "cropped"), 24, 95)


def cmd_apply(args):
    from upright import folders_rows, jpg_path
    todo = [(folder, r) for folder, rows in folders_rows(args.folders) for r in rows.values()
            if not sc.rejected(r) and (r.get("color_fix") or r.get("shadows") or r.get("exposure"))
            and os.path.exists(jpg_path(folder, r["file"]))      # not exported yet: export does it
            and (state_of(r) != current(r) or (args.all and needs_fix(r)))]
    back = sum(state_of(r) == "-0" for _, r in todo)
    if back:
        print(f"note: {back} slides go back to the camera look, from the JPEG embedded in the NEF "
              "(the camera JPGs are deleted; stronger compression)")
    done = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for (folder, r), (f, note) in tqdm(zip(todo, ex.map(export_job, todo)), total=len(todo),
                                           desc="export", unit="img"):
            if note:
                tqdm.write(f"  {f}: {note}")
            done.setdefault(folder, []).append(r["file"])
    for folder, files in done.items():
        rows = sc.read_csv(folder)
        for f in files:
            rows[f]["fixed"] = state_of(rows[f])
        sc.write_csv(folder, rows)
    print(f"{len(todo)} re-exported")


def accept_proposals(folders):
    """take the proposal as the choice for every candidate not chosen yet"""
    new = {}
    for folder, f, r, tags in candidates(folders):
        if not (r.get("color_fix") or r.get("shadows")):
            new.setdefault(folder, {})[f] = proposal(tags, r)
    for folder, choice in new.items():
        rows = sc.read_csv(folder)
        for f, (colour, shadows) in choice.items():
            rows[f].update(color_fix=colour_value(colour), shadows=str(shadows))
        sc.write_csv(folder, rows)
    n = [c for v in new.values() for c in v.values()]
    print(f"{len(n)} proposals accepted ({sum(c == 1 for c, _ in n)} colour, {sum(c == 0.5 for c, _ in n)} mild, "
          f"{sum(s > 0 for _, s in n)} shadows)")


# ---------------------------------------------------------------- review

PREVIEW = 1600


def encode(img):
    k = PREVIEW / max(img.shape[:2])
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    items = []
    cache = {}
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def image(self, it, colour, shadows):
        key = (it["id"], colour, shadows)
        if key not in self.cache:
            from upright import jpg_path, preview_path
            nef, r = os.path.join(it["folder"], it["nef"]), it["row"]
            o = r.get("upright") or 1
            if colour is None:                   # the camera look
                if os.path.exists(preview_path(it["folder"], it["nef"])):
                    img = orient(cv2.imread(preview_path(it["folder"], it["nef"])), o)
                elif current(r) == "-0":
                    img = cv2.imread(jpg_path(it["folder"], it["nef"]), cv2.IMREAD_REDUCED_COLOR_2)
                else:
                    rgb, _ = sc.load_image(nef)
                    img = cv2.cvtColor(orient(sc.crop_image(rgb, r, 24), o), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(orient(develop(nef, r, 24, colour, shadows, half=True), o), cv2.COLOR_RGB2BGR)
            self.cache[key] = encode(img)
        return self.cache[key]

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/items":
            out = [{k: it[k] for k in ("id", "rack", "file", "tags", "colour", "shadows", "chosen")}
                   for it in self.items]
            self.send(200, json.dumps(out).encode(), "application/json")
        elif u.path.startswith("/img/"):
            it = self.items[int(u.path[5:])]
            q = parse_qs(u.query)
            if "orig" in q:
                body = self.image(it, None, 0)
            else:
                body = self.image(it, float(q.get("c", ["0"])[0]), int(q.get("s", ["0"])[0]))
            self.send(200, body, "image/jpeg")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/set":
            return self.send(404, b"not found", "text/plain")
        with self.lock:
            sets = data["items"]                 # [{id, colour, shadows}]
            for folder in {self.items[int(s["id"])]["folder"] for s in sets}:
                rows = sc.read_csv(folder)
                for s in sets:
                    it = self.items[int(s["id"])]
                    if it["folder"] == folder and int(s["shadows"]) in SHADOWS:
                        rows[it["nef"]].update(color_fix=colour_value(s["colour"]),
                                               shadows=str(int(s["shadows"])))
                        it.update(colour=float(s["colour"]), shadows=int(s["shadows"]), chosen=True)
                        it["row"] = rows[it["nef"]]
                sc.write_csv(folder, rows)
        self.send(200, b'{"ok": true}', "application/json")


def cmd_review(args):
    cands = candidates(args.folders)
    if not cands:
        print("nothing to review (run quality.py scan first)")
        return
    items = []
    for n, (folder, f, r, tags) in enumerate(cands):
        chosen = bool(r.get("color_fix") or r.get("shadows"))
        colour, shadows = wanted(r) if chosen else proposal(tags, r)
        items.append({"id": n, "folder": folder, "nef": f, "row": r, "rack": os.path.basename(folder),
                      "file": f[:-4], "tags": tags, "colour": colour, "shadows": shadows, "chosen": chosen})
    Handler.items = items
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    n = {t: sum(t in it["tags"] for it in items) for t in ("colour", "mild", "dark face")}
    print(f"{len(items)} slides ({n['colour']} full colour, {n['mild']} mild, {n['dark face']} dark face candidates, "
          f"{sum(it['chosen'] for it in items)} already chosen). Review page: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Correction review</title>
<style>
  :root { --bg:#16181c; --panel:#1f2329; --text:#e6e6e6; --muted:#9aa3ad; --line:#333a44; --pick:#3ddc84; --prop:#ffb020; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px -apple-system,system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:14px; align-items:center; padding:8px 14px; flex-wrap:wrap; }
  .muted { color:var(--muted); }
  button, select { background:#2a3038; color:var(--text); border:1px solid var(--line); border-radius:6px;
                   padding:6px 12px; font-size:14px; cursor:pointer; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(560px, 1fr)); gap:12px; padding:12px; }
  .t { background:#0d0f12; border-radius:6px; padding:6px; border:2px solid transparent; }
  .t.chosen { border-color:var(--pick); }
  .pair { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .v { aspect-ratio:1; display:flex; align-items:center; justify-content:center; cursor:zoom-in; overflow:hidden; }
  .v img { max-width:100%; max-height:100%; }
  .cap { font-size:12px; color:var(--muted); padding:4px 2px; display:flex; justify-content:space-between; gap:8px; }
  .ctl { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
  .ctl .grp { display:flex; gap:2px; align-items:center; margin-right:10px; }
  .ctl button { padding:4px 10px; font-size:12px; }
  .ctl button.on { background:var(--prop); color:#000; font-weight:600; }
  .t.chosen .ctl button.on { background:var(--pick); }
  #lb { position:fixed; inset:0; z-index:5; background:#000e; display:none; flex-direction:column; }
  #lb.open { display:flex; }
  #lbimg { flex:1; overflow:auto; display:flex; align-items:center; justify-content:center; }
  #lbimg img { max-width:100%; max-height:100%; }
  #lbbar { display:flex; gap:10px; align-items:center; padding:8px 14px; background:var(--panel); flex-wrap:wrap; }
</style></head><body>
<header>
  <select id="filter">
    <option value="all">all</option>
    <option value="open">not yet chosen</option>
    <option value="colour">full colour candidates</option>
    <option value="mild">mild colour candidates</option>
    <option value="dark face">dark face candidates</option>
    <option value="changed">chosen: corrected</option>
  </select>
  <button id="prev">← page</button><span id="pg"></span><button id="next">page →</button>
  <button id="accept">page: accept shown settings</button><button id="orig">page: all original</button>
  <span class="muted">left = camera look · right = with the settings · orange = proposal, green = chosen · saved immediately</span>
  <span class="muted" id="stats"></span>
</header>
<div id="grid"></div>
<div id="lb">
  <div id="lbimg"><img></div>
  <div id="lbbar"><button id="lbprev">←</button><button id="lbnext">→</button>
    <button id="lbswap">other side (space)</button><span id="lbcap"></span><span class="ctl" id="lbctl"></span>
    <span class="muted">c colour · 0/3 shadows · ←/→ · Esc</span><button id="lbclose" style="margin-left:auto">✕ close</button></div>
</div>
<script>
const PER = 12;
let items = [], view = [], page = 0, cur = -1, side = 1;
async function load() { items = await (await fetch("/api/items")).json(); apply(); }
function apply() {
  const f = document.getElementById("filter").value;
  view = items.filter(i => f === "all" || (f === "open" && !i.chosen) || i.tags.includes(f) ||
                           (f === "changed" && i.chosen && (i.colour || i.shadows)));
  page = Math.min(page, Math.max(0, Math.ceil(view.length / PER) - 1));
  render();
}
const src = (i, s) => s ? `/img/${i.id}?c=${i.colour}&s=${i.shadows}` : `/img/${i.id}?orig=1`;
const ctl = i => `<span class="grp">colour ${[[0, "off"], [0.5, "mild"], [1, "full"]].map(([v, n]) =>
  `<button data-c="${v}" class="${i.colour === v ? "on" : ""}">${n}</button>`).join("")}</span>
  <span class="grp">shadows ${[0, 35].map(s => `<button data-s="${s}" class="${i.shadows === s ? "on" : ""}">${s}%</button>`).join("")}</span>`;
function tile(i, n) {
  return `<div class="t ${i.chosen ? "chosen" : ""}" data-id="${i.id}" data-n="${n}">
    <div class="pair"><div class="v" data-s="0"><img src="${src(i, 0)}" loading="lazy"></div>
    <div class="v" data-s="1"><img src="${src(i, 1)}" loading="lazy"></div></div>
    <div class="cap"><span>${i.rack} ${i.file}</span><span>${i.tags.join(", ")}${i.chosen ? " · chosen" : " · proposal"}</span></div>
    <div class="ctl">${ctl(i)}</div></div>`;
}
function render() {
  document.getElementById("grid").innerHTML = view.slice(page * PER, page * PER + PER).map((i, k) => tile(i, page * PER + k)).join("");
  document.getElementById("pg").textContent = `${view.length ? page + 1 : 0} / ${Math.ceil(view.length / PER)}`;
  stats(); window.scrollTo(0, 0);
}
function stats() {
  const ch = items.filter(i => i.chosen);
  document.getElementById("stats").textContent = `${items.length} slides · chosen ${ch.length} ` +
    `(colour ${ch.filter(i => i.colour).length}, shadows ${ch.filter(i => i.shadows).length})`;
}
function refresh(it) {
  const t = document.querySelector(`.t[data-id="${it.id}"]`);
  if (t) t.outerHTML = tile(it, +t.dataset.n);
  if (cur >= 0 && view[cur].id === it.id) lbshow();
  stats();
}
async function save(list) {
  list.forEach(i => { i.chosen = true; refresh(i); });
  await fetch("/api/set", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({items: list.map(i => ({id: i.id, colour: i.colour, shadows: i.shadows}))})});
}
function setting(it, b) {
  if (b.dataset.c !== undefined) it.colour = +b.dataset.c;
  if (b.dataset.s !== undefined) it.shadows = +b.dataset.s;
  save([it]);
}
document.getElementById("grid").addEventListener("click", e => {
  const t = e.target.closest(".t"); if (!t) return;
  const it = items[+t.dataset.id], b = e.target.closest("button");
  if (b) return setting(it, b);
  const v = e.target.closest(".v"); if (v) show(+t.dataset.n, +v.dataset.s);
});
const pageItems = () => view.slice(page * PER, page * PER + PER);
document.getElementById("accept").onclick = () => save(pageItems());
document.getElementById("orig").onclick = () => save(pageItems().map(i => Object.assign(i, {colour: false, shadows: 0})));
// full-size viewer
function lbshow() {
  const it = view[cur];
  document.querySelector("#lbimg img").src = src(it, side);
  document.getElementById("lbcap").textContent = `${it.rack} ${it.file} · ${side ? "with settings" : "camera look"}`;
  document.getElementById("lbctl").innerHTML = ctl(it);
}
function show(n, s) {
  if (n < 0 || n >= view.length) return;
  cur = n; side = s; lbshow(); document.getElementById("lb").classList.add("open");
}
function hide() {
  document.getElementById("lb").classList.remove("open");
  page = Math.floor(cur / PER); cur = -1; render();
}
document.getElementById("lbctl").onclick = e => { const b = e.target.closest("button"); if (b) setting(view[cur], b); };
document.getElementById("lbswap").onclick = () => { side = 1 - side; lbshow(); };
document.getElementById("lbprev").onclick = () => show(cur - 1, side);
document.getElementById("lbnext").onclick = () => show(cur + 1, side);
document.getElementById("lbclose").onclick = hide;
document.addEventListener("keydown", e => {
  if (cur < 0) return;
  const it = view[cur];
  if (e.key === "Escape") hide();
  else if (e.key === "ArrowLeft") show(cur - 1, side);
  else if (e.key === "ArrowRight") show(cur + 1, side);
  else if (e.key === " ") { e.preventDefault(); side = 1 - side; lbshow(); }
  else if (e.key === "c") { it.colour = {0: 0.5, 0.5: 1, 1: 0}[it.colour]; save([it]); }
  else if ("03".includes(e.key)) { it.shadows = {0: 0, 3: 35}[e.key]; save([it]); }
});
document.getElementById("prev").onclick = () => { if (page > 0) { page--; render(); } };
document.getElementById("next").onclick = () => { if ((page + 1) * PER < view.length) { page++; render(); } };
document.getElementById("filter").onchange = () => { page = 0; apply(); };
load();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["review", "apply"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--all", action="store_true",
                   help="apply: re-export every corrected slide (after a change of the method)")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    {"review": cmd_review, "apply": cmd_apply}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
