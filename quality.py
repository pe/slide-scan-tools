#!/usr/bin/env python3
"""
Measure every slide; find blank, blurry and double slides (two pictures in one).

    python quality.py scan   FOLDER [FOLDER ...]   measurements into crops.csv
    python quality.py faces  FOLDER [FOLDER ...]   face brightness (after upright.py guess)
    python quality.py review FOLDER [FOLDER ...]   page with the suspicious slides
    python quality.py list   FOLDER [FOLDER ...]   print the flagged slides
                                  [--md FILE]      rescan list as Markdown, grouped by rack

scan: per slide (1500 px version, 5 % border ignored) writes to crops.csv
    q_std    brightness spread; < 0.06 = nearly uniform (blank, washed out)
    q_sharp  strongest edges / contrast range; low = soft
    q_ratio  edge strength at full vs quarter size; ~1 = nothing finer than coarse
             blur (the most direct blur measure, but also low for soft subjects)
    q_ev     exposure time vs the rack's median, in EV (from EXIF). Two slides
             stuck together in the projector let through much less light: +3..+5.5
             (but dark single slides, e.g. night shots, also reach +3)
    q_double CLIP (ViT-L-14, models/clip) similarity to "double exposure / two
             photos overlaid" minus "a normal photograph"; finds double exposures
             made in the camera, which q_ev can't
    q_castm  colour cast: distance of the mean midtone colour from grey (Lab a*b* of
             the pixels with L 60-200); used by color_fix.py (mild / full correction)
    q_warm   CLIP "sunset / fire / candle light / warm evening light" minus "ordinary
             daytime photo": genuine warm colours that must not be corrected
    q_face   brightness (Lab L, 0-1) of the darkest face found by macOS Vision on the
             upright slide, "none" without faces (`faces` command; dark faces -> shadow
             lift in color_fix.py)
review: local page (http://127.0.0.1:8767). Candidates by the fixed thresholds in
    REF (from the whole collection): std < 0.06 or q_sharp / q_ratio below the
    collection's lowest 3 % ("blank / blurry"), z(q_double) + 0.5 z(q_ev) in its
    highest 4 % ("double"). Buttons under each slide toggle the flags; click the
    image for full size (keys 1/2/3 flag, arrows = next/previous, Esc = close).
    Saved at once to crops.csv (column flag, e.g. "blurry,double"; flags of other
    tools are kept; slides flagged duplicate are left to dupes.py). The export
    removes the JPEGs of blank/blurry/duplicate slides.
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from tqdm import tqdm

import slide_crop as sc
from upright import folders_rows, full_crop, img_path, jpg_path, load_small, orient

STD_BLANK = 0.06
# Candidate thresholds, fixed from the whole collection (3818 slides, 2026-10-29), so a
# single rack or a handful of rescans is judged by the same standard. They match the
# percentages used on the collection: blur lowest 3 %, double highest 4 %, warm scene
# highest 5 %. mean/std pairs turn a measurement into a z-score.
REF = {
    "q_sharp": 1.2837, "q_ratio": 1.917,            # below = blurry candidate
    "q_ev": (0.2404, 1.228), "q_double": (0.0182, 0.0251), "double": 1.7415,
    "q_castm": 30.0,                                 # midtone cast from here: mild colour correction
    "q_castm_full": 47.0,                            # from here: full correction (= the former whole-
                                                     # image cast 37.8, collection's highest 5 %)
    "q_warm": -0.0020,                  # above = genuine warm scene (collection 95 %)
}


def zref(r, k):
    """z-score of measurement k against the collection"""
    mean, std = REF[k]
    return (float(r.get(k) or mean) - mean) / std


def double_score(r):
    return zref(r, "q_double") + EV_WEIGHT * zref(r, "q_ev")


FLAGS = ("blank", "blurry", "double")
CLIP_MODEL = ("ViT-L-14", "openai")
CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip")
DOUBLE_PROMPTS = ["a double exposure photo with two scenes superimposed",
                  "two photographs overlaid on top of each other, ghosting"]
SINGLE_PROMPTS = ["a normal photograph", "a single clear photo of one scene"]
# CLIP scores: column -> (prompts for, prompts against)
CLIP_SCORES = {
    "q_double": (DOUBLE_PROMPTS, SINGLE_PROMPTS),
    "q_warm": (["a sunset or sunrise sky", "a campfire or bonfire at night",
                "fireworks or candle light in the dark", "a photo lit by warm orange evening light"],
               ["an ordinary daytime photo", "a photo taken indoors", "people outdoors in daylight"]),
}
EV_WEIGHT = 0.5      # long exposures also come from dark night/flash shots; 1.0 let them crowd the list


def edge_strength(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.percentile(np.hypot(gx, gy), 99.5))


def measure_slide(a):
    """measurements of a slide from its full-size crop in the camera look, scaled to
    1/4 (camera JPG or the JPEG embedded in the NEF, as pipeline.py does while making
    the preview; the 1600 px preview itself is ~25 % less sharp and does not fit REF)"""
    folder, r = a
    rgb, _ = sc.load_image(os.path.join(folder, r["file"]))
    crop = cv2.cvtColor(sc.crop_image(rgb, r, 24), cv2.COLOR_RGB2BGR)
    return measure_img(cv2.resize(crop, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA))


def measure_img(img):
    """BGR image at ~1500 px (the size and sharpness of a JPEG decoded at 1/4; the
    thresholds in REF depend on it)"""
    h, w = img.shape[:2]
    img = img[int(h * .05):int(h * .95), int(w * .05):int(w * .95)]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    lo, hi = np.percentile(g, [1, 99])
    rng = max(hi - lo, 1e-3)
    fine = edge_strength(cv2.GaussianBlur(g, (0, 0), 0.8))
    coarse = edge_strength(cv2.resize(g, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)) / 4
    mid = (lab[..., 0] > 60) & (lab[..., 0] < 200)
    castm = np.hypot(lab[..., 1][mid].mean() - 128, lab[..., 2][mid].mean() - 128) if mid.any() else 0.0
    return dict(q_std=f"{g.std():.4f}", q_sharp=f"{fine / rng:.4f}",
                q_ratio=f"{fine / max(coarse, 1e-6):.3f}", q_castm=f"{castm:.1f}")


def exposure(path):
    """exposure time from the EXIF of the NEF (or a JPEG)"""
    from PIL import Image
    with Image.open(path) as im:
        t = im.getexif().get_ifd(0x8769).get(0x829a)
    return float(t) if t else None


def clip_scores(paths, columns, batch=16):
    """CLIP_SCORES columns for each path: {column: [score per path]}"""
    import open_clip
    import torch
    from PIL import Image
    model, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1],
                                                          cache_dir=CLIP_DIR)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer(CLIP_MODEL[0])
    text = {}
    with torch.no_grad():
        for c in columns:
            pos, neg = CLIP_SCORES[c]
            tf = model.encode_text(tok(pos + neg).to(dev))
            text[c] = (tf / tf.norm(dim=-1, keepdim=True), len(pos))
    out = {c: [] for c in columns}
    for i in tqdm(range(0, len(paths), batch), desc="clip", unit="batch"):
        ims = torch.stack([pre(Image.fromarray(cv2.cvtColor(load_small(p, 448), cv2.COLOR_BGR2RGB)))
                           for p in paths[i:i + batch]]).to(dev)
        with torch.no_grad():
            f = model.encode_image(ims)
            f /= f.norm(dim=-1, keepdim=True)
            for c, (tf, n) in text.items():
                sim = (f @ tf.T).cpu().numpy()
                out[c] += [float(q[:n].mean() - q[n:].mean()) for q in sim]
    return out


def face_brightness(path, o):
    """brightness (Lab L 0-1) of the darkest face on the upright slide, or "none"
    (macOS Vision face rectangles; faces < 0.15 % of the image or confidence < 0.5 ignored)"""
    import objc
    import Vision
    import upright_ocr
    img = orient(load_small(path, 1200), o)
    h, w = img.shape[:2]
    with objc.autorelease_pool():
        req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        hd = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            upright_ocr._cgimage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), None)
        hd.performRequests_error_([req], None)
        boxes = [(b.boundingBox().origin.x, b.boundingBox().origin.y, b.boundingBox().size.width,
                  b.boundingBox().size.height, float(b.confidence())) for b in req.results() or []]
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255
    faces = [float(np.median(L[max(int((1 - y - bh) * h), 0):int((1 - y) * h), max(int(x * w), 0):int((x + bw) * w)]))
             for x, y, bw, bh, c in boxes if c >= 0.5 and bw * bh >= 0.0015]
    return f"{min(faces):.3f}" if faces else "none"


def face_job(a):
    path, o = a
    try:
        return face_brightness(path, o)
    except Exception:
        return ""


def cmd_faces(args):
    """q_face for every slide that has an orientation (run after upright.py guess)"""
    todo = [(folder, f, r) for folder, rows in folders_rows(args.folders) for f, r in sorted(rows.items())
            if img_path(folder, f) and not sc.rejected(r) and (args.redo or not r.get("q_face"))]
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=200) as ex:
        res = list(tqdm(ex.map(face_job, [(img_path(fo, f), r.get("upright") or 1) for fo, f, r in todo], chunksize=8),
                        total=len(todo), desc="faces", unit="img"))
    by = {}
    for (folder, f, _), v in zip(todo, res):
        by.setdefault(folder, {})[f] = v
    for folder, vals in by.items():
        rows = sc.read_csv(folder)
        for f, v in vals.items():
            rows[f]["q_face"] = v
        sc.write_csv(folder, rows)
    print(f"{len(todo)} slides: faces measured")


def flags(r):
    return [f for f in (r.get("flag") or "").split(",") if f]


def cmd_scan(args):
    for folder, rows in tqdm(folders_rows(args.folders), desc="racks", unit="rack"):
        todo = [f for f in sorted(rows) if img_path(folder, f) and (args.redo or not rows[f].get("q_castm"))]
        with ProcessPoolExecutor(args.jobs) as ex:
            for f, m in zip(todo, ex.map(measure_slide, [(folder, rows[f]) for f in todo], chunksize=4)):
                rows[f].update(m)
        # exposure (from the NEF) relative to the rack median
        have = [f for f in rows if os.path.exists(os.path.join(folder, f))]
        t = {f: exposure(os.path.join(folder, f)) for f in have}
        med = np.median([v for v in t.values() if v]) if any(t.values()) else None
        for f in have:
            if t[f] and med:
                rows[f]["q_ev"] = f"{np.log2(t[f] / med):.2f}"
        sc.write_csv(folder, rows)
    # CLIP scores (GPU, one process), for each column only where missing
    for c in CLIP_SCORES:
        todo = [(folder, f) for folder, rows in folders_rows(args.folders) for f, r in sorted(rows.items())
                if img_path(folder, f) and (args.redo or not r.get(c))]
        if not todo:
            continue
        scores = clip_scores([img_path(folder, f) for folder, f in todo], [c])[c]
        new = {}
        for (folder, f), v in zip(todo, scores):
            new.setdefault(folder, {})[f] = v
        for folder, vals in new.items():
            rows = sc.read_csv(folder)
            for f, v in vals.items():
                rows[f][c] = f"{v:.4f}"
            sc.write_csv(folder, rows)


def rescan_reason(r):
    """why a slide should be rescanned, or None"""
    fl = flags(r)
    if "blank" in fl:
        return "blank"
    if "blurry" in fl:
        return "blurry"
    if "double" in fl:
        return "double (two pictures in one)"
    return None


def rescan_markdown(folders):
    """Markdown list of the slides to rescan, grouped by rack"""
    racks, total = [], 0
    for folder, rows in folders_rows(folders):
        order = sorted(rows)
        hit = [(f, rescan_reason(r)) for f, r in sorted(rows.items()) if rescan_reason(r)]
        if hit:
            racks.append((os.path.basename(folder), order, hit))
            total += len(hit)
    out = ["# Slides to rescan", "",
           f"{total} slides in {len(racks)} racks, from the flags set in `quality.py review` "
           "(generated with `quality.py list --md`).", "",
           "- **blank / blurry**: the JPEG was deleted, the NEF kept.",
           "- **double**: two pictures in one shot, checked by eye.",
           "- **Position** = place in the rack's shooting order (n of all shots of the rack); "
           "deleted empty slots and duplicates can shift it by a few places.", ""]
    for name, order, hit in racks:
        out += [f"## {name}", "", "| file | position | reason |", "|---|---|---|"]
        out += [f"| {f[:-4]} | {order.index(f) + 1} / {len(order)} | {why} |" for f, why in hit]
        out.append("")
    return "\n".join(out)


def cmd_list(args):
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(rescan_markdown(args.folders))
        print(f"written: {args.md}")
        return
    n = 0
    for folder, rows in folders_rows(args.folders):
        hit = [(f, flags(r)) for f, r in sorted(rows.items()) if flags(r)]
        if hit:
            print(os.path.basename(folder))
            for f, fl in hit:
                print(f"  {f[:-4]}  {', '.join(fl)}")
            n += len(hit)
    print(f"{n} slides flagged")


# ---------------------------------------------------------------- review

class Handler(BaseHTTPRequestHandler):
    items = []
    thumbs = {}
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

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/items":
            out = [{k: it[k] for k in ("id", "rack", "file", "why", "cand", "flag", "rq", "rd")} for it in self.items]
            self.send(200, json.dumps(out).encode(), "application/json")
        elif self.path.startswith("/thumb/"):
            i = int(self.path[7:])
            if i not in self.thumbs:
                it = self.items[i]
                img = orient(load_small(img_path(it["folder"], it["nef"]), 480), it["upright"])
                self.thumbs[i] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            self.send(200, self.thumbs[i], "image/jpeg")
        elif self.path.startswith("/full/"):
            it = self.items[int(self.path[6:])]
            if os.path.exists(jpg_path(it["folder"], it["nef"])):
                with open(jpg_path(it["folder"], it["nef"]), "rb") as f:
                    self.send(200, f.read(), "image/jpeg")
            else:
                img = full_crop(it["folder"], it["nef"], sc.read_csv(it["folder"])[it["nef"]])
                self.send(200, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), "image/jpeg")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/set":
            return self.send(404, b"not found", "text/plain")
        flag = [f for f in FLAGS if f in data["flag"]]
        with self.lock:
            it = self.items[int(data["id"])]
            rows = sc.read_csv(it["folder"])
            other = [f for f in (rows[it["nef"]].get("flag") or "").split(",") if f and f not in FLAGS]
            flag = ",".join(flag + other)        # keep flags of other tools (duplicate)
            rows[it["nef"]]["flag"] = flag
            sc.write_csv(it["folder"], rows)
            it["flag"] = flag
        self.send(200, b'{"ok": true}', "application/json")


def cmd_review(args):
    all_ = []
    for folder, rows in folders_rows(args.folders):
        for f, r in sorted(rows.items()):
            if r.get("q_castm") and img_path(folder, f) and "duplicate" not in (r.get("flag") or ""):
                all_.append((folder, f, r))
    if not all_:
        print("no measurements, run scan first")
        return
    # percentile rank of each metric (0 = most suspicious)
    rank = {}
    for k, rev in (("q_sharp", False), ("q_ratio", False)):
        order = sorted(range(len(all_)), key=lambda i: float(all_[i][2][k]), reverse=rev)
        rank[k] = {i: n / len(all_) for n, i in enumerate(order)}
    # double: z-scores (against the collection) of CLIP score and exposure offset
    dz = np.array([double_score(r) for _, _, r in all_])
    order = np.argsort(-dz)
    rank["double"] = {int(i): n / len(all_) for n, i in enumerate(order)}
    items = []
    for i, (folder, f, r) in enumerate(all_):
        blank = float(r["q_std"]) < STD_BLANK
        soft = [k for k in ("q_sharp", "q_ratio") if float(r[k]) < REF[k]]
        double = dz[i] > REF["double"]
        why = ([f"std {float(r['q_std']):.3f}"] if blank else []) + \
              [f"{k[2:]} {float(r[k]):.2f}" for k in soft] + \
              ([f"double {dz[i]:.1f} (ev {r.get('q_ev') or '?'})"] if double else [])
        cand = (["q"] if blank or soft else []) + (["d"] if double else [])
        if cand or r.get("flag"):
            score = 0 if blank else min(rank[k][i] for k in rank)
            items.append({"folder": folder, "nef": f, "upright": r.get("upright") or 1, "rack": os.path.basename(folder), "file": f[:-4],
                          "why": ", ".join(why), "cand": cand, "flag": r.get("flag") or "", "score": score,
                          "rq": 0 if blank else min(rank[k][i] for k in ("q_sharp", "q_ratio")),
                          "rd": rank["double"][i]})
    items.sort(key=lambda it: it["score"])
    for n, it in enumerate(items):
        it["id"] = n
    Handler.items = items
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    n = {c: sum(c in it["cand"] for it in items) for c in "qd"}
    print(f"{len(items)} candidates of {len(all_)} slides ({n['q']} blank/blurry, {n['d']} double). "
          f"Review page: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Quality review</title>
<style>
  :root { --bg:#16181c; --panel:#1f2329; --text:#e6e6e6; --muted:#9aa3ad; --line:#333a44;
          --blank:#ff4d4d; --blurry:#ffb020; --double:#4dc3ff; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px -apple-system,system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:14px; align-items:center; padding:8px 14px; flex-wrap:wrap; }
  .muted { color:var(--muted); }
  button, select { background:#2a3038; color:var(--text); border:1px solid var(--line); border-radius:6px;
                   padding:6px 12px; font-size:14px; cursor:pointer; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:8px; padding:12px; }
  .t { background:#0d0f12; border:3px solid transparent; border-radius:6px; overflow:hidden; }
  .t.f-blank { border-color:var(--blank); } .t.f-blurry { border-color:var(--blurry); } .t.f-double { border-color:var(--double); }
  .box { width:100%; aspect-ratio:1; display:flex; align-items:center; justify-content:center; cursor:zoom-in; }
  .box img { max-width:100%; max-height:100%; }
  .cap { font-size:11px; color:var(--muted); padding:3px 6px; display:flex; justify-content:space-between; gap:6px; }
  .fl { display:flex; gap:4px; padding:0 6px 6px; }
  .fl button { flex:1; padding:4px 0; font-size:12px; }
  .fl button.on[data-f=blank] { background:var(--blank); color:#000; }
  .fl button.on[data-f=blurry] { background:var(--blurry); color:#000; }
  .fl button.on[data-f=double] { background:var(--double); color:#000; }
  #lb { position:fixed; inset:0; z-index:5; background:#000e; display:none; flex-direction:column; }
  #lb.open { display:flex; }
  #lbimg { flex:1; overflow:auto; display:flex; align-items:center; justify-content:center; cursor:zoom-in; }
  #lbimg img { max-width:100%; max-height:100%; }
  #lbimg.real { display:block; cursor:zoom-out; } #lbimg.real img { max-width:none; max-height:none; }
  #lbbar { display:flex; gap:10px; align-items:center; padding:8px 14px; background:var(--panel); flex-wrap:wrap; }
  #lbbar .fl { padding:0; } #lbbar .fl button { padding:6px 14px; font-size:14px; }
</style></head><body>
<header>
  <select id="filter">
    <option value="q">blank / blurry candidates</option>
    <option value="d">double candidates</option>
    <option value="all">all candidates</option>
    <option value="flag">flagged only</option>
  </select>
  <button id="prev">← page</button><span id="pg"></span><button id="next">page →</button>
  <span class="muted">Buttons toggle the flags · click image = full size · saved immediately, nothing deleted</span>
  <span class="muted" id="stats"></span>
</header>
<div id="grid"></div>
<div id="lb">
  <div id="lbimg"><img></div>
  <div id="lbbar"><button id="lbprev">←</button><button id="lbnext">→</button><span id="lbcap"></span>
    <span class="fl" id="lbfl"></span>
    <span class="muted">1 blank · 2 blurry · 3 double · click image = 100 % / fit · ←/→ · Esc</span>
    <button id="lbclose" style="margin-left:auto">✕ close</button></div>
</div>
<script>
const PER = 48, FL = ["blank", "blurry", "double"],
      LABEL = {blank: "blank", blurry: "blurry", double: "double"};
let items = [], view = [], page = 0, cur = -1;
async function load() { items = await (await fetch("/api/items")).json(); apply(); }
function apply() {
  const f = document.getElementById("filter").value;
  view = items.filter(i => f === "all" ? i.cand.length : f === "flag" ? i.flag : i.cand.includes(f));
  if (f === "q") view.sort((a, b) => a.rq - b.rq);
  if (f === "d") view.sort((a, b) => a.rd - b.rd);
  page = Math.min(page, Math.max(0, Math.ceil(view.length / PER) - 1));
  render();
}
const has = (i, f) => i.flag.split(",").includes(f);
const buttons = i => FL.map(f => `<button data-f="${f}" class="${has(i, f) ? "on" : ""}">${LABEL[f]}</button>`).join("");
function tile(i, n) {
  const c = FL.filter(f => has(i, f)).map(f => "f-" + f)[0] || "";
  return `<div class="t ${c}" data-id="${i.id}" data-n="${n}"><div class="box"><img src="/thumb/${i.id}" loading="lazy"></div>
    <div class="cap"><span>${i.rack} ${i.file.slice(4)}</span><span>${i.why}</span></div>
    <div class="fl">${buttons(i)}</div></div>`;
}
function render() {
  document.getElementById("grid").innerHTML = view.slice(page * PER, page * PER + PER).map((i, k) => tile(i, page * PER + k)).join("");
  document.getElementById("pg").textContent = `${view.length ? page + 1 : 0} / ${Math.ceil(view.length / PER)}`;
  stats(); window.scrollTo(0, 0);
}
function stats() {
  const c = f => items.filter(i => has(i, f)).length;
  document.getElementById("stats").textContent =
    `${view.length} shown · flagged: ${c("blank")} blank, ${c("blurry")} blurry, ${c("double")} double`;
}
async function toggle(it, f) {
  const s = new Set(it.flag.split(",").filter(Boolean));
  s.has(f) ? s.delete(f) : s.add(f);
  it.flag = FL.filter(x => s.has(x)).join(",");
  const t = document.querySelector(`.t[data-id="${it.id}"]`);
  if (t) t.outerHTML = tile(it, +t.dataset.n);
  if (cur >= 0) document.getElementById("lbfl").innerHTML = buttons(view[cur]);
  stats();
  await fetch("/api/set", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: it.id, flag: it.flag})});
}
document.getElementById("grid").addEventListener("click", e => {
  const t = e.target.closest(".t"); if (!t) return;
  const b = e.target.closest("button");
  if (b) return toggle(items[+t.dataset.id], b.dataset.f);
  if (e.target.closest(".box")) show(+t.dataset.n);
});
// full-size viewer
function show(n) {
  if (n < 0 || n >= view.length) return;
  cur = n; const it = view[n];
  document.getElementById("lbimg").className = "";
  document.querySelector("#lbimg img").src = `/full/${it.id}`;
  document.getElementById("lbcap").textContent = `${it.rack} ${it.file} · ${it.why}`;
  document.getElementById("lbfl").innerHTML = buttons(it);
  document.getElementById("lb").classList.add("open");
}
function hide() {
  document.getElementById("lb").classList.remove("open");
  const p = Math.floor(cur / PER); cur = -1;
  if (p !== page) { page = p; render(); }
}
document.getElementById("lbimg").onclick = e => {
  const box = document.getElementById("lbimg"), img = box.querySelector("img");
  const fx = e.offsetX / img.clientWidth, fy = e.offsetY / img.clientHeight;
  box.classList.toggle("real");
  if (box.classList.contains("real")) {
    box.scrollLeft = fx * img.naturalWidth - box.clientWidth / 2;
    box.scrollTop = fy * img.naturalHeight - box.clientHeight / 2;
  }
};
document.getElementById("lbfl").onclick = e => { const b = e.target.closest("button"); if (b) toggle(view[cur], b.dataset.f); };
document.getElementById("lbprev").onclick = () => show(cur - 1);
document.getElementById("lbnext").onclick = () => show(cur + 1);
document.getElementById("lbclose").onclick = hide;
document.addEventListener("keydown", e => {
  if (cur < 0) return;
  if (e.key === "Escape") hide();
  else if (e.key === "ArrowLeft") show(cur - 1);
  else if (e.key === "ArrowRight") show(cur + 1);
  else if ("123".includes(e.key)) toggle(view[cur], FL[+e.key - 1]);
});
document.getElementById("prev").onclick = () => { if (page > 0) { page--; render(); } };
document.getElementById("next").onclick = () => { if ((page + 1) * PER < view.length) { page++; render(); } };
document.getElementById("filter").onchange = () => { page = 0; apply(); };
load();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["scan", "faces", "review", "list"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--redo", action="store_true", help="scan: measure again")
    p.add_argument("--md", help="list: write the rescan list (blank, blurry, double) as Markdown to this file")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    {"scan": cmd_scan, "faces": cmd_faces, "review": cmd_review, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
