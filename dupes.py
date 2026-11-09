#!/usr/bin/env python3
"""
Find duplicates across the whole collection: the same slide shot twice, also in
different racks, turned or mirrored.

    python dupes.py scan   FOLDER [FOLDER ...]   embeddings + candidate pairs
    python dupes.py review FOLDER [FOLDER ...]   page: decide per pair
    python dupes.py list   FOLDER [FOLDER ...]   slides marked as duplicate

scan, two stages:
  1. CLIP (ViT-L-14, as quality.py) embedding of every slide, upright (orientation
     from crops.csv) and mirrored; cached in models/embeddings.npz. Pairs with
     cosine >= CLIP_MIN (the better of as-is and mirrored) are candidates. CLIP
     finds similar content, not identical pictures: night forests, sunset series.
  2. pixel check of each candidate: ORB feature points, similarity transform
     (RANSAC, scale 0.8-1.25), then normalised correlation of the aligned images
     (CLAHE, blurred, overlap >= 50 %), as is and mirrored. Pairs with
     NCC >= NCC_MIN go to duplicates.json (root of the drive).
  Pairs are compared among all cached slides, so a rescan is checked against the
  whole collection. New pairs are pre-marked: match >= DUP_AUTO (0.99/1.00 on the
  page) as duplicate keeping the first slide (a), the rest as different; the review
  page only needs a look to correct.
review: local page (http://127.0.0.1:8769), pairs side by side, upright, highest
  NCC first. Buttons: different / keep left / keep right; click an image for full
  size. "keep left" marks the right slide: flag `duplicate`, column `duplicate_of`
  = rack/file of the kept one. The export skips flagged duplicates and removes
  their existing JPEG; the NEF stays.
"""

import argparse
import json
import os
import sys
import threading
import unicodedata
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from tqdm import tqdm

import slide_crop as sc
from upright import folders_rows, full_crop, img_path, load_small, orient

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "models", "embeddings.npz")
PAIRS = os.path.join(ROOT, "duplicates.json")
CLIP_MIN = 0.92
NCC_MIN = 0.6
DUP_AUTO = 0.985     # new pairs from here up (0.99/1.00 on the page) are pre-marked duplicate,
                     # below as different (user's rule; their 25 duplicates matched 0.984-0.999)


def nfc(s):
    """one Unicode form: the drive returns "Vättis" decomposed (a + ¨) or composed
    depending on how the name was obtained"""
    return unicodedata.normalize("NFC", str(s))


def key_of(folder, nef):
    return nfc(f"{os.path.basename(folder.rstrip('/'))}/{nef}")


_folders = {}


def folder_of(key):
    """real folder path of a key (whatever Unicode form the drive uses)"""
    if not _folders:
        _folders.update({nfc(d): os.path.join(ROOT, d) for d in os.listdir(ROOT)})
    return _folders.get(nfc(key.split("/")[0]), os.path.join(ROOT, key.split("/")[0]))


def upright_rgb(folder, nef, row, size):
    return cv2.cvtColor(orient(load_small(img_path(folder, nef), size), row.get("upright") or 1), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- scan

def embed(items):
    """CLIP embeddings (as is, mirrored) of [(folder, nef, row)]"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import open_clip
    import torch
    from PIL import Image
    import quality
    model, _, pre = open_clip.create_model_and_transforms(quality.CLIP_MODEL[0], pretrained=quality.CLIP_MODEL[1],
                                                          cache_dir=quality.CLIP_DIR)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev).eval()
    E, M = [], []
    for i in tqdm(range(0, len(items), 16), desc="embeddings", unit="batch"):
        ims, mir = [], []
        for folder, nef, row in items[i:i + 16]:
            img = upright_rgb(folder, nef, row, 448)
            ims.append(pre(Image.fromarray(img)))
            mir.append(pre(Image.fromarray(np.ascontiguousarray(img[:, ::-1]))))
        with torch.no_grad():
            for lst, out in ((ims, E), (mir, M)):
                e = model.encode_image(torch.stack(lst).to(dev))
                out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.vstack(E), np.vstack(M)


def gray(key):
    folder, nef = folder_of(key), key.split("/")[1]
    row = sc.read_csv(folder)[nef]
    g = cv2.cvtColor(cv2.cvtColor(upright_rgb(folder, nef, row, 800), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)          # exposure differences matter less


def match(a, b):
    """(inliers, ncc) of b aligned onto a"""
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return 0, 0.0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 20:
        return 0, 0.0
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=4)
    if M is None:
        return 0, 0.0
    n = int(inl.sum())
    if not 0.8 < np.hypot(M[0, 0], M[0, 1]) < 1.25:      # same slide: about the same scale
        return n, 0.0
    w = cv2.warpAffine(b, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.full(b.shape, 255, np.uint8), M, (a.shape[1], a.shape[0])) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if mask.mean() < 0.5:                                  # overlap too small
        return n, 0.0
    x = cv2.GaussianBlur(a, (0, 0), 2)[mask].astype(np.float32)
    y = cv2.GaussianBlur(w, (0, 0), 2)[mask].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    return n, float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))


def verify(pair):
    ka, kb, clip = pair
    try:
        a, b = gray(ka), gray(kb)
        n, ncc = max(match(a, b), match(a, np.ascontiguousarray(b[:, ::-1])), key=lambda t: t[1])
    except Exception:
        n, ncc = 0, 0.0
    return ka, kb, clip, n, round(ncc, 4)


def load_pairs():
    return json.load(open(PAIRS)) if os.path.exists(PAIRS) else []


def cmd_scan(args):
    cache = dict(np.load(CACHE, allow_pickle=False)) if os.path.exists(CACHE) else {}
    keys = [nfc(k) for k in cache.get("keys", [])]
    E = cache.get("E", np.zeros((0, 768), np.float32))
    M = cache.get("M", np.zeros((0, 768), np.float32))
    orient_of = dict(zip(keys, cache.get("orient", [])))
    # slides of the given folders that are missing or whose orientation changed
    todo, mine = [], set()
    for folder, rows in folders_rows(args.folders):
        for nef, r in sorted(rows.items()):
            if not img_path(folder, nef) or sc.rejected(r):
                continue
            k = key_of(folder, nef)
            mine.add(k)
            if args.redo or k not in orient_of or str(orient_of[k]) != str(r.get("upright") or 1):
                todo.append((folder, nef, r))
    if todo:
        e, m = embed(todo)
        new = [key_of(f, n) for f, n, _ in todo]
        keep = [i for i, k in enumerate(keys) if k not in set(new)]
        keys = [keys[i] for i in keep] + new
        E = np.vstack([E[keep], e]).astype(np.float32)
        M = np.vstack([M[keep], m]).astype(np.float32)
        orient_of.update({key_of(f, n): str(r.get("upright") or 1) for f, n, r in todo})
        np.savez(CACHE, keys=np.array(keys), E=E, M=M, orient=np.array([orient_of[k] for k in keys]))
    # drop slides that are gone (deleted, rejected) from the comparison
    alive = []
    for i, k in enumerate(keys):
        folder, nef = folder_of(k), k.split("/")[1]
        r = sc.read_csv(folder).get(nef) if os.path.isdir(folder) else None
        if r and not sc.rejected(r) and img_path(folder, nef):
            alive.append(i)
    keys = [keys[i] for i in alive]
    E, M = E[alive], M[alive]
    S = np.maximum(E @ E.T, E @ M.T)
    np.fill_diagonal(S, -1)
    iu = np.triu_indices(len(keys), 1)
    v = S[iu]
    cand = [(keys[iu[0][i]], keys[iu[1][i]], round(float(v[i]), 4)) for i in np.where(v >= CLIP_MIN)[0]
            if keys[iu[0][i]] in mine or keys[iu[1][i]] in mine]
    print(f"{len(keys)} slides compared, {len(cand)} CLIP candidates (>= {CLIP_MIN}) to check")
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(tqdm(ex.map(verify, cand, chunksize=8), total=len(cand), desc="pixel check", unit="pair"))
    old = {(p["a"], p["b"]): p for p in load_pairs()}
    pairs, auto = [], 0
    for ka, kb, clip, n, ncc in res:
        if ncc >= NCC_MIN:
            p = old.get((ka, kb), {"a": ka, "b": kb, "decision": ""})
            p.update(clip=clip, inliers=n, ncc=ncc)
            if not p["decision"]:                          # new pair: pre-mark (keep a = lower name)
                set_decision(p, "keep_a" if ncc >= DUP_AUTO else "different")
                auto += p["decision"] == "keep_a"
            pairs.append(p)
    # keep decided pairs that were not re-checked this time (other racks, or one slide
    # already marked duplicate and so left out of the comparison)
    seen = {(p["a"], p["b"]) for p in pairs}
    pairs += [p for k, p in old.items() if k not in seen and p["decision"]]
    pairs.sort(key=lambda p: -p["ncc"])
    json.dump(pairs, open(PAIRS, "w"), indent=1)
    print(f"{len(pairs)} candidate pairs (pixel match >= {NCC_MIN}) in {PAIRS}; "
          f"{auto} new ones pre-marked duplicate (>= {DUP_AUTO}), the other new ones as different")


def cmd_list(args):
    n = 0
    for folder, rows in folders_rows(args.folders):
        for nef, r in sorted(rows.items()):
            if "duplicate" in (r.get("flag") or ""):
                print(f"{key_of(folder, nef)[:-4]}  duplicate of {r.get('duplicate_of', '')}")
                n += 1
    print(f"{n} slides marked as duplicate")


# ---------------------------------------------------------------- review

def set_decision(p, decision):
    """decision: different / keep_a / keep_b / '' (undo); flags the dropped slide"""
    for key, other in ((p["a"], p["b"]), (p["b"], p["a"])):
        folder, nef = folder_of(key), key.split("/")[1]
        rows = sc.read_csv(folder)
        r = rows[nef]
        fl = [f for f in (r.get("flag") or "").split(",") if f and f != "duplicate"]
        dropped = (decision == "keep_a" and key == p["b"]) or (decision == "keep_b" and key == p["a"])
        if r.get("duplicate_of") == other[:-4] and not dropped:
            r["duplicate_of"] = ""
        elif dropped:
            r["duplicate_of"] = other[:-4]
        if r.get("duplicate_of"):
            fl.append("duplicate")
        r["flag"] = ",".join(fl)
        sc.write_csv(folder, rows)
    p["decision"] = decision


class Handler(BaseHTTPRequestHandler):
    pairs = []
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

    def image(self, key, size):
        if (key, size) not in self.thumbs:
            folder, nef = folder_of(key), key.split("/")[1]
            row = sc.read_csv(folder)[nef]
            if size:
                img = orient(load_small(img_path(folder, nef), size), row.get("upright") or 1)
            else:
                img = full_crop(folder, nef, row)
            self.thumbs[(key, size)] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        return self.thumbs[(key, size)]

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/pairs":
            self.send(200, json.dumps(self.pairs).encode(), "application/json")
        elif self.path.startswith("/img/") or self.path.startswith("/full/"):
            big = self.path.startswith("/full/")
            i, side = self.path.split("/")[2:4]
            p = self.pairs[int(i)]
            self.send(200, self.image(p["a"] if side == "a" else p["b"], 0 if big else 520), "image/jpeg")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/set" or data["decision"] not in ("different", "keep_a", "keep_b", ""):
            return self.send(404, b"not found", "text/plain")
        with self.lock:
            p = self.pairs[int(data["id"])]
            set_decision(p, data["decision"])
            all_ = load_pairs()
            for q in all_:
                if (q["a"], q["b"]) == (p["a"], p["b"]):
                    q["decision"] = p["decision"]
            json.dump(all_, open(PAIRS, "w"), indent=1)
        self.send(200, b'{"ok": true}', "application/json")


def cmd_review(args):
    pairs = load_pairs()
    if not pairs:
        print("no candidate pairs, run scan first")
        return
    Handler.pairs = [dict(p, id=i) for i, p in enumerate(pairs)]
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(pairs)} pairs ({sum(1 for p in pairs if p['decision'])} decided). Review page: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Duplicate review</title>
<style>
  :root { --bg:#16181c; --panel:#1f2329; --text:#e6e6e6; --muted:#9aa3ad; --line:#333a44; --pick:#3ddc84; --drop:#ff4d4d; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px -apple-system,system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:14px; align-items:center; padding:8px 14px; flex-wrap:wrap; }
  .muted { color:var(--muted); }
  button, select { background:#2a3038; color:var(--text); border:1px solid var(--line); border-radius:6px;
                   padding:6px 12px; font-size:14px; cursor:pointer; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(620px, 1fr)); gap:12px; padding:12px; }
  .p { background:#0d0f12; border-radius:6px; padding:6px; border:2px solid transparent; }
  .p.done { border-color:var(--line); }
  .pair { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .v { position:relative; aspect-ratio:1; display:flex; align-items:center; justify-content:center; cursor:zoom-in;
       border:3px solid transparent; border-radius:4px; }
  .v img { max-width:100%; max-height:100%; }
  .v.keep { border-color:var(--pick); } .v.drop { border-color:var(--drop); } .v.drop img { opacity:.35; }
  .cap { font-size:12px; color:var(--muted); padding:4px 2px; display:flex; justify-content:space-between; gap:8px; }
  .ch { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }
  .ch button.on { background:var(--pick); color:#000; font-weight:600; }
  #lb { position:fixed; inset:0; z-index:5; background:#000e; display:none; flex-direction:column; }
  #lb.open { display:flex; }
  #lbimg { flex:1; display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:8px; min-height:0; }
  #lbimg div { display:flex; align-items:center; justify-content:center; min-height:0; }
  #lbimg img { max-width:100%; max-height:100%; }
  #lbbar { display:flex; gap:10px; align-items:center; padding:8px 14px; background:var(--panel); flex-wrap:wrap; }
</style></head><body>
<header>
  <select id="filter"><option value="open">not yet decided</option><option value="all">all pairs</option>
    <option value="dup">marked duplicate</option></select>
  <button id="prev">← page</button><span id="pg"></span><button id="next">page →</button>
  <span class="muted">most similar first · click an image = both full size · saved immediately, nothing deleted</span>
  <span class="muted" id="stats"></span>
</header>
<div id="grid"></div>
<div id="lb"><div id="lbimg"><div><img id="lba"></div><div><img id="lbb"></div></div>
  <div id="lbbar"><button id="lbprev">←</button><button id="lbnext">→</button><span id="lbcap"></span>
    <span class="ch" id="lbch" style="display:flex;gap:6px"></span>
    <span class="muted">1 different · 2 keep left · 3 keep right · ←/→ · Esc</span>
    <button id="lbclose" style="margin-left:auto">✕ close</button></div></div>
<script>
const PER = 10, D = {different: "different", keep_a: "keep left", keep_b: "keep right"};
let pairs = [], view = [], page = 0, cur = -1;
async function load() { pairs = await (await fetch("/api/pairs")).json(); apply(); }
function apply() {
  const f = document.getElementById("filter").value;
  view = pairs.filter(p => f === "all" || (f === "open" && !p.decision) || (f === "dup" && p.decision.startsWith("keep")));
  page = Math.min(page, Math.max(0, Math.ceil(view.length / PER) - 1));
  render();
}
const name = k => k.replace(".NEF", "").replace("/DSC_", " ");
const buttons = p => Object.entries(D).map(([k, v]) => `<button data-d="${k}" class="${p.decision === k ? "on" : ""}">${v}</button>`).join("");
function tile(p) {
  const ca = p.decision === "keep_a" ? "keep" : p.decision === "keep_b" ? "drop" : "";
  const cb = p.decision === "keep_b" ? "keep" : p.decision === "keep_a" ? "drop" : "";
  return `<div class="p ${p.decision ? "done" : ""}" data-id="${p.id}">
    <div class="pair"><div class="v ${ca}"><img src="/img/${p.id}/a" loading="lazy"></div><div class="v ${cb}"><img src="/img/${p.id}/b" loading="lazy"></div></div>
    <div class="cap"><span>${name(p.a)}</span><span>match ${p.ncc.toFixed(2)}</span><span>${name(p.b)}</span></div>
    <div class="ch">${buttons(p)}</div></div>`;
}
function render() {
  document.getElementById("grid").innerHTML = view.slice(page * PER, page * PER + PER).map(tile).join("");
  document.getElementById("pg").textContent = `${view.length ? page + 1 : 0} / ${Math.ceil(view.length / PER)}`;
  stats(); window.scrollTo(0, 0);
}
function stats() {
  const c = d => pairs.filter(p => p.decision === d).length;
  document.getElementById("stats").textContent =
    `${pairs.length} pairs · different ${c("different")} · duplicates ${c("keep_a") + c("keep_b")} · open ${c("")}`;
}
async function decide(p, d) {
  p.decision = p.decision === d ? "" : d;
  const t = document.querySelector(`.p[data-id="${p.id}"]`); if (t) t.outerHTML = tile(p);
  if (cur >= 0) lbshow();
  stats();
  await fetch("/api/set", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: p.id, decision: p.decision})});
}
document.getElementById("grid").addEventListener("click", e => {
  const t = e.target.closest(".p"); if (!t) return;
  const p = pairs[+t.dataset.id], b = e.target.closest("button");
  if (b) return decide(p, b.dataset.d);
  if (e.target.closest(".v")) show(view.indexOf(p));
});
function lbshow() {
  const p = view[cur];
  document.getElementById("lba").src = `/full/${p.id}/a`; document.getElementById("lbb").src = `/full/${p.id}/b`;
  document.getElementById("lbcap").textContent = `${name(p.a)}  ·  ${name(p.b)}  ·  match ${p.ncc.toFixed(2)}`;
  document.getElementById("lbch").innerHTML = buttons(p);
}
function show(n) { if (n < 0 || n >= view.length) return; cur = n; lbshow(); document.getElementById("lb").classList.add("open"); }
function hide() { document.getElementById("lb").classList.remove("open"); page = Math.floor(cur / PER); cur = -1; render(); }
document.getElementById("lbch").onclick = e => { const b = e.target.closest("button"); if (b) decide(view[cur], b.dataset.d); };
document.getElementById("lbprev").onclick = () => show(cur - 1);
document.getElementById("lbnext").onclick = () => show(cur + 1);
document.getElementById("lbclose").onclick = hide;
document.addEventListener("keydown", e => {
  if (cur < 0) return;
  if (e.key === "Escape") hide();
  else if (e.key === "ArrowLeft") show(cur - 1);
  else if (e.key === "ArrowRight") show(cur + 1);
  else if ("123".includes(e.key)) decide(view[cur], ["different", "keep_a", "keep_b"][+e.key - 1]);
});
document.getElementById("prev").onclick = () => { if (page > 0) { page--; render(); } };
document.getElementById("next").onclick = () => { if ((page + 1) * PER < view.length) { page++; render(); } };
document.getElementById("filter").onchange = () => { page = 0; apply(); };
load();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["scan", "review", "list"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--redo", action="store_true", help="scan: compute all embeddings again")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--port", type=int, default=8769)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    {"scan": cmd_scan, "review": cmd_review, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
