#!/usr/bin/env python3
"""
Turn the cropped slides upright, losslessly (EXIF orientation flag; the image data
of cropped/*.jpg stays byte-identical). The orientation is stored in crops.csv
(column `upright`, EXIF value 1-8), so a later export keeps it.

    python upright.py guess     FOLDER [FOLDER ...]   automatic guess for every slide
    python upright.py textcheck FOLDER [FOLDER ...]   read text on every slide (column upright_text)
    python upright.py review    FOLDER [FOLDER ...]   grid page to check and correct

guess: photos -> image model (upright_model.py, ResNeXt50 from
github.com/ternaus/check_orientation, rotation only); slides with readable text ->
macOS text recognition in all 8 orientations, scored by real words
(upright_ocr.best8; also finds mirrored slides) when the text is clear (text_ok():
TEXT_MIN_SCORE, TEXT_MARGIN).
Slides set by hand (upright_by = manual) are kept unless --redo.

textcheck: the same text reading for every slide, stored as upright_text
("<exif> <score> <second best score> <words>"), without changing the orientation.
The review page lists the slides where clear text disagrees with the orientation
("text disagrees"): mirrored or turned slides the image model can't see.

review: local page (http://127.0.0.1:8766) with the slides as they will be shown.
Click = turn 90 deg clockwise, Alt+click = mirror. Every change is saved at once
(crops.csv + JPEG). "Page OK" marks the page as checked (upright_checked = 1).
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

# (mirrored left-right first, then rotated clockwise) <-> EXIF orientation
TO_EXIF = {(False, 0): 1, (False, 90): 6, (False, 180): 3, (False, 270): 8,
           (True, 0): 2, (True, 90): 7, (True, 180): 4, (True, 270): 5}
FROM_EXIF = {v: k for k, v in TO_EXIF.items()}
TEXT_MIN_SCORE = 9           # at least one dictionary word (3 x 3 letters)
TEXT_MARGIN = 1.5            # and clearly better than the best other orientation


def text_ok(score, second):
    return score >= TEXT_MIN_SCORE and score >= TEXT_MARGIN * second


def text_says(r):
    """(exif, score, words) from the upright_text column if the text is clear, else None"""
    parts = (r.get("upright_text") or "").split(" ", 3)
    if len(parts) < 3 or not text_ok(int(parts[1]), int(parts[2])):
        return None
    return int(parts[0]), int(parts[1]), parts[3] if len(parts) > 3 else ""


def jpg_path(folder, nef):
    return os.path.join(folder, "cropped", os.path.splitext(nef)[0] + ".jpg")


def preview_path(folder, nef):
    """small crop in the camera look, written by pipeline.py analyze (no EXIF orientation)"""
    return os.path.join(folder, "_preview", os.path.splitext(nef)[0] + ".jpg")


def img_path(folder, nef):
    """the image the analysis and the review pages read: the preview if there is one,
    otherwise the exported JPEG; None if neither exists"""
    for p in (preview_path(folder, nef), jpg_path(folder, nef)):
        if os.path.exists(p):
            return p
    return None


def load_small(path, long_side):
    """BGR image as stored (EXIF orientation ignored), about `long_side` px long;
    big JPEGs are decoded reduced (fast)"""
    from PIL import Image
    with Image.open(path) as im:
        size = max(im.size)
    flag = cv2.IMREAD_COLOR
    for f, k in ((cv2.IMREAD_REDUCED_COLOR_8, 8), (cv2.IMREAD_REDUCED_COLOR_4, 4), (cv2.IMREAD_REDUCED_COLOR_2, 2)):
        if size / k >= long_side:
            flag = f
            break
    img = cv2.imread(path, flag | cv2.IMREAD_IGNORE_ORIENTATION)
    k = long_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img


def orient(img, o):
    """apply EXIF orientation o to an image array (as a viewer would)"""
    mirror, rot = FROM_EXIF[int(o or 1)]
    if mirror:
        img = img[:, ::-1]
    return np.ascontiguousarray(np.rot90(img, -rot // 90))


def full_crop(folder, nef, row):
    """full-size crop in the camera look, upright, BGR: the exported JPEG if there is
    one, else cut from the camera JPG / embedded JPEG (slow, for zooming in)"""
    p = jpg_path(folder, nef)
    if os.path.exists(p):
        return cv2.imread(p)                         # imread applies the orientation
    rgb, _ = sc.load_image(os.path.join(folder, nef))
    return cv2.cvtColor(orient(sc.crop_image(rgb, row, 24), row.get("upright")), cv2.COLOR_RGB2BGR)


def load_rgb(path, long_side=1500):
    return cv2.cvtColor(load_small(path, long_side), cv2.COLOR_BGR2RGB)


def ocr_job(path):
    """(exif, score, second, words) of the best text reading; 2000 px, so small signs
    are read too"""
    import upright_ocr
    try:
        m, r, score, words, second = upright_ocr.best8(load_rgb(path, 2000))
        return TO_EXIF[(m, r)], score, second, words
    except Exception:
        return 1, 0, 0, ""


def text_column(res):
    exif, score, second, words = res
    return f"{exif} {score} {second} {words}".strip()


def folders_rows(folders):
    out = []
    for folder in folders:
        folder = folder.rstrip("/")
        if os.path.exists(os.path.join(folder, sc.CSV_NAME)):
            out.append((folder, sc.read_csv(folder)))
    return out


# ---------------------------------------------------------------- guess

def cmd_guess(args):
    import upright_model
    todo = []
    for folder, rows in folders_rows(args.folders):
        for f, r in sorted(rows.items()):
            if not img_path(folder, f) or sc.rejected(r):
                continue
            if r.get("upright_by") == "manual" and not args.redo:
                continue
            if r.get("upright") and not args.redo:    # already guessed
                continue
            todo.append((folder, f))
    print(f"{len(todo)} slides to guess")
    model, dev = upright_model.load_model()
    probs = {}
    for i in tqdm(range(0, len(todo), 32), desc="image model", unit="batch"):
        part = todo[i:i + 32]
        imgs = [load_rgb(img_path(fo, f), 750) for fo, f in part]
        for key, p in zip(part, upright_model.predict(model, dev, imgs)):
            probs[key] = p
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=100) as ex:   # fresh workers: bounded memory
        ocr = dict(zip(todo, tqdm(ex.map(ocr_job, [img_path(fo, f) for fo, f in todo], chunksize=4),
                                  total=len(todo), desc="text", unit="img")))
    by_folder = {}
    for key in todo:
        exif, score, second, _ = ocr[key]
        p = probs[key]
        if text_ok(score, second):
            orient, by, conf = exif, "text", min(1.0, score / (score + second + 1e-9))
        else:
            k = int(np.argmax(p))
            orient, by, conf = TO_EXIF[(False, k * 90)], "model", float(p.max())
        by_folder.setdefault(key[0], []).append((key[1], orient, by, conf, text_column(ocr[key])))
    changed = 0
    for folder, items in tqdm(by_folder.items(), desc="write", unit="rack"):
        rows = sc.read_csv(folder)
        for f, orient, by, conf, txt in items:
            rows[f].update(upright=str(orient), upright_by=by, upright_conf=f"{conf:.2f}", upright_checked="",
                           upright_text=txt)
            if os.path.exists(jpg_path(folder, f)):  # already exported: patch it (lossless)
                changed += sc.set_jpeg_orientation(jpg_path(folder, f), orient)
        sc.write_csv(folder, rows)
    print(f"done: {sum(1 for v in by_folder.values() for x in v if x[2] == 'text')} by text, "
          f"{sum(1 for v in by_folder.values() for x in v if x[2] == 'model')} by model, "
          f"{changed} JPEGs turned; {sum(1 for v in by_folder.values() for x in v if x[3] < 0.9)} uncertain (< 0.9)")


def cmd_textcheck(args):
    """read text on every slide; saved after each rack, so an interrupted run continues"""
    todo = [(folder, f) for folder, rows in folders_rows(args.folders) for f, r in sorted(rows.items())
            if img_path(folder, f) and not sc.rejected(r) and (args.redo or not r.get("upright_text"))]
    print(f"{len(todo)} slides to read")
    disagree, pending = 0, {}

    def save(folder):
        nonlocal disagree
        rows = sc.read_csv(folder)
        for f, txt in pending.pop(folder).items():
            rows[f]["upright_text"] = txt
            t = text_says(rows[f])
            disagree += bool(t and str(t[0]) != (rows[f].get("upright") or "1"))
        sc.write_csv(folder, rows)

    # (a progress bar created before the pool hangs the worker start-up: tqdm makes a
    # multiprocessing lock; so the bar wraps the results, as in guess)
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=100) as ex:   # fresh workers: bounded memory
        results = ex.map(ocr_job, [img_path(fo, f) for fo, f in todo], chunksize=2)
        for (folder, f), x in tqdm(zip(todo, results), total=len(todo), desc="text", unit="img"):
            if pending and folder not in pending:          # previous rack complete
                save(next(iter(pending)))
            pending.setdefault(folder, {})[f] = text_column(x)
        while pending:
            save(next(iter(pending)))
    print(f"{len(todo)} slides read; clear text disagrees with the orientation on {disagree} "
          "(review page, filter \"text disagrees\")")


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
            out = [{k: it[k] for k in ("id", "rack", "file", "upright", "by", "conf", "checked", "text")}
                   for it in self.items]
            self.send(200, json.dumps(out).encode(), "application/json")
        elif self.path.startswith("/thumb/"):
            i = int(self.path[7:])
            if i not in self.thumbs:
                it = self.items[i]
                img = load_small(img_path(it["folder"], it["nef"]), 320)
                self.thumbs[i] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            self.send(200, self.thumbs[i], "image/jpeg")
        elif self.path.startswith("/big/"):                  # full-size view, as stored (CSS turns it)
            it = self.items[int(self.path[5:])]
            img = load_small(img_path(it["folder"], it["nef"]), 2400)
            self.send(200, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), "image/jpeg")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            if self.path == "/api/set":
                it = self.items[int(data["id"])]
                orient = int(data["upright"])
                rows = sc.read_csv(it["folder"])
                rows[it["nef"]].update(upright=str(orient), upright_by="manual", upright_checked="1")
                sc.write_csv(it["folder"], rows)
                if os.path.exists(jpg_path(it["folder"], it["nef"])):
                    sc.set_jpeg_orientation(jpg_path(it["folder"], it["nef"]), orient)
                it.update(upright=orient, by="manual", checked=True)
            elif self.path == "/api/checked":
                ids = [int(i) for i in data["ids"]]
                for folder in {self.items[i]["folder"] for i in ids}:
                    rows = sc.read_csv(folder)
                    for i in ids:
                        it = self.items[i]
                        if it["folder"] == folder:
                            rows[it["nef"]]["upright_checked"] = "1"
                            it["checked"] = True
                    sc.write_csv(folder, rows)
            else:
                return self.send(404, b"not found", "text/plain")
        self.send(200, b'{"ok": true}', "application/json")


def cmd_review(args):
    items = []
    for folder, rows in folders_rows(args.folders):
        for f, r in sorted(rows.items()):
            if img_path(folder, f) and not sc.rejected(r):
                items.append({"id": len(items), "folder": folder, "nef": f, "rack": os.path.basename(folder),
                              "file": f[:-4], "upright": int(r.get("upright") or 1),
                              "by": r.get("upright_by") or "", "conf": float(r.get("upright_conf") or 0),
                              "checked": r.get("upright_checked") == "1",
                              "text": (lambda t: {"exif": t[0], "words": t[2]} if t else None)(text_says(r))})
    Handler.items = items
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(items)} slides. Review page: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Upright review</title>
<style>
  :root { --bg:#16181c; --panel:#1f2329; --text:#e6e6e6; --muted:#9aa3ad; --line:#333a44;
          --warn:#ffb020; --text-c:#5aa9ff; --man:#3ddc84; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px -apple-system,system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:14px; align-items:center; padding:8px 14px; flex-wrap:wrap; }
  header .muted { color:var(--muted); }
  button, select { background:#2a3038; color:var(--text); border:1px solid var(--line); border-radius:6px;
                   padding:6px 12px; font-size:14px; cursor:pointer; }
  #ok { background:#2f6f4a; color:#fff; font-weight:600; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(190px, 1fr)); gap:8px; padding:12px; }
  .t { background:#0d0f12; border:2px solid transparent; border-radius:6px; overflow:hidden; cursor:pointer;
       user-select:none; }
  .t.unsure { border-color:var(--warn); } .t.text { border-color:var(--text-c); } .t.manual { border-color:var(--man); }
  .box { width:100%; aspect-ratio:1; display:flex; align-items:center; justify-content:center; overflow:hidden; }
  .box img { max-width:100%; max-height:100%; image-orientation:none; transition:transform .12s; }
  .cap { font-size:11px; color:var(--muted); padding:3px 6px; display:flex; justify-content:space-between; }
  .chk::after { content:" ✓"; color:var(--man); }
  .legend span { margin-right:10px; } .legend b { display:inline-block; width:10px; height:10px; border-radius:2px; }
  .t { position:relative; }
  .zoom { position:absolute; top:4px; right:4px; background:#000a; color:#fff; border-radius:4px; padding:1px 6px;
          font-size:14px; cursor:zoom-in; z-index:1; }
  #lb { position:fixed; inset:0; z-index:5; background:#000e; display:none; flex-direction:column; }
  #lb.open { display:flex; }
  #lbimg { flex:1; display:flex; align-items:center; justify-content:center; overflow:hidden; }
  #lbimg img { image-orientation:none; max-width:96vw; max-height:88vh; }
  #lbimg img.side { max-width:88vh; max-height:96vw; }
  #lbbar { display:flex; gap:10px; align-items:center; padding:8px 14px; background:var(--panel); flex-wrap:wrap; }
</style></head><body>
<header>
  <select id="filter">
    <option value="all">all slides</option>
    <option value="unsure">uncertain only</option>
    <option value="textdiff">text disagrees</option>
    <option value="unchecked">not yet checked</option>
  </select>
  <select id="rack"></select>
  <button id="prev">← page</button><span id="pg"></span><button id="next">page →</button>
  <button id="ok">Page OK → next</button>
  <span class="muted">Click = turn 90° clockwise · Alt+click = mirror · 🔍 = full size · saved immediately</span>
  <span class="legend muted"><span><b style="background:var(--warn)"></b> uncertain</span>
  <span><b style="background:var(--text-c)"></b> decided by text</span><span><b style="background:var(--man)"></b> set by you</span></span>
  <span class="muted" id="stats"></span>
</header>
<div id="grid"></div>
<div id="lb">
  <div id="lbimg"><img></div>
  <div id="lbbar"><button id="lbprev">←</button><button id="lbnext">→</button>
    <button id="lbturn">turn 90° (r)</button><button id="lbmirror">mirror (m)</button><span id="lbcap"></span>
    <span class="muted">←/→ previous/next · Esc close · saved immediately</span>
    <button id="lbclose" style="margin-left:auto">✕ close</button></div>
</div>
<script>
const PER = 60;
const TO = {"0,0":1,"0,90":6,"0,180":3,"0,270":8,"1,0":2,"1,90":7,"1,180":4,"1,270":5};
const FROM = Object.fromEntries(Object.entries(TO).map(([k, v]) => [v, k.split(",").map(Number)]));
let items = [], view = [], page = 0, cur = -1;

const css = o => { const [m, r] = FROM[o]; return `rotate(${r}deg) scaleX(${m ? -1 : 1})`; };
async function load() {
  items = await (await fetch("/api/items")).json();
  const racks = [...new Set(items.map(i => i.rack))];
  document.getElementById("rack").innerHTML = `<option value="">all racks</option>` +
    racks.map(r => `<option>${r}</option>`).join("");
  apply();
}
function apply() {
  const f = document.getElementById("filter").value, rk = document.getElementById("rack").value;
  view = items.filter(i => (!rk || i.rack === rk) &&
    (f === "all" || (f === "unsure" && i.conf < 0.9 && i.by !== "manual") || (f === "unchecked" && !i.checked) ||
    (f === "textdiff" && i.text && i.text.exif !== i.upright)));
  page = Math.min(page, Math.max(0, Math.ceil(view.length / PER) - 1));
  render();
}
function cls(i) { return i.by === "manual" ? "manual" : i.by === "text" ? "text" : i.conf < 0.9 ? "unsure" : ""; }
function render() {
  const part = view.slice(page * PER, page * PER + PER);
  document.getElementById("grid").innerHTML = part.map(i =>
    `<div class="t ${cls(i)}" data-id="${i.id}"><span class="zoom" title="full size">🔍</span><div class="box"><img src="/thumb/${i.id}" style="transform:${css(i.upright)}"></div>
     <div class="cap"><span class="${i.checked ? "chk" : ""}">${i.rack} ${i.file.slice(4)}</span><span>${i.by === "manual" ? "you" : (i.conf * 100).toFixed(0) + "%"}</span></div>
     ${i.text && i.text.exif !== i.upright ? `<div class="cap" style="color:#5aa9ff">text says ${i.text.exif}: ${i.text.words.slice(0, 40)}</div>` : ""}</div>`).join("");
  document.getElementById("pg").textContent = `${view.length ? page + 1 : 0} / ${Math.ceil(view.length / PER)}`;
  const done = items.filter(i => i.checked).length;
  document.getElementById("stats").textContent = `checked ${done} of ${items.length}`;
  window.scrollTo(0, 0);
}
async function turn(it, mirror) {                 // turn 90° clockwise, or mirror left-right as shown
  let [m, r] = FROM[it.upright];
  if (mirror) { m = 1 - m; r = (360 - r) % 360; } else r = (r + 90) % 360;
  it.upright = TO[`${m},${r}`]; it.by = "manual"; it.checked = true;
  const t = document.querySelector(`.t[data-id="${it.id}"]`);
  if (t) {
    t.querySelector(".box img").style.transform = css(it.upright); t.className = "t manual";
    t.querySelector(".cap span:last-child").textContent = "you";
    t.querySelector(".cap span").className = "chk";
  }
  if (cur >= 0) lbshow();
  await fetch("/api/set", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: it.id, upright: it.upright})});
}
document.getElementById("grid").addEventListener("click", e => {
  const t = e.target.closest(".t"); if (!t) return;
  const it = items[+t.dataset.id];
  if (e.target.closest(".zoom")) return show(view.indexOf(it));
  turn(it, e.altKey);
});
// full-size view
function lbshow() {
  const it = view[cur], img = document.querySelector("#lbimg img");
  if (!img.src.endsWith(`/big/${it.id}`)) img.src = `/big/${it.id}`;
  img.style.transform = css(it.upright);
  img.className = FROM[it.upright][1] % 180 ? "side" : "";
  document.getElementById("lbcap").textContent = `${it.rack} ${it.file}` +
    (it.text ? ` · text: ${it.text.words.slice(0, 50)}` : "");
}
function show(n) {
  if (n < 0 || n >= view.length) return;
  cur = n; lbshow(); document.getElementById("lb").classList.add("open");
}
function hide() {
  document.getElementById("lb").classList.remove("open");
  const p = Math.floor(cur / PER); cur = -1;
  if (p !== page) { page = p; render(); }
}
document.getElementById("lbturn").onclick = () => turn(view[cur], false);
document.getElementById("lbmirror").onclick = () => turn(view[cur], true);
document.getElementById("lbprev").onclick = () => show(cur - 1);
document.getElementById("lbnext").onclick = () => show(cur + 1);
document.getElementById("lbclose").onclick = hide;
document.addEventListener("keydown", e => {
  if (cur < 0) return;
  if (e.key === "Escape") hide();
  else if (e.key === "ArrowLeft") show(cur - 1);
  else if (e.key === "ArrowRight") show(cur + 1);
  else if (e.key === "r") turn(view[cur], false);
  else if (e.key === "m") turn(view[cur], true);
});
document.getElementById("ok").onclick = async () => {
  const ids = view.slice(page * PER, page * PER + PER).map(i => i.id);
  await fetch("/api/checked", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ids})});
  ids.forEach(id => items[id].checked = true);
  if ((page + 1) * PER < view.length) page++;
  document.getElementById("filter").value === "unchecked" ? apply() : render();
};
document.getElementById("prev").onclick = () => { if (page > 0) { page--; render(); } };
document.getElementById("next").onclick = () => { if ((page + 1) * PER < view.length) { page++; render(); } };
document.getElementById("filter").onchange = () => { page = 0; apply(); };
document.getElementById("rack").onchange = () => { page = 0; apply(); };
load();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["guess", "textcheck", "review"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--redo", action="store_true", help="guess again also for slides set by hand")
    p.add_argument("--jobs", type=int, default=8, help="parallel text recognition processes")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    {"guess": cmd_guess, "textcheck": cmd_textcheck, "review": cmd_review}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
