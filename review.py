#!/usr/bin/env python3
"""
Review page for flagged slides (status "check: ...") in crops.csv.

    python review.py FOLDER [FOLDER ...]        e.g.  review.py /path/to/slides/*/

Opens a page in the browser (served locally, http://127.0.0.1:8765). For every
flagged slide it shows the current crop (red) and a rack-guided suggestion
(green, slide_crop.detect_guided). Choose one, or drag its corners / move it,
then save: the box is written to that folder's crops.csv with status "manual"
(slide_crop.py detect never overwrites manual rows). Run export afterwards.

Boxes are the film window (edge to edge); the dashed line is the final crop
inside the export's safety margin.
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rawpy
from tqdm import tqdm

import slide_crop as sc

BOX_KEYS = ("cx", "cy", "w", "h", "angle")


def box_of(row):
    return {k: float(row[k]) for k in BOX_KEYS} if row else None


def collect(folders):
    items = []
    for folder in folders:
        folder = folder.rstrip("/")
        if not os.path.exists(os.path.join(folder, sc.CSV_NAME)):
            continue
        rows = sc.read_csv(folder)
        typical = sc.rack_typical_sizes(rows)
        for f, r in sorted(rows.items()):
            if r["status"].startswith("check"):
                items.append({"folder": folder, "rack": os.path.basename(folder), "file": f,
                              "status": r["status"], "red": box_of(r), "typical": typical,
                              "img_w": int(r["img_w"]), "img_h": int(r["img_h"])})
    return items


def suggest(item):
    try:
        return sc.detect_guided(os.path.join(item["folder"], item["file"]), item["typical"])
    except Exception:
        return None


def orientation(w, h):
    a = max(w, h) / min(w, h)
    return "square" if a <= 1.08 else ("landscape" if w > h else "portrait")


class Handler(BaseHTTPRequestHandler):
    items = []
    lock = threading.Lock()
    thumbs = {}

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
            out = [{"id": i, "rack": it["rack"], "file": it["file"], "status": it["status"],
                    "red": it["red"], "green": box_of(it.get("green")),
                    "img_w": it["img_w"], "img_h": it["img_h"], "saved": it.get("saved")}
                   for i, it in enumerate(self.items)]
            self.send(200, json.dumps({"items": out, "margin": 24}).encode(), "application/json")
        elif self.path.startswith("/img/"):
            i = int(self.path[5:])
            if i not in self.thumbs:
                it = self.items[i]
                with rawpy.imread(os.path.join(it["folder"], it["file"])) as raw:
                    self.thumbs[i] = bytes(raw.extract_thumb().data)
            self.send(200, self.thumbs[i], "image/jpeg")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self.send(404, b"not found", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        it = self.items[int(data["id"])]
        box = {k: round(float(data["box"][k]), 3 if k == "angle" else 2) for k in BOX_KEYS}
        with self.lock:
            rows = sc.read_csv(it["folder"])
            base = dict(it["green"]) if data["choice"] == "green" and it.get("green") else dict(rows[it["file"]])
            base = {k: str(v) for k, v in base.items()}
            base.update({k: str(v) for k, v in box.items()})
            base.update(file=it["file"], status="manual", orientation=orientation(box["w"], box["h"]),
                        img_w=str(it["img_w"]), img_h=str(it["img_h"]))
            rows[it["file"]] = base
            sc.write_csv(it["folder"], rows)
            it["saved"] = data["choice"]
        self.send(200, b'{"ok": true}', "application/json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = p.parse_args()
    items = collect(args.folders)
    if not items:
        print("no flagged slides")
        return
    print(f"{len(items)} flagged slides, computing suggestions ...")
    with ProcessPoolExecutor() as ex:
        for it, g in zip(items, tqdm(ex.map(suggest, items), total=len(items), unit="img")):
            it["green"] = g
    Handler.items = items
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"review page: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Slide review</title>
<style>
  :root { --bg:#16181c; --panel:#1f2329; --text:#e6e6e6; --muted:#9aa3ad; --red:#ff4d4d;
          --green:#3ddc84; --edit:#ffd43b; --line:#333a44; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px -apple-system,system-ui,sans-serif;
         display:grid; grid-template-columns:260px 1fr 300px; height:100vh; overflow:hidden; }
  #list { overflow:auto; border-right:1px solid var(--line); }
  #list div { padding:7px 10px; border-bottom:1px solid var(--line); cursor:pointer; }
  #list div.cur { background:#2c333d; }
  #list .st { color:var(--muted); font-size:12px; }
  #list .done::before { content:"✓ "; color:var(--green); }
  #stage { position:relative; }
  canvas#c { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
  #side { padding:14px; border-left:1px solid var(--line); background:var(--panel); overflow:auto; }
  h2 { font-size:15px; margin:0 0 4px; } .muted { color:var(--muted); font-size:12px; }
  button { display:block; width:100%; margin:6px 0; padding:9px; border-radius:6px; border:1px solid var(--line);
           background:#2a3038; color:var(--text); font-size:14px; cursor:pointer; text-align:left; }
  button.sel { outline:2px solid currentColor; }
  #bRed { color:var(--red); } #bGreen { color:var(--green); } #bEdit { color:var(--edit); }
  #bSave { background:#2f6f4a; color:#fff; text-align:center; font-weight:600; }
  .row { display:flex; gap:6px; } .row button { text-align:center; }
  canvas#loupe { width:270px; height:270px; border:1px solid var(--line); margin-top:10px; background:#000; }
  kbd { background:#333; border-radius:3px; padding:0 4px; font-size:11px; }
  #msg { min-height:18px; color:var(--green); margin-top:6px; }
</style></head><body>
<div id="list"></div>
<div id="stage"><canvas id="c"></canvas></div>
<div id="side">
  <h2 id="title"></h2><div class="muted" id="reason"></div>
  <p class="muted" id="progress"></p>
  <button id="bRed">1 &nbsp;Red: current crop</button>
  <button id="bGreen">2 &nbsp;Green: suggestion</button>
  <button id="bEdit" disabled>Yellow: edited</button>
  <div class="row"><button id="rotL">⟲ 0.2°</button><button id="rotR">⟳ 0.2°</button></div>
  <button id="bSave">Save &amp; next <kbd>Enter</kbd></button>
  <div class="row"><button id="prev">← prev</button><button id="next">skip →</button></div>
  <div id="msg"></div>
  <canvas id="loupe" width="270" height="270"></canvas>
  <p class="muted">Drag a corner of the active box to resize, drag inside to move.
  The magnifier shows full resolution around the mouse. Dashed = final crop inside the
  safety margin. <kbd>1</kbd> <kbd>2</kbd> choose, <kbd>Esc</kbd> undo edits,
  <kbd>←</kbd> <kbd>→</kbd> navigate.</p>
</div>
<script>
let items = [], MARGIN = 24, cur = 0, img = null, off = 8;
let choice = "red", edit = null, view = {s: 1, ox: 0, oy: 0}, drag = null, mouse = null;
const C = document.getElementById("c"), X = C.getContext("2d");
const L = document.getElementById("loupe"), LX = L.getContext("2d");
const col = {red: "#ff4d4d", green: "#3ddc84", edit: "#ffd43b"};

function corners(b, m = 0) {        // same as slide_crop.corners: raw-image coordinates
  const w = b.w - 2 * m, h = b.h - 2 * m, t = b.angle * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  return [[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]].map(([x, y]) => [b.cx + c*x - s*y, b.cy + s*x + c*y]);
}
const active = () => choice === "edit" ? edit : items[cur][choice];
const toScreen = ([x, y]) => [(x - off) * view.s + view.ox, (y - off) * view.s + view.oy];
const toRaw = (px, py) => [(px - view.ox) / view.s + off, (py - view.oy) / view.s + off];

async function load() {
  const r = await (await fetch("/api/items")).json();
  items = r.items; MARGIN = r.margin;
  renderList(); show(0);
}
function renderList() {
  document.getElementById("list").innerHTML = items.map((it, i) =>
    `<div class="${i === cur ? "cur" : ""}" onclick="show(${i})"><span class="${it.saved ? "done" : ""}">${it.rack}</span><br>
     <span class="st">${it.file.replace(".NEF", "")} · ${it.saved ? "saved (" + it.saved + ")" : it.status.replace("check: ", "")}</span></div>`).join("");
}
function show(i) {
  cur = (i + items.length) % items.length; edit = null;
  const it = items[cur];
  choice = it.green ? "green" : "red";
  document.getElementById("title").textContent = `${it.rack} / ${it.file}`;
  document.getElementById("reason").textContent = it.status;
  document.getElementById("progress").textContent =
    `${cur + 1} of ${items.length} · saved ${items.filter(x => x.saved).length}`;
  document.getElementById("bGreen").disabled = !it.green;
  document.getElementById("msg").textContent = "";
  renderList(); buttons();
  img = new Image();
  img.onload = () => { off = (it.img_w - img.naturalWidth) / 2; draw(); };
  img.src = "/img/" + cur;
  draw();
}
function buttons() {
  for (const [id, c] of [["bRed", "red"], ["bGreen", "green"], ["bEdit", "edit"]])
    document.getElementById(id).classList.toggle("sel", choice === c);
  document.getElementById("bEdit").disabled = !edit;
}
function quad(ctx, pts, color, width, dash = []) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath(); pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.stroke();
  ctx.restore();
}
function boxes() {
  const it = items[cur], out = [];
  for (const c of ["red", "green"]) if (it[c]) out.push([c, it[c]]);
  if (edit) out.push(["edit", edit]);
  return out;
}
function draw() {
  const r = C.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  C.width = r.width * dpr; C.height = r.height * dpr; X.setTransform(dpr, 0, 0, dpr, 0, 0);
  X.fillStyle = "#000"; X.fillRect(0, 0, r.width, r.height);
  if (!img || !img.complete || !img.naturalWidth) return;
  const s = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
  view = {s, ox: (r.width - img.naturalWidth * s) / 2, oy: (r.height - img.naturalHeight * s) / 2};
  X.drawImage(img, view.ox, view.oy, img.naturalWidth * s, img.naturalHeight * s);
  for (const [c, b] of boxes()) {
    const on = c === choice;
    quad(X, corners(b).map(toScreen), col[c], on ? 2.5 : 1.2);
    if (on) {
      quad(X, corners(b, MARGIN).map(toScreen), col[c], 1, [5, 4]);
      for (const p of corners(b).map(toScreen)) { X.fillStyle = col[c]; X.fillRect(p[0] - 5, p[1] - 5, 10, 10); }
    }
  }
  drawLoupe();
}
function drawLoupe() {
  LX.fillStyle = "#000"; LX.fillRect(0, 0, 270, 270);
  if (!mouse || !img.naturalWidth) return;
  const Z = 1.2, half = 135 / Z;                     // 1.2 screen px per raw px
  const [rx, ry] = toRaw(...mouse), sx = rx - off - half, sy = ry - off - half;
  LX.drawImage(img, sx, sy, 2 * half, 2 * half, 0, 0, 270, 270);
  const T = ([x, y]) => [(x - off - sx) * Z, (y - off - sy) * Z];
  for (const [c, b] of boxes()) {
    quad(LX, corners(b).map(T), col[c], c === choice ? 2 : 1);
    if (c === choice) quad(LX, corners(b, MARGIN).map(T), col[c], 1, [5, 4]);
  }
}
function rot(b) { const t = b.angle * Math.PI / 180; return [Math.cos(t), Math.sin(t)]; }
function startEdit() {
  if (choice !== "edit") { edit = {...active()}; choice = "edit"; buttons(); }
}
C.addEventListener("mousedown", e => {
  const b = active(); if (!b) return;
  const p = [e.offsetX, e.offsetY], cs = corners(b).map(toScreen);
  const k = cs.findIndex(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 12);
  startEdit();
  if (k >= 0) drag = {type: "corner", k};
  else drag = {type: "move", start: toRaw(...p), c0: [edit.cx, edit.cy]};
  draw();
});
addEventListener("mouseup", () => drag = null);
C.addEventListener("mouseleave", () => { mouse = null; drawLoupe(); });
C.addEventListener("mousemove", e => {
  mouse = [e.offsetX, e.offsetY];
  if (drag && edit) {
    const [x, y] = toRaw(...mouse), [c, s] = rot(edit);
    if (drag.type === "move") {
      edit.cx = drag.c0[0] + x - drag.start[0]; edit.cy = drag.c0[1] + y - drag.start[1];
    } else {
      const sg = [[-1,-1],[1,-1],[1,1],[-1,1]][drag.k];
      const q = corners(edit)[(drag.k + 2) % 4];                 // opposite corner stays
      const dx = x - q[0], dy = y - q[1];
      const u = c * dx + s * dy, v = -s * dx + c * dy;           // into the box frame
      if (u * sg[0] > 20 && v * sg[1] > 20) {
        edit.w = Math.abs(u); edit.h = Math.abs(v);
        edit.cx = q[0] + (c * u - s * v) / 2; edit.cy = q[1] + (s * u + c * v) / 2;
      }
    }
    draw();
  } else drawLoupe();
});
function choose(c) { if (c === "green" && !items[cur].green) return; choice = c; buttons(); draw(); }
function rotate(d) { if (!active()) return; startEdit(); edit.angle += d; draw(); }
async function save() {
  const b = active(); if (!b) return;
  const r = await fetch("/api/save", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: items[cur].id, choice, box: b})});
  if (r.ok) { items[cur].saved = choice; document.getElementById("msg").textContent = "saved";
              const nxt = items.findIndex((x, i) => i > cur && !x.saved);
              show(nxt >= 0 ? nxt : cur + 1); }
  else document.getElementById("msg").textContent = "save failed";
}
document.getElementById("bRed").onclick = () => choose("red");
document.getElementById("bGreen").onclick = () => choose("green");
document.getElementById("bEdit").onclick = () => choose("edit");
document.getElementById("rotL").onclick = () => rotate(-0.2);
document.getElementById("rotR").onclick = () => rotate(0.2);
document.getElementById("bSave").onclick = save;
document.getElementById("prev").onclick = () => show(cur - 1);
document.getElementById("next").onclick = () => show(cur + 1);
addEventListener("keydown", e => {
  if (e.key === "1") choose("red");
  else if (e.key === "2") choose("green");
  else if (e.key === "Enter") save();
  else if (e.key === "ArrowRight") show(cur + 1);
  else if (e.key === "ArrowLeft") show(cur - 1);
  else if (e.key === "Escape") { edit = null; choice = items[cur].green ? "green" : "red"; buttons(); draw(); }
});
addEventListener("resize", draw);
load();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
