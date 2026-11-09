#!/usr/bin/env python3
"""
The whole workflow for new or rescanned racks: analyse first, decide, export once.

    python pipeline.py run     FOLDER [FOLDER ...] [--auto]   analyse (+ decide + export with --auto)
    python pipeline.py analyze FOLDER [FOLDER ...]            only write crops.csv and previews
    python pipeline.py decide  FOLDER [FOLDER ...] --auto     accept the correction proposals
    python pipeline.py export  FOLDER [FOLDER ...]            write every JPEG once
    python pipeline.py status  FOLDER [FOLDER ...]            what is done per rack

analyze (writes only crops.csv and FOLDER/_preview/, never cropped/):
  1. detect the crops (slide_crop.py detect)
  2. previews: each crop in the camera look, 1600 px, in _preview/ (column `preview`
     = the crop it was made from; a changed crop, e.g. after review.py, makes a new
     preview and clears that slide's measurements and non-manual orientation)
  3. measurements: blur, blank, colour cast from the full-size crop while making the
     preview (same scale as on exported JPEGs, so quality.REF applies), exposure from
     the NEF, CLIP double and warm-scene scores on the previews
  4. orientation guess on the previews (upright.py guess), then face brightness on the
     upright previews (quality.py faces)
  5. duplicates against the whole collection (dupes.py scan)
  Every step only does what is missing, so an interrupted run just continues.

decide: by hand with the review pages (all work on the previews before export):
    review.py (crops)  -> quality.py review (flags)  -> upright.py review
    -> color_fix.py review (colour / shadows)
  or `--auto`: accept the colour/shadow proposals for every slide not chosen yet
  (flags and orientation need no decision: unflagged = keep, guess = orientation).

export: slide_crop.py export: one JPEG per slide with the crop, colour/shadow
  correction from the NEF where chosen, orientation, date (dates.py); slides flagged
  blank/blurry/duplicate get no JPEG (an existing one is removed). Existing JPEGs
  are kept unless --overwrite.

After a checked export the camera JPGs can go (only with the user's OK).
"""

import argparse
import os
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor

import cv2
from tqdm import tqdm

import slide_crop as sc
from upright import folders_rows, jpg_path, preview_path

PREVIEW = 1600
MARGIN = 24
CLEAR = ("q_std", "q_sharp", "q_ratio", "q_castm", "q_ev", "q_double", "q_warm", "q_face")


def crop_key(r):
    return ",".join(r.get(k, "") for k in ("cx", "cy", "w", "h", "angle"))


def preview_job(a):
    """preview + quality measurements from the full-size crop"""
    import quality
    folder, r = a
    rgb, note = sc.load_image(os.path.join(folder, r["file"]))
    img = sc.crop_image(rgb, r, MARGIN)
    # measured at 1/4 like a JPEG decoded reduced: a measurement on the (resized,
    # recompressed) preview is ~25 % less sharp and would not match quality.REF
    m = quality.measure_img(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), None, fx=.25, fy=.25,
                                       interpolation=cv2.INTER_AREA))
    k = PREVIEW / max(img.shape[:2])
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    p = preview_path(folder, r["file"])
    cv2.imwrite(p + ".tmp.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    os.replace(p + ".tmp.jpg", p)
    return r["file"], m, note


def make_previews(folders, jobs, redo=False):
    for folder, rows in folders_rows(folders):
        os.makedirs(os.path.join(folder, "_preview"), exist_ok=True)
        todo = [r for r in rows.values() if redo or not os.path.exists(preview_path(folder, r["file"]))
                or r.get("preview") != crop_key(r)]
        if not todo:
            continue
        done = {}
        with ProcessPoolExecutor(jobs) as ex:
            for f, m, note in tqdm(ex.map(preview_job, [(folder, r) for r in todo]), total=len(todo),
                                   desc=f"previews {os.path.basename(folder)}", unit="img"):
                done[f] = m
                if note:
                    tqdm.write(f"  {f}: {note}")
        rows = sc.read_csv(folder)
        for f, m in done.items():
            r = rows[f]
            if r.get("preview") and r["preview"] != crop_key(r):     # crop changed: measure again
                for k in CLEAR:
                    r[k] = ""
                if r.get("upright_by") != "manual":
                    r.update(upright="", upright_by="", upright_conf="", upright_checked="")
            r.update(m)
            r["preview"] = crop_key(r)
        sc.write_csv(folder, rows)


def cmd_analyze(args):
    for folder in args.folders:
        sc.cmd_detect(folder.rstrip("/"), Namespace(jobs=args.jobs, redo=False))
    print("previews")
    make_previews(args.folders, args.jobs, args.redo)
    print("measurements")
    import quality
    quality.cmd_scan(Namespace(folders=args.folders, jobs=args.jobs, redo=False))
    print("orientation")
    import upright
    upright.cmd_guess(Namespace(folders=args.folders, jobs=8, redo=False))
    print("faces")
    quality.cmd_faces(Namespace(folders=args.folders, jobs=8, redo=False))
    print("duplicates (against the whole collection)")
    import dupes
    dupes.cmd_scan(Namespace(folders=args.folders, jobs=8, redo=False))


def cmd_decide(args):
    if not args.auto:
        print("decide by hand (each opens a page; stop it with Ctrl+C when done):\n"
              "  .venv/bin/python review.py R          crops flagged `check`\n"
              "  .venv/bin/python quality.py review R  blank / blurry / double flags\n"
              "  .venv/bin/python upright.py review R  orientation\n"
              "  .venv/bin/python dupes.py review R    duplicate pairs\n"
              "  .venv/bin/python color_fix.py review R  colour / shadows\n"
              "or accept the proposals: pipeline.py decide R --auto")
        return
    import color_fix
    color_fix.accept_proposals(args.folders)


def cmd_export(args):
    for folder in args.folders:
        sc.cmd_export(folder.rstrip("/"), Namespace(jobs=args.jobs, overwrite=args.overwrite, margin=MARGIN,
                                                    quality=95))


def cmd_run(args):
    cmd_analyze(args)
    if args.auto:
        cmd_decide(args)
        cmd_export(args)
    else:
        cmd_decide(args)
        print("then: pipeline.py export R")


def cmd_status(args):
    print(f"{'rack':34s} {'NEFs':>5s} {'check':>5s} {'prev':>5s} {'meas':>5s} {'upr':>5s} "
          f"{'flag':>5s} {'corr':>5s} {'JPEG':>5s}")
    for folder, rows in folders_rows(args.folders):
        rs = list(rows.values())
        c = lambda pred: sum(1 for r in rs if pred(r))
        print(f"{os.path.basename(folder)[:34]:34s} {len(rs):5d} {c(lambda r: r['status'].startswith('check')):5d} "
              f"{c(lambda r: os.path.exists(preview_path(folder, r['file']))):5d} {c(lambda r: r.get('q_warm')):5d} "
              f"{c(lambda r: r.get('upright')):5d} {c(lambda r: r.get('flag')):5d} "
              f"{c(lambda r: r.get('color_fix') == 'auto' or int(r.get('shadows') or 0)):5d} "
              f"{c(lambda r: os.path.exists(jpg_path(folder, r['file']))):5d}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "analyze", "decide", "export", "status"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--auto", action="store_true", help="accept the correction proposals (and export, with run)")
    p.add_argument("--redo", action="store_true", help="analyze: make the previews again")
    p.add_argument("--overwrite", action="store_true", help="export: replace existing JPEGs")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze, "decide": cmd_decide, "export": cmd_export,
     "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
