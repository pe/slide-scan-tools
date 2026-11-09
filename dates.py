#!/usr/bin/env python3
"""
Set the date a slide was taken, estimated from its rack name. The NEF's dates are
the scan date (2024); with them, photo apps file every slide under 2024.

    python dates.py apply FOLDER [FOLDER ...]   patch cropped/*.jpg (lossless)
    python dates.py list  FOLDER [FOLDER ...]   show the date per rack

Rule (agreed with the user): the year from the rack name, the day from the camp
type: summer camps 15 July, Pentecost camps the Pentecost Sunday of that year,
winter camps 15 February, every other rack with a year 1 January. A rack name
with two events takes the first. Racks without a year are left alone.
The keywords are configured in the table below; adapt them to your own naming.
A rack name starting with an exact date, "YYYY-MM-DD ..." (month as number or
English/German 3-letter name: "1987-Aug-05"; "YYYY-MM ..." = the 1st of that month), takes that date, whatever the camp type.

Written: EXIF DateTimeOriginal = that day, 12:00:00 + n seconds for the n-th shot of
the rack (keeps the rack order in photo apps; SubSecTimeOriginal = 00, so that exiftool
and Immich, which prefer SubSecCreateDate to DateTimeOriginal, use this date), XMP
photoshop:DateCreated = the year only (the honest value; Lightroom/digiKam show
it, macOS Photos only reads EXIF), and xmp:CreateDate = the same date and time as
DateTimeOriginal (Immich reads xmp:CreateDate before EXIF and would otherwise file
every slide under the scan date); the stray DateTimeOriginal in IFD0 (a copy of the
scan date, found in the NEF's IFD0) is removed, exiftool/Immich read it before the
Exif IFD one. The scan date stays in DateTimeDigitized and
IFD0 DateTime. slide_crop.export writes the same, so re-exports keep it.

    python dates.py apply FOLDER ... --library DIR   JPEGs are in DIR/<rack name>/
                                                     (moved out of FOLDER/cropped)
"""

import argparse
import datetime
import os
import re
import sys

import slide_crop as sc

FIXED = [  # (pattern in the rack name, month, day), first match wins
    (r"\bSola\b", 7, 15),
    (r"\bWoLa\b", 7, 15),
    (r"\bSkiLa\b", 2, 15),
    (r"Neujahr", 1, 1),
]
MONTHS = {m: i for i, ms in enumerate(
    ["jan", "feb", "mar mär", "apr", "may mai", "jun", "jul", "aug", "sep", "oct okt", "nov", "dec dez"], 1)
    for m in ms.split()}
PENTECOST = r"\b(PfiLa|PifLa|PfaLa)\b"


def easter(y):
    """Gregorian Easter Sunday (anonymous algorithm)"""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    h = (19 * a + b - d - (b - (b + 8) // 25 + 1) // 3 + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return datetime.date(y, month, (h + l - 7 * m + 114) % 31 + 1)


def rack_date(name):
    """(date, rule) for a rack folder name, or None without a year"""
    m = re.match(r"\s*((?:19|20)\d\d)-(\d\d|[A-Za-zä]{3})(?:-(\d\d?))?(?![\w-])", name)
    if m:
        mon = m.group(2)
        mon = int(mon) if mon.isdigit() else MONTHS.get(mon.lower())
        try:
            return datetime.date(int(m.group(1)), mon, int(m.group(3) or 1)), "exact date"
        except (ValueError, TypeError):
            pass
    m = re.match(r"\s*(19\d\d|20\d\d)\b(.*)", name)
    if not m:
        return None
    year, rest = int(m.group(1)), m.group(2)
    rest = re.split(r"\b(?:19|20)\d\d\b", rest)[0]          # first event only
    if re.search(PENTECOST, rest):
        return easter(year) + datetime.timedelta(days=49), "Pentecost"
    for pat, mo, d in FIXED:
        if re.search(pat, rest):
            return datetime.date(year, mo, d), pat.strip("\\b")
    return datetime.date(year, 1, 1), "year only"


def slide_datetime(folder, nef, rows):
    """EXIF date string for a slide, or None; n = position of the NEF in the rack"""
    rd = rack_date(os.path.basename(folder.rstrip("/")))
    if not rd:
        return None
    n = sorted(rows).index(nef)
    t = datetime.datetime.combine(rd[0], datetime.time(12)) + datetime.timedelta(seconds=n)
    return t.strftime("%Y:%m:%d %H:%M:%S"), str(rd[0].year)


def set_exif_date(exif, when):
    ifd = exif.get_ifd(0x8769)
    ifd[0x9003] = when
    ifd[0x9291] = "00"                                      # SubSecTimeOriginal: without it exiftool has no SubSecDateTimeOriginal and Immich takes SubSecCreateDate (the scan date)
    exif.pop(0x9003, None)                                  # stray copy of the scan date in IFD0 (from the NEF); Immich reads it first


def xmp_create_date(when):
    """EXIF date string 'YYYY:MM:DD HH:MM:SS' -> XMP date 'YYYY-MM-DDTHH:MM:SS'"""
    return when.replace(":", "-", 2).replace(" ", "T")


def set_xmp_dates(xmp, when, year):
    """in an XMP packet (bytes or str): photoshop:DateCreated = year, xmp:CreateDate =
    the EXIF date string `when` -> bytes"""
    s = xmp.decode("utf-8") if isinstance(xmp, bytes) else xmp
    s = re.sub(r'\s*photoshop:DateCreated="[^"]*"', "", s)
    s = re.sub(r"<photoshop:DateCreated>[^<]*</photoshop:DateCreated>", "", s)
    s = re.sub(r'\s*xmp:CreateDate="[^"]*"', "", s)
    s = re.sub(r"<xmp:CreateDate>[^<]*</xmp:CreateDate>", "", s)
    ns = {"photoshop": "http://ns.adobe.com/photoshop/1.0/", "xmp": "http://ns.adobe.com/xap/1.0/"}
    attr = "".join(f' xmlns:{k}="{v}"' for k, v in ns.items() if f"xmlns:{k}=" not in s)
    attr += f' photoshop:DateCreated="{year}" xmp:CreateDate="{xmp_create_date(when)}"'
    s = re.sub(r"<rdf:Description\b", lambda m: m.group(0) + attr, s, count=1)
    return s.encode("utf-8")


def xmp_is_dated(xmp, when):
    """True if the XMP packet already has the photo date in xmp:CreateDate"""
    if not xmp:
        return True
    s = xmp.decode("utf-8") if isinstance(xmp, bytes) else xmp
    return (f'xmp:CreateDate="{xmp_create_date(when)}"' in s
            or f"<xmp:CreateDate>{xmp_create_date(when)}</xmp:CreateDate>" in s) \
        and "photoshop:DateCreated" in s


def date_one(job):
    """date one JPEG; 1 = rewritten, 0 = already right"""
    from PIL import Image
    p, when = job
    with Image.open(p) as im:
        exif, xmp = im.getexif(), im.info.get("xmp")
    ifd = exif.get_ifd(0x8769)
    if (ifd.get(0x9003) == when[0] and ifd.get(0x9291) == "00" and 0x9003 not in exif
            and xmp_is_dated(xmp, when[0])):
        return 0
    set_exif_date(exif, when[0])
    sc.rewrite_jpeg_metadata(p, exif, set_xmp_dates(xmp, when[0], when[1]) if xmp else None)
    return 1


def cmd_apply(args):
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    from upright import folders_rows, jpg_path
    if args.library:
        jpg_path = lambda folder, nef: os.path.join(
            args.library, os.path.basename(folder), os.path.splitext(nef)[0] + ".jpg")
    jobs = []
    for folder, rows in folders_rows(args.folders):
        for f in sorted(rows):
            when = slide_datetime(folder, f, rows)
            p = jpg_path(folder, f)
            if when and os.path.exists(p):
                jobs.append((p, when))
    with ProcessPoolExecutor(args.jobs) as pool:
        n = sum(tqdm(pool.map(date_one, jobs, chunksize=8), total=len(jobs), unit="img", smoothing=0.05))
    print(f"{n} of {len(jobs)} JPEGs dated, {len(jobs) - n} were already right")


def cmd_list(args):
    for folder in args.folders:
        name = os.path.basename(folder.rstrip("/"))
        rd = rack_date(name)
        print(f"{name:40s} {rd[0].isoformat() + '  (' + rd[1] + ')' if rd else '-  (no year, unchanged)'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["apply", "list"])
    p.add_argument("folders", nargs="+")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes (apply)")
    p.add_argument("--library", help="directory holding the JPEGs as DIR/<rack name>/DSC_xxxx.jpg")
    args = p.parse_args()
    {"apply": cmd_apply, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
