# Slide cropping (Dias)

Turns photographed slides into clean JPEGs. Input: Nikon NEF photos of slides in a
projector (camera on a fixed stand). Output per slide: a JPEG cropped to the film
window, shown upright (EXIF flag), colour- and shadow-corrected where chosen;
blank and blurry shots are flagged for rescanning. The NEFs are never modified.

## Contents

- [Contents](#contents)
- [Overview](#overview)
- [Workflow](#workflow)
- [Output per folder](#output-per-folder)
- [Environment](#environment)
- [Design decisions](#design-decisions)
- [Algorithm (`slide_crop.py`)](#algorithm-slide_croppy)
- [Performance](#performance)
- [Validation notes](#validation-notes)
- [Metadata (EXIF)](#metadata-exif)
- [Things tried and rejected](#things-tried-and-rejected)
- [Open points / ideas](#open-points--ideas)

## Overview

- Layout: one folder per magazine of slides ("rack") on an external drive, holding
  the NEFs (originals) and `crops.csv`; the final images go to
  `library/<rack>/DSC_xxxx.jpg`. Slides flagged blank/blurry/duplicate have no JPEG;
  all NEFs are kept. A fresh export writes to `<rack>/cropped/`; the JPEGs can then be
  moved to `library/<rack>/`. The camera JPGs shot alongside (RAW+JPEG) can be deleted
  after a checked export.
- The tools (`*.py`, `review.sh`, this `README.md`) live next to the rack folders; run
  them from there with `../.venv/bin/python`. Also: `requirements.txt`, `models/`
  (orientation model and CLIP weights) and `.venv/`.
  `review.sh crops|flags|orientation|colour|duplicates` opens a review page for the
  racks listed in it.
  For tests, link NEFs into a scratch folder (`ln -s`) instead of writing onto the drive.

## Workflow

**Capture** (once per magazine; Raspberry Pi + camera, `foto.sh`). The Pi advances the
projector through GPIO pins 24/25 and triggers the camera with gphoto2. The rack name is
stored as the camera's image comment (read back as EXIF `UserComment`):

```
./foto.sh NUMBER_OF_SLIDES NUMBER_OF_BLANKS NAME_OF_RACK
```

It stops the gvfs daemon (it blocks the camera), sets the capture target to the SD card,
shoots one frame per slide, then advances over the blank slots of the magazine and once
more to leave the projector ready for the next one.

**Sorting.** After copying the files from the card, `exiftool` moves every file into a
folder named after its rack (run in the folder that holds the NEFs):

```
exiftool -charset exif=Latin1 "-directory<UserComment" .
```

**Processing.** For a new or rescanned rack (`R` = rack folder(s), e.g. `/path/to/slides/*/`),
run from the drive root (`cd /path/to/slides`):

```
.venv/bin/python pipeline.py run R --auto     # everything, no review
.venv/bin/python pipeline.py run R            # analyse only, then review, then:
.venv/bin/python pipeline.py export R
.venv/bin/python pipeline.py status R         # what is done per rack
```

`pipeline.py` works in three phases; every JPEG is written once, in the export:

| phase | steps | writes |
|---|---|---|
| analyse (`analyze`) | 1 crops, 2 previews and measurements, 3 orientation guess and face brightness, duplicate scan | `crops.csv`, `_preview/`, `duplicates.json` |
| decide (`decide`) | 4 crop review, 5 flags, duplicates, 6 orientation review, 7 colour and shadows — by hand on the review pages, or `decide --auto` | `crops.csv` |
| export (`export`) | 8 one JPEG per slide: crop, colour/shadows, orientation, date; removes JPEGs of slides flagged blank/blurry/duplicate | `cropped/` |

- Each phase only does what is missing, so an interrupted run continues. A crop
  changed after the analysis (e.g. in step 4) gets a new preview, and its
  measurements and (non-manual) orientation are recomputed (column `preview`).
- The review pages show the preview if there is one (`_preview/`), else the
  exported JPEG; the crop review (`review.py`) shows the whole NEF frame.
- Candidate thresholds (blur, colour, double, warm scene) are fixed values from the
  reference collection (`quality.REF`), so a single rack or a few rescans are judged
  by the same standard.
- A test on a copy of one rack gave the same crops and measurements as the reviewed run.
- Camera JPGs are read for the previews and the export; they can be deleted after
  a checked export (after your OK).
- Before a new rack: check for empty projector slots and duplicate shots (see
  [Validation notes](#validation-notes)).

Every step can also be run on its own with the commands given below.

### Analyse

#### 1. Crops

```fish
.venv/bin/python slide_crop.py detect FOLDER [FOLDER ...]   # find crops
```

| option | meaning |
|---|---|
| `--redo` | detect again (rows with status `manual` are kept) |
| `--jobs N` | worker processes (default: all cores) |

Skips work already done (`crops.csv` is checkpointed every 20 images). Debug
output for the edge-angle logic: `SLIDE_CROP_DEBUG=1`. See
[Algorithm](#algorithm-slide_croppy).

#### 2. Previews and measurements

`pipeline.py analyze` makes a preview of each crop (`_preview/`, camera look,
1600 px) and measures from the full-size crop at 1/4 scale (the scale the
thresholds in `quality.REF` belong to); `quality.py scan` adds what is missing,
measured the same way (`measure_slide`, not on the preview):

```
.venv/bin/python quality.py scan  R   # measurements into crops.csv (--redo: all)
.venv/bin/python quality.py faces R   # face brightness, after the orientation guess (--redo: all)
```

| column | measure |
|---|---|
| `q_std` | brightness spread; < 0.06 = nearly uniform (blank, washed out) |
| `q_sharp`, `q_ratio` | edge strength / contrast, and fine vs coarse edges: low = blurry (also low for soft, low-contrast subjects: sunsets, fog, close-ups) |
| `q_ev` | exposure time vs the rack median in EV (from the NEF): two slides stuck together let through much less light, +3 … +5.5 (dark single slides, e.g. night shots, also reach +3) |
| `q_double` | CLIP ViT-L-14 (open_clip, OpenAI weights in `models/clip`, 1.6 GB): "double exposure / two photos overlaid" minus "a normal photograph"; finds double exposures made in the camera |
| `q_castm` | colour cast: mean midtone colour (L 60–200) away from grey (Lab a*b*); faded film, lamp light, but also genuinely red subjects |
| `q_warm` | CLIP: "sunset / campfire / fireworks, candle light / warm evening light" minus "ordinary daytime / indoors / outdoors in daylight": genuine warm colours, not corrected |
| `q_face` | brightness of the darkest face (macOS Vision face detection on the upright slide; `quality.py faces`, run after the orientation guess), `none` without faces |

Candidates (`quality.REF`, from a reference collection): **blank/blurry**:
`q_std` < 0.06, or `q_sharp`/`q_ratio` in the collection's lowest 3 %;
**double**: z(`q_double`) + 0.5 z(`q_ev`) in the highest 4 %; colour and dark
faces: see step 7.

Double ranking: tested on known doubles, the combined score ranks them far higher than CLIP alone or exposure alone (night/flash shots need long exposures too). A
softmax over the prompts is useless (all scores ~1). The top of the list holds a few real
doubles, the rest are mostly dark, violet night/flash shots: the list needs a look
by eye.

#### 3. Orientation guess

```fish
.venv/bin/python upright.py guess R    # only slides without a guess; --redo: all, also manual
```

Stored in `crops.csv` (`upright` = EXIF orientation 1–8, `upright_by` =
`model`/`text`/`manual`, `upright_conf`, `upright_checked`) and written into the
JPEGs by the export.

- **Photos** → image model `upright_model.py`: pretrained ResNeXt50 from
  [ternaus/check_orientation](https://github.com/ternaus/check_orientation)
  (weights in `models/`, 4 classes = upright image turned k × 90° counter-clockwise,
  224 × 224, ImageNet normalisation), averaged over the 4 test rotations. Runs on
  the Apple GPU (MPS). Rotation only: a mirrored photo can't be recognised.
- **Slides with readable text** → `upright_ocr.py` (library, no command line) `upright_ocr.best8`, macOS Vision text
  recognition (pyobjc) at 2000 px in all 8 orientations (4 rotations × mirrored or
  not); the reading with the most real words wins, its rotation corrected by the
  direction of its text lines. Score = 3 × letters of dictionary words (macOS spell
  checker, de/fr/en/it; words capitalised, it skips words in capitals) + 1 ×
  letters of other Latin words ≥ 3 letters (place names); digits and other
  characters count nothing. Used when clear: score ≥ 9 (one dictionary word) and
  ≥ 1.5 × the best other orientation (`upright.text_ok()`: `TEXT_MIN_SCORE`, `TEXT_MARGIN`). Why: Vision reports
  confidence 1.0 for nonsense too, reads mirrored letters as Cyrillic or garbage
  ("BESUCHSTAG" mirrored → "BERNCHaIYC", same length), doesn't read upside-down
  text, and misses small signs at 1500 px.
  The image model can't see mirroring at all, so mirrored photos are only found
  through text.

Pilot on one rack: the model got all photos right but no title
slide (it can't read; several title slides are mirrored, i.e. inserted the wrong
way round); text recognition got nearly all title slides right (mirrored ones too).
Combined, almost all slides were right. Slides went into the projector in every possible way, so
there is no per-rack rule. Most slides are decided by the model, the rest by
text; uncertain ones (< 90 %) are reviewed by eye. A random sample showed all confident ones correct,
~85 % of the uncertain ones.

### Decide

The review pages save every click at once to `crops.csv` (the crop review on
Save / Enter); stop them with Ctrl+C. All have `--port` and `--no-browser`.
Without review, `pipeline.py decide R --auto` takes the proposals (step 7);
unflagged slides are kept and the orientation guesses stand.

#### 4. Crop review

```fish
.venv/bin/python review.py R
```

`review.py` collects every slide whose status is `check: …`, computes a
rack-guided suggestion (`slide_crop.detect_guided`) and opens a local page
(http://127.0.0.1:8765). Per slide: **red** = current crop, **green** =
suggestion; keys `1`/`2` choose, dragging a corner resizes, dragging inside
moves (the box turns **yellow** = edited), ⟲/⟳ rotate by 0.2°. A magnifier shows
full resolution around the mouse; the dashed line is the final crop inside the
safety margin. **Save** (or Enter: save & next) writes the box to that rack's `crops.csv` with status
`manual`.

#### 5. Flags: blank, blurry, double

```
.venv/bin/python quality.py review R   # page to flag (port 8767)
.venv/bin/python quality.py list   R   # flagged slides
.venv/bin/python quality.py list   R --md Rescan.md   # rescan list, by rack
```

Page: candidate lists (blank/blurry, double, flagged), buttons blank / blurry /
double per slide (column `flag`, comma-separated); click an image for full size
(keys 1–3 flag, arrows, Esc). Blank and blurry slides get no JPEG in the export;
blank, blurry and double slides are on the rescan list. Colour is not flagged
here: a cast from faded dyes is on the film, a rescan won't remove it; step 7
corrects it.

Two causes of doubles: two slides stuck
together in the projector (high `q_ev`) and double exposures on the film (normal exposure).
A double or blurry slide is added to the rescan list (`quality.py list --md`).

#### 5b. Duplicates

```
.venv/bin/python dupes.py scan   R   # embeddings + candidate pairs (against the whole collection)
.venv/bin/python dupes.py review R   # page: all pairs in duplicates.json, whole collection (port 8769)
.venv/bin/python dupes.py list   R   # slides marked as duplicate
```

Two stages. **CLIP** embedding of every slide, upright and mirrored (cached in
`models/embeddings.npz`); pairs with cosine ≥ 0.92 are candidates . CLIP finds similar content, not identical pictures (night forests,
sunset series), so each candidate gets a **pixel check**: ORB feature points,
similarity transform (RANSAC, scale 0.8–1.25), normalised correlation of the
aligned images (CLAHE, blurred, overlap ≥ 50 %), as is and mirrored. Pairs with a
match ≥ 0.6 go to `duplicates.json`. Clear duplicates
match ≥ 0.97 (e.g. a run of slides shot twice, or a mirrored copy);
series of very similar shots of one scene (sunsets) also match high, so the decision is by eye.
New pairs are pre-marked (default rule): match ≥ 0.985 (0.99/1.00 on the page) as
duplicate keeping the first slide (a), the rest as different; the review is only
a check. In practice pairs from 0.984 up were nearly all real duplicates; all other pairs were different.
Page: pairs side by side, upright, most similar first; different / keep left /
keep right; click = both full size (keys 1/2/3). The dropped slide gets flag
`duplicate` and `duplicate_of` = the kept one; the orientation, quality and
correction pages, the measurements and the rescan list skip it, the export removes
its JPEG (the NEF stays). `dupes.py scan` sets `duplicate` itself for new pairs
≥ 0.985. Folder names with umlauts come back from
the drive in two Unicode forms; keys are normalised (NFC), otherwise slides were
compared with themselves.

#### 6. Orientation review

```fish
.venv/bin/python upright.py review R
```

```fish
.venv/bin/python upright.py textcheck R   # read text on every slide (column upright_text)
```

Page (http://127.0.0.1:8766): slides as they will be shown; click = turn 90°
clockwise, Alt+click = mirror; 🔍 = full size (r = turn, m = mirror, ←/→, Esc);
"Page OK" marks a page as checked; filters
"uncertain only" / "not yet checked" / "text disagrees" and per rack. "text
disagrees" lists slides where clear text (see step 3) points to another orientation
than the current one, with the words read (blue line under the slide); found
mirrored slides the image model was sure about and slides set wrongly by hand
Not every hit is right
(short words, logos); the decision is by eye. Borders: orange = uncertain
(< 90 %), blue = decided by text, green = set by hand. On exported JPEGs the
change is lossless (only the EXIF orientation is replaced,
`slide_crop.set_jpeg_orientation`).

Confident guesses are not checked by eye unless you do (`upright_checked` empty).

#### 7. Colour and shadows

```
.venv/bin/python color_fix.py review R          # choose per slide (port 8768)
.venv/bin/python pipeline.py decide R --auto    # or: accept all proposals (default)
.venv/bin/python color_fix.py apply R [--all]   # re-export changed (--all: every corrected) slide
```

Per slide: `color_fix` = off / **mild** / **auto** (full), `shadows` = 0 / 35 %,
`exposure` = EV correction for overexposed slides (set by hand in `crops.csv`).

- **Tone: the camera's own curve.** Everything developed from the NEF is mapped
  with a tone curve measured on 60 slides (raw data vs the JPEG embedded in the NEF on the
  same pixel grid, both at half size, median per raw value; `models/camera_curve.npz`, made by
  `models/measure_camera_curve.py OUT.npz SLIDES_ROOT`). The camera renders midtones much brighter than
  plain sRGB (linear 0.18 → 0.68 instead of 0.46); without it corrected slides were
  darker and flatter than the camera look, faces included. Nikon's Active
  D-Lighting (local, per image) is only approximated by a fixed curve.
- **Full colour** (strong casts, faded film): from the NEF (16-bit linear, camera
  white balance): levels per channel (0.1/99.7 %) → camera curve → per-channel
  midtone gamma bringing the channel medians together (strength 0.8, damped for
  green-dominated pictures, which otherwise turn violet) → median brightness
  restored. Damages genuine colours (fire, night skies).
- **Mild** (moderate casts, `q_castm` ≥ 30, no warm scene): the camera look with
  70 % of the midtone cast removed (Lab a*/b* shift, full in the midtones, less
  towards black and white), so contrast and brightness stay exactly the camera's.
  Source: the exported JPEG while it is still the camera look (made from the camera
  JPG, the best quality left), else the camera JPEG via `load_image`. Tried first:
  the full method at half strength from the NEF: casts gone, but bright hazy slides
  came out washed out.
- **Mild + shadows**: from the NEF, levels and midtone balance at half strength.
- **Exposure** (column `exposure`, e.g. `-1`; by hand, no proposal): for
  overexposed slides. Developed from the NEF with highlight headroom (LibRaw
  highlight blending, levels white at 99.98 % instead of 99.7 %), darkened by that
  many EV before the camera curve; combinable with colour/shadows. Recovers what the
  raw still holds; areas clipped in the scan stay white (in one test 4.3 % of the photosites clipped; full colour + -1 EV gave natural colours and some face
  detail; −0.5 to −1.5 EV compared). The real fix for clipped areas is a darker
  rescan.
- **Shadows** (35 %): local tone mapping: the large-scale brightness, smoothed with
  an edge-preserving guided filter (no halos), is pulled towards the brightest
  areas, the fine detail kept. Always with at least mild colour (`wanted()`), and
  afterwards the remaining midtone tint is removed: lifted shadows reveal the cast
  hidden in them. Compared on 8 slides (camera look / 55 % / mild + 35 % / mild +
  35 % + tint removal): 55 % made faces brightest but the pictures flat and hazy
  with a pink/violet veil (without colour) or bluish-violet darks (with mild);
  mild + 35 % + tint removal kept contrast and natural colours, faces still clearly
  visible.

Proposals (`color_fix.candidates`, thresholds in `quality.REF`): **full** for
midtone cast `q_castm` ≥ 47 or the old flag `color`; **mild** for `q_castm` ≥ 30
unless CLIP calls it a warm scene (`q_warm`: sunset, fire, candle light, warm
evening light; above the collection's 95 %); **shadows 35 %** for faces darker
than 0.25 (`q_face`, macOS Vision face detection on the upright slide). 47 corresponds to the former whole-image cast
threshold (regression of the two measures, r = 0.89); the collection's full and
mild groups overlap in `q_castm` (full: median 49.6, a quarter below 44, chosen by
flag or whole-image cast), so a rescan can get full where a similar collection
slide got mild (7 % of the mild slides lie above 47).

Page: already chosen slides and all candidates; each twice,
upright (left camera look, right with the settings). Unchosen slides show the
proposal in orange; a click on a setting or "page: accept shown settings" makes it
a choice (green). Full size: click (space = other side, c = colour off/mild/full,
0/3 = shadows).

The camera look can always be rebuilt from the JPEG embedded in the
NEF.

### Export

#### 8. Export

```fish
.venv/bin/python pipeline.py export R            # = slide_crop.py export for each rack
```

| option | meaning |
|---|---|
| `--overwrite` | replace existing JPEGs (`slide_crop.py export`, `pipeline.py export`) |
| `--margin 24` | safety inset in px per side (`slide_crop.py` only; the pipeline uses 24) |
| `--quality 95` | JPEG quality (`slide_crop.py` only; the pipeline uses 95) |
| `--jobs N` | worker processes |

`slide_crop.py run R` = detect + export; `pipeline.py analyze --redo` makes the
previews again.

One JPEG per slide in `cropped/`: the crop (24 px margin), developed from the NEF
with colour/shadows where chosen (else the camera look), EXIF/XMP from the NEF
with the orientation and the date taken. Slides flagged blank, blurry or
duplicate get no JPEG, and an existing one is removed, so `cropped/` always
follows `crops.csv` (the NEF stays; the flag says why; clear the flag and export to
get the JPEG back). Column
`fixed` records which corrections the JPEG has (`c35` = colour + shadows 35 %,
`-0` = camera look). Existing JPEGs are kept unless `--overwrite`; with the camera
JPGs deleted, a re-export of the camera look uses the NEF's embedded JPEG
(stronger compression).

**Date taken** (`dates.py`; configurable): the NEFs only carry the scan
date. The date is estimated from the rack name: the year from the name;
summer camp types → 15 July, Pentecost camp types → Pentecost Sunday of that year
(computed from Easter), winter camp → 15 February,
Neujahrsweekend → 1 January, other racks with a year (event names) → 1 January. A rack name with two events takes the first. Racks without a year (e.g. misc) keep the scan date.
Written: EXIF `DateTimeOriginal` = that day 12:00:00 + n s for the n-th shot of the
rack (keeps the order; `SubSecTimeOriginal` removed) and XMP
`photoshop:DateCreated` = the year only. The scan date stays in
`DateTimeDigitized`, `xmp:CreateDate` and IFD0 `DateTime` (which `sips` shows as
"creation"). EXIF can't hold a year alone (format `YYYY:MM:DD`; blank parts are
allowed by the standard but not understood by apps), hence the estimated day plus
the year-only XMP. Checked with macOS ImageIO (what Photos and Finder use):
`DateTimeOriginal` is read as the photo date. `dates.py list R` shows the date per
rack. Rescans: the date
comes from the folder name, so they belong into folders named like their rack.

**Changing exported JPEGs** (decisions changed after the export; flags newly set
are handled by running `export` again):

```
.venv/bin/python color_fix.py apply R    # re-export slides whose colour/shadows choice changed
.venv/bin/python dates.py apply R        # (re)write the date, lossless
```

The orientation review patches exported JPEGs itself. Metadata changes are
lossless (`slide_crop.rewrite_jpeg_metadata` replaces only the EXIF and XMP
segments; image data byte-identical, file date kept).

### Disk space

Plan disk space: JPEGs are ~2 MB each; `crops.csv` + review sheets are small, previews
~0.35 MB per slide. Test copies: give them a rack name not used on the drive, the
duplicate scan caches embeddings by rack/file name.

## Output per folder

- `crops.csv`: one row per NEF, the crop. Kept so the NEFs can be converted to
  other formats later with the same crops.
- `cropped/DSC_xxxx.jpg`: cropped images (file date = NEF date, EXIF/XMP copied
  from the NEF; see [Metadata](#metadata-exif)).
- `cropped/_review/review_NNN.jpg`: contact sheets with the crop drawn on:
  green = ok, red = `check: <reason>`, light blue = manual.
- `_preview/DSC_xxxx.jpg` (only for racks processed with `pipeline.py`): the crop
  in the camera look, 1600 px, no EXIF; read by the analysis and the review pages
  (~0.35 MB each). Can be deleted after the export (they are made again if needed,
  as long as the camera JPG or the NEF's embedded JPEG is there).

### crops.csv

Coordinates are in pixels of the **full-size developed raw image** as LibRaw/rawpy
produces it (6032 × 4032), origin top-left.

| column | meaning |
|---|---|
| `file` | NEF name |
| `orientation` | `landscape`, `portrait` (35 mm, 3:2) or `square` (126 film) |
| `cx`, `cy` | centre of the film window |
| `w`, `h` | window size (edge to edge, **before** the safety margin) |
| `angle` | rotation in degrees, positive = clockwise |
| `status` | `ok`, `check: <reasons>`, `check: no slide?` (empty projector slot) or `manual` |
| `img_w`, `img_h` | raw image size |
| `orient_margin`, `straight`, `contrast`, `unevenness`, `film_edge`, `angle_fixed` | diagnostics, see [Algorithm](#algorithm-slide_croppy) |
| `search` | `35mm` (3:2 only), `all` (3:2 + square, square racks) or `35mm, square rejected` (rack was searched for squares, none found) |
| `upright` | EXIF orientation 1–8 written into the JPEG (`upright.py`) |
| `upright_by` | `model`, `text` (OCR) or `manual` |
| `upright_conf` | guess confidence 0–1 (< 0.9 = uncertain) |
| `upright_checked` | `1` = page confirmed on the upright review page |
| `upright_text` | best text reading (`upright.py guess` / `textcheck`): `<exif> <score> <second best> <words>` |
| `q_std`, `q_sharp`, `q_ratio`, `q_castm`, `q_ev`, `q_double`, `q_warm`, `q_face` | quality metrics (`quality.py scan`), see [step 2](#2-previews-and-measurements) |
| `flag` | `blank`, `blurry`, `double`, `duplicate` (comma-separated; `color` from the first colour pass remains as data); by hand, `duplicate` also pre-set by `dupes.py scan` (≥ 0.985); blank/blurry/duplicate rows get no JPEG |
| `duplicate_of` | rack/file of the slide this one duplicates (`dupes.py`) |
| `color_fix` | `auto` = full colour correction, `mild` = cast removed on the camera look (or half strength with shadows), `off` = chosen off; empty = not chosen |
| `shadows` | shadow lift in % (0/35) |
| `exposure` | exposure correction in EV for overexposed slides (e.g. `-1`), by hand |
| `fixed` | what the JPEG currently has (`c35` = full colour + shadows 35 %, `m0` = mild, `c0e-1` = full colour at -1 EV, `-0` = camera look) |
| `preview` | crop (`cx,cy,w,h,angle`) the preview in `_preview/` was made from (`pipeline.py`) |

**Manual fix:** edit `cx/cy/w/h/angle` of a row, set `status` to `manual`, delete
that slide's JPEG and run `pipeline.py analyze` + `export` (new preview,
measurements, orientation; only that JPEG is written). `--overwrite` would re-export
the whole rack from the more compressed embedded JPEG. Detection never overwrites
manual rows.

The camera's embedded JPEG is 6016 × 4016: the same grid minus an 8 px border on
each side (measured by image registration: offset −7.3…−7.9 px, scale 0.9998).
`load_image()` pads it back by 8 px so the CSV coordinates apply unchanged.

## Environment

- `.venv` on the drive, **Python 3.14** (Homebrew). The drive is **FAT32, which
  has no symbolic links**, so the venv is built with `--copies`; it only works
  while the drive is attached to the Mac (it uses the Homebrew Python). Python
  3.15 is also installed but avoided (rawpy/OpenCV wheels).
- Packages: exact versions in `requirements.txt`. Needed: rawpy, opencv-python,
  numpy, pillow, tqdm; torch, torchvision, timm, pyobjc-framework-Vision
  (orientation); open_clip_torch (CLIP scores).
- Rebuild: `/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv --copies --clear .venv && .venv/bin/python -m pip install -r requirements.txt`
  (`.venv/bin/pip` itself may not start after moving; `python -m pip` does).
- CLIP weights: the Hugging Face cache normally links snapshot → blob; on FAT32 the
  file sits directly in
  `models/clip/models--timm--vit_large_patch14_clip_224.openai/snapshots/…/open_clip_model.safetensors`.
  A fresh download onto the drive would copy instead of link (huggingface_hub does
  that when links are not possible).
- macOS writes `._*` files (AppleDouble, 4 KB) next to files on FAT32. Inside
  `.venv` and `models` they break Python (pip lists them as invalid
  distributions, open_clip reads `._*.json` as a model config and crashes). After
  every `pip install`, remove them:
  `find .venv models -name '._*' -type f -size 4096c -delete`
  (running the scripts creates none; elsewhere they are harmless).
- No exiftool installed; metadata is handled with Pillow.

## Design decisions

- **Crop to the true film window**: **3:2** for 35 mm (measured ~1.48–1.53), not
  4:3 as first assumed; **square** for 126/Instamatic slides.
- **Rotation**: slides are shown upright via the EXIF orientation flag (model +
  text recognition, uncertain ones checked by eye).
- **JPEG only**, NEFs untouched. A "cropped NEF" is not possible: a NEF always holds
  the full sensor data, and viewers have no crop field for it. Sidecar/XMP crops
  were rejected because Quick Look, which ignores sidecars.
  DNG has a real crop tag but was not pursued.
- **Camera look** chosen after side-by-side comparison of 8 representative slides
  (brighter and more contrasty than a neutral raw development); corrections from
  the NEF use the camera's measured tone curve so they keep that look.
- **Colour casts and dark shadows** corrected from the NEF, per slide; the
  proposals (full / mild colour by cast, shadows for dark faces) are
  accepted without review by default (`decide --auto`), choices made by hand win.
- **NEFs are never deleted** (apart from empty slots and exact duplicates removed early on). Slides flagged blank, blurry or duplicate only lose their JPEG (the
  export removes it); blank, blurry and double slides are rescanned (`quality.py list --md`).
- **Date taken** estimated per camp type from the rack name (step 8).
- **One export**: everything is decided in `crops.csv` first, then each JPEG is
  written once (`pipeline.py`).
- **Safety margin 24 px**. The mount edge is out of focus (it sits mm in front of
  the film), so the transition is ~20 px wide. With 12 px, thin seams remained.
- **Copy all EXIF/XMP** to the JPEGs (except the MakerNote, which doesn't fit).
- **Empty projector slots** (shots of the projector's insides, or a pure white
  frame) occur in some racks. Some real slides also get
  `no slide?` or `no edges found` (very pale or overexposed slides);
  those must not be deleted.
- **Some mounts show a strip of bare film** between mount and picture (~150 px). The strip is uniform along the border and
  can be brighter or darker than the picture. Narrow strips are not uniform and are not detected; they leave a thin
  seam of ~5–8 px.
- **The window is not a perfect rectangle in the photo** (keystone/lens
  distortion/film skewed in mount). The edge angles are consistent for the fixed setup
  (a few tenths of a degree per edge).
- **Short borders of portrait slides are often misled by picture content**
  (wrong angle of ±2–4°).
- Slides can also sit **genuinely skewed** in their mount,
  so forcing a common angle can let a corner slip outside the window.
- A shadow inside the picture can produce a stronger edge than the real mount
  edge, which gives a crop that is a bit too tight (the safe failure).
- A **soft light fall-off** at one border can put the
  coarse box ~60 px off, beyond the fine search band.

## Algorithm (`slide_crop.py`)

1. **Load** the NEF at half size (`rawpy.postprocess(half_size=True)`), which is
   exactly the full grid / 2.
2. **Coarse search** (`coarse_search`, image 200 px wide): gradients of
   log-brightness (contrast-invariant), clipped so long straight edges beat short
   strong content edges. An exhaustive search over box size, position and format
   using cumulative sums. Formats (`FORMATS`): landscape/portrait 3:2 (aspect
   1.38–1.62, long side 40–75 % of the image width) and square (aspect 0.95–1.08,
   22–55 %). The score is the mean signed gradient on the 4 borders (left +gx,
   right −gx, top +gy, bottom −gy). `orient_margin` = how clearly the winner beat
   its rival (the other 3:2 orientation; for square: the better 3:2 box).
3. **Fine fit** (`refine`, half-size image): per border, sub-pixel peak of the
   signed gradient every 6 px, then a robust line fit. Two passes: a wide band
   (±3 % of size, skipping 8 % near corners; retried with 2× and 3× the band if
   no edge is found), then a narrow band (±4 px, skipping only 12 px near
   corners, because mount edges often bend near corners).
4. **Angle outliers**: a border whose angle deviates > 0.8° from the median of the
   four is refitted with the median angle (offset only), **if** the new line is at
   least 0.8× as well supported by a real edge (`support()`). Additionally, per
   end of that border: if the original line is clearly better supported there
   (> 1.5× and > 0.05), the new line is pushed inward to stay inside it.
5. **Film-strip check** (`film_edge_inside`): looking inward up to 7 % of the size for
   a second straight parallel edge (either polarity, present along ≥ 70 % of the
   border) with a featureless band in between (std along the border < 0.15). If
   found, the border moves to that edge (+3 px).
6. **Uneven edges**: each border is shifted inward by the 98th percentile of its
   residuals.
7. **Inscribed rectangle** (`lines_to_rect`): intersect the 4 lines and take the
   largest rectangle at the mean angle that lies **inside** the quadrilateral
   (an average-size rectangle stuck out by up to ~16 px at corners).
8. **Assessment** (`assess_all`/`assess`, after the whole folder):
   - `no slide?` if contrast < 0.005, or contrast < 0.012 with orientation margin
     < 0.1 (empty slots: 0.000–0.009; real slides down to 0.007, but with a clear
     margin)
   - otherwise `check` if orientation margin < 0.06, straightness < 0.45,
     |angle| > 2.5°, unevenness > 20 px, aspect outside 1.38–1.60 (and not ≤ 1.08),
     or the short side deviates > 4 % from the folder median **of the same format
     family** (3:2 vs square, only with ≥ 3 slides).
9. **Square slides** (`cmd_detect`): searching square for every slide spoils 3:2
   results (small wrong boxes compete, orientation margins collapse), so the first
   pass is 3:2 only. If ≥ 10 % of a rack is flagged (empty slots not counted), the
   **whole rack** is redone with all formats (this leaves ok 3:2 results
   unchanged). The square results are kept only if the rack has **≥ 3 ok squares**
   (otherwise the rack keeps its 3:2 results, `search = 35mm, square rejected`), and
   a single slide only becomes square if its square crop is ok. Result on the whole
   collection: squares found only in the square racks, none elsewhere.
   Single square slides in 3:2 racks stay flagged.
10. **Export** (`export_one`, `crop_image`): warpAffine (rotation + crop − margin)
    from the camera look (camera JPG, else the JPEG embedded in the NEF;
    `load_image`), or the correction (`color_fix.develop`: mild = tint removed on
    the camera look, everything else from the NEF), then save as JPEG with Pillow,
    including the NEF's EXIF/XMP (`nef_metadata()`), the orientation from `upright`
    and the date taken. Rows flagged blank/blurry/duplicate get no JPEG, an existing
    one is removed (`rejected()`).

Parallelism: detection, review tiles and export run in a `ProcessPoolExecutor`
with tqdm progress bars.

## Performance

| step | rate | ~4000 images |
|---|---|---|
| detect from the internal disk (raw decode 0.38 s, coarse 0.05 s, refine 0.07 s) | ~12 img/s | ~6 min |
| detect from the external drive (I/O bound, ~300 % CPU) | 2–5 img/s | ~20–45 min |
| review sheets | ~35 img/s (internal disk) | a few min |
| export, camera look | ~44 img/s (internal disk) | slower from the drive |
| `quality.py scan` (from the drive) | ~60 img/s | ~1 min |
| `color_fix.py apply` (full raw decode + correction, from the drive) | ~3 img/s | |
| `pipeline.py run --auto` (detect, previews, measurements, CLIP, orientation, export; NEFs on the drive, output on the internal disk; before faces and duplicates were added) | ~1.1 img/s | |

Raw decoding and reading from the drive dominate; more parallelism won't help.

## Validation notes

- Crops were first run for 3:2 only; the flags clustered in the square (126) racks,
  so automatic square handling was added. Empty slots and unusual mounts with correct
  crops made up most of the rest. No 3:2 slide that was ok before moved.
- A visual check showed that about half of the flagged crops were wrong (especially
  `orientation unsure`). **Rack-guided re-detection** of flagged slides
  (`slide_crop.detect_guided`, used by `review.py`): search again for a box of the
  rack's typical window size (from its ok slides; square = 1.165 × the 3:2 short
  side), on a finer coarse grid (400 px), then the normal fine fit. Nearly all results
  were correct; the remainder (mounts smaller than the rack's typical window, dark
  strips at the edge, a mount with a printed date, one landscape slide taken as
  square) were done with `review.py`.
- **Duplicates / overlays** (an earlier scratch tool, not today's `dupes.py`): each shot's window region in
  the NEF's 160×120 preview was compared with the shot before and after
  (correlation of log brightness; overlay = least-squares mix of both neighbours).
  Groups were checked by eye; real duplicates were deleted, overlays (double
  exposures on the slide) kept. Only neighbouring shots were compared.
- `crops.csv` is in every rack folder.
- **Export source**: the camera JPGs (6016 × 4016, highest JPEG quality, 12–14 MB)
  have the same rendering as the JPEG embedded in the NEF (mean difference ~1.5/255,
  aligned within 0.5 px), but the embedded one is compressed to 1–2 MB and loses
  film grain and fine detail. So the export reads the camera JPG when it exists.
- Camera JPGs were deleted after export, each only after checking that its NEF and
  cropped JPEG exist. They cannot be regenerated at that quality:
  a re-export without them uses the JPEG embedded in the NEF (same look, stronger
  compression). Do not re-export with `--overwrite` unless that is acceptable.
- Sampled exports were checked visually: correct except square slides (since fixed),
  empty slots (flagged) and thin seams on a few slides.
- Frame-remnant check: the outermost 3 px of each border segment vs. 8–20 px
  inward. Useful as a QA tool to find seams, but **not** usable as an automatic
  flag: dark picture content gives ratios as low as 0.24, real remnants were
  0.27–0.52.

## Metadata (EXIF)

What the NEFs contain (checked on one rack):

- **UserComment = rack name** (e.g. "Camp 88 01"), set in the camera; the only
  metadata about the content.
- **Dates = scan date**, **not** the photo date (the JPEGs get an estimated
  date taken, see [step 8](#8-export)). The order of timestamps and
  file numbers is the order in the rack; gaps in the numbering are probably
  deleted shots.
- Camera/settings describe the reproduction setup: Nikon D750, 170 mm, f/11,
  ISO 100, aperture priority; the exposure time only reflects slide density.
- No GPS position (only GPSVersionID); Artist/Copyright are empty (NUL-filled).
- Nikon MakerNote: ~198 KB. Of that, 167 KB is a 640×424 preview JPEG of the whole
  shot and ~29 KB is Nikon-internal, partly encrypted data (ShotInfo, ColorBalance,
  LensData, unknown blocks). The readable rest is camera-only: lens 70–180 mm
  f/4.5–5.6 (manual focus), RAW, auto WB, Active D-Lighting/vignette/picture
  control, camera serial number, shutter count. Nothing about the slides.
- XMP: 12 KB (CreatorTool, CreateDate).

What the export copies (`nef_metadata()`): Pillow opens the NEF as TIFF (first
page = 160×120 preview) and reads IFD0, the Exif IFD, the GPS IFD and the XMP with
data types intact. It removes the IFD0 tags that describe that preview (size, strips,
SubIFDs, XMLPacket …) and **drops the MakerNote**, because a JPEG EXIF segment is
limited to 64 KB (this also keeps the camera serial number out of the copies). It
sets PixelX/YDimension to the crop size. Everything else is copied unchanged.
Verified by reading the JPEGs back (Pillow, macOS `sips`).

The EXIF inside the camera's embedded JPEG is **not** the original. It is a
minimal block written by LibRaw ("Software: dcraw"), so it is not used.

## Things tried and rejected

- **Variance/overlay of all images** (earlier scripts `frame_remover.py` and notebooks, deleted): fails because
  slides shift between shots. Also rejected as a "rough crop".
- **"Outermost" edge peak instead of strongest**: it caught the glow outside the
  window, so frame remained.
- **Forcing all borders to the common angle**: broke genuinely skewed slides.
  Replaced by the support-based rule.
- **12 px margin**: left seams.
- **Square format searched for every slide**: wrong boxes on 3:2 slides, many
  false "orientation unsure" flags. Replaced by the rack-wide fallback.
- **Retrying single flagged slides as square**: of 14 such "squares" only one was
  real; two wrong ones even passed as ok. Removed.
- **Manual `--square` option per rack**: unnecessary once the automatic rack-wide
  handling worked on all six square racks. Removed.
- **Colour: levels per channel only**: fixed blue casts but not the magenta haze
  of faded film; the same on the 8-bit JPEG did even less. Replaced by levels +
  midtone balance from the NEF (`color_fix.py`).
- **Colour correction with sRGB and a fixed brightening (gamma 0.75)**: darker and
  flatter than the camera look (faces too); brightening to the camera look's
  median then made hazy slides washed out. Replaced by the measured camera curve.
- **Backlit detection** (dark/bright area share + CLIP "silhouettes against a bright
  sky") to propose shadow lifts: also caught sunsets and trees where no lift is
  wanted; replaced by face brightness (`q_face`). Two cast measures (whole image and
  midtones) merged into `q_castm`; the quality page no longer flags colour (the
  correction page decides it).
- **Quality measurements on the previews**: the preview is resized and
  recompressed, sharpness comes out ~25 % lower than on the exported JPEGs, so the
  thresholds would flag far too many slides. Measured from the full-size crop at
  1/4 scale instead (identical values).
- **Candidate lists as percentages of the processed slides**: useless for a single
  rack or a few rescans (2–4 slides); fixed thresholds (`quality.REF`) instead.
- **Image buffer for Vision via `CGDataProviderCreateWithData`** (no release
  callback): every buffer leaked, ~95 MB per slide with 8 readings at 2000 px; a
  full `textcheck` with 8 workers exhausted the machine's memory and WindowServer froze
  (kernel panic, watchdog). Now `CFData` + an autorelease pool per reading
  (`upright_ocr.read_text`; memory flat at ~400 MB per worker) and workers restarted
  every 100 slides (`max_tasks_per_child`).
- **Local vision AI model** (discussed, not tried): too imprecise for edges
  (±2–5 % of the image), slow, 5–10 GB. Possibly useful later for upright
  rotation or as a second check of the finished crops.

## Open points / ideas

- The rack-guided search is in `slide_crop.py` (`detect_guided`) but only used as
  the suggestion in `review.py`, not automatically: it fixed most flagged slides, but every result should be checked by eye (failure: mounts smaller than the rack's typical window).
- Rack-level second pass: learn typical window size and edge angles per rack and
  use them as guidance (against shadow edges, soft edges, thin seams).
- Narrow film strips leave thin seams; adaptive or per-rack margin.
- Dates are estimates per camp type (step 8); an exact date goes into the rack name
  (`2020-Aug-05 …`, `2020-08 …` = 1st of the month),
  which `dates.py` reads first.
- Rescans: blank, blurry and double slides are listed by rack (`quality.py list --md`).
  A whole rescanned rack is shot as `<rack> RESCAN` and replaces the old rack folder
  (NEFs, `crops.csv`, library JPEGs) under the plain rack name, then runs through the [workflow](#workflow).
- Colour correction strength per slide (slider) if some corrected slides look
  over- or under-corrected; `MID_STRENGTH`/`MILD_TINT` are global for now.
- Upright: unchecked guesses can be wrong even when confident-looking (e.g. a mirrored
  slide where text recognition misread a cap).
- Mirrored photos without readable text can't be found automatically (the image
  model can't see mirroring; `textcheck` covers slides with text: signs, T-shirts,
  flags). Script lettering is often not read.
- `quality.REF` was measured on the collection's JPEGs (camera look); recheck it
  once rescans come in.
- Upright: the confident guesses could still be skimmed with the "not yet checked" filter.

## License

[GNU General Public License v3.0](LICENSE) (GPL-3.0).
