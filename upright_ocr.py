"""Orientation of slides by text recognition (macOS Vision).

best8(): read the slide in all 8 orientations (4 rotations x mirrored or not) and
keep the reading with the most real words. Vision itself is no judge: it reports
confidence 1.0 for nonsense too, reads mirrored Latin letters as Cyrillic or
garbage ("BESUCHSTAG" mirrored -> "BERNCHaIYC"), and reads some rotated text but
not all of it. Score = 3 x letters of dictionary words (macOS spell checker, de/fr/
en/it, word capitalised because it skips words in capitals) + 1 x letters of other
Latin words of >= 3 letters (place names: Vättis, Ettiswil); digits and other
characters count nothing. The rotation of a reading is corrected by the direction
of its text lines (corner points), because Vision reads some rotated text."""
import re

import numpy as np
import Quartz
import Vision

WORD = re.compile(r"[A-Za-zÄÖÜäöüÉÈÊÀÂÇéèêàâçß]{3,}")
DICT_WEIGHT = 3
LANGS = ("de", "fr", "en", "it")
_spell = None


def is_word(w):
    """known word in one of LANGS (macOS spell checker)"""
    global _spell
    if _spell is None:
        from AppKit import NSSpellChecker
        _spell = NSSpellChecker.sharedSpellChecker()
    w = w.capitalize()
    for lang in LANGS:
        r = _spell.checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(
            w, 0, lang, False, 0, None)
        if r[0].length == 0:
            return True
    return False


def words_score(text):
    """(score, words) of a reading: dictionary words count DICT_WEIGHT x their letters"""
    words = WORD.findall(text)
    score = sum(len(w) * (DICT_WEIGHT if is_word(w) else 1) for w in words)
    return score, " ".join(w for w in words)


def _cgimage(rgb):
    h, w = rgb.shape[:2]
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)]).copy()
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    # CFData, so the pixels are freed with the image (CGDataProviderCreateWithData
    # without a release callback leaked every buffer: ~95 MB per slide in best8,
    # which exhausted the memory of the Mac with 8 workers)
    from Foundation import NSData
    prov = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes))
    return Quartz.CGImageCreate(w, h, 8, 32, w * 4, cs, Quartz.kCGImageAlphaNoneSkipLast,
                                prov, None, False, Quartz.kCGRenderingIntentDefault)


def read_text(rgb):
    """(score, text, direction): score = sum of confidence * letters over the lines;
    direction = letter-weighted mean reading direction (dx, dy), image coords, y down.
    Runs in an autorelease pool: without it the Objective-C objects of every call
    pile up in long-running processes."""
    import objc
    with objc.autorelease_pool():
        return _read_text(rgb)


def _read_text(rgb):
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    req.setRecognitionLanguages_(["de-DE", "en-US"])
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_cgimage(rgb), None)
    handler.performRequests_error_([req], None)
    score, text, d = 0.0, [], np.zeros(2)
    for obs in req.results() or []:
        c = obs.topCandidates_(1)[0]
        s = str(c.string())
        letters = sum(ch.isalpha() for ch in s)
        score += float(c.confidence()) * letters
        text.append(s)
        tl, tr = obs.topLeft(), obs.topRight()            # normalised, origin bottom-left
        d += letters * np.array([tr.x - tl.x, -(tr.y - tl.y)])
    return score, " / ".join(text), d


def best8(rgb):
    """(mirrored, rot_cw, score, words, second): the best of the 8 orientations
    (flip left-right first, then rotate clockwise); second = score of the best
    reading with a different (mirrored, rot_cw)"""
    res = {}
    for m in (False, True):
        for r in (0, 90, 180, 270):
            img = rgb[:, ::-1] if m else rgb
            img = np.ascontiguousarray(np.rot90(img, -r // 90))
            _, text, d = read_text(img)
            score, words = words_score(text)
            if not score:
                continue
            extra = int((-round(np.degrees(np.arctan2(d[1], d[0])) / 90) * 90) % 360)
            key = (m, (r + extra) % 360)
            if score > res.get(key, (0, ""))[0]:
                res[key] = (score, words)
    if not res:
        return False, 0, 0, "", 0
    ranked = sorted(res.items(), key=lambda kv: -kv[1][0])
    (m, r), (score, words) = ranked[0]
    return m, r, score, words, ranked[1][1][0] if len(ranked) > 1 else 0
