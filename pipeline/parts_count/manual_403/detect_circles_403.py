"""
Circle (used-part marker) detector tuned for 403.pdf's Parts List page.

Reuses the same core method as detect_circles.py (Hough circles + dark-fill
ratio) but with different constants: 403.pdf's embedded page image is a much
higher native resolution than 949.pdf's per the actual pixel measurements
(21px circle diameter here vs 949's 11-17px), so radius range, minDist and
the Hough param2 all needed re-tuning. Also, this page has NO grey margin
around the white page (see detect_circles.find_page_bbox's updated
docstring) - handled generically there now, not here.
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '..')
from detect_circles import find_page_bbox

# calibrated by sampling real circles on this page: a plain single-digit
# marker measures ~21px diameter (r=10.5). A first pass with a loose
# minRadius=8/maxRadius=15 showed a clean bimodal radius histogram: the
# real markers cluster tightly at r=9.6-11.2, and nearly all of the
# leftover false positives (plain round plastic part details - joints,
# lenses, pegs - that happen to be solid and dark enough to pass the fill
# check) sit either well above that (r=13.2-14.4) or, in a couple of
# cases, a bit below (r=8.4-8.8). Tightening the search range to the real
# cluster removes most of them for free, the same lever as 949's
# per-runner radius cap but applied globally here since this page has
# only one native contrast/resolution regime (no recall-boost pass
# needed - see README).
MIN_RADIUS = 9
MAX_RADIUS = 12
MIN_DIST = 13
PARAM2 = 13
# 0.60 was the first cutoff tried (matching the valley in a page-wide fill
# histogram at loose radius/no top-cutoff - see README) but a visual
# audit (crop-and-zoom every runner against its own hand count) found 4
# genuine digit circles sitting right at fill=0.58-0.60 that it excluded
# (two 2-digit numbers on C - "10" and "14" - plus one each on A and G) -
# all 4 confirmed real by inspecting the source crop, not noise. Lowering
# to 0.58 recovers exactly those 4 with no new false positives; going
# further to 0.55 starts pulling in real false positives too (confirmed:
# one is the same "+"-shaped sprue-frame bar junction the x-mark detector
# had to filter out - see xmark_detector_403.py), so 0.58 is the floor.
FILL_CUTOFF = 0.58
# A handful of plain, unlabeled round plastic parts (ball joints / frame
# attachment nubs - the same false-positive class 949's README documents
# for "S"/"M") pass the Hough+fill test because they render as solid dark
# disks too. One case found on this page (a frame nub on E) is a clean
# fill=1.00 - no digit is cut into it at all, so nothing lightens the
# disk's center - matching 949's exact "solid-fill >=0.97" false-positive
# signature; reusing that same cutoff here removes it (and one bullet-
# point glyph in the page footer, which is harmless either way since nothing
# assigns it to a runner). This does NOT catch every false positive of
# this class - see README's G runner caveat for two that remain.
SOLID_FILL_REJECT = 0.97

# This page's top ~900px is entirely the safety-warning header and the
# pictogram legend (scissors/glue/×2/rotate icons etc.) - NOT any runner's
# parts. Several legend pictograms contain genuine solid black circles
# (the "build this first/later" icons) and the warning text boxes contain
# enough dense Kanji ink to trip the fill-ratio filter too. Since no
# runner's real content starts before this, just exclude the whole zone
# outright rather than trying to filter each false positive individually.
CONTENT_TOP_Y = 900


def detect_circles(img_path, debug_out=None, fill_cutoff=FILL_CUTOFF):
    img = cv2.imread(img_path)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    px, py, pw, ph = find_page_bbox(gray_full)
    crop = gray_full[py:py+ph, px:px+pw]
    blur = cv2.medianBlur(crop, 3)

    circles_raw = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=MIN_DIST,
        param1=80, param2=PARAM2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS,
    )

    _, dark_mask = cv2.threshold(crop, 190, 255, cv2.THRESH_BINARY_INV)

    circles = []
    if circles_raw is not None:
        for x, y, r in circles_raw[0]:
            xi, yi, ri = int(round(x)), int(round(y)), int(round(r))
            m = np.zeros(crop.shape, dtype=np.uint8)
            cv2.circle(m, (xi, yi), max(1, int(ri * 0.85)), 255, -1)
            region = cv2.bitwise_and(dark_mask, m)
            total = cv2.countNonZero(m)
            dark = cv2.countNonZero(region)
            fill = dark / total if total > 0 else 0
            if fill < fill_cutoff or fill >= SOLID_FILL_REJECT:
                continue
            gx, gy = x + px, y + py
            if gy < CONTENT_TOP_Y:
                continue
            w = h = r * 2
            circles.append({
                'cx': float(gx), 'cy': float(gy), 'r': float(r), 'fill': fill,
                'x': int(gx - r), 'y': int(gy - r), 'w': int(w), 'h': int(h),
            })

    # de-duplicate near-identical detections (radius-aware, same rationale
    # as recall_boost2's dedup: a fixed pixel cutoff doesn't scale with
    # circle size)
    circles.sort(key=lambda c: -c['fill'])
    kept = []
    for c in circles:
        if any(((c['cx']-k['cx'])**2 + (c['cy']-k['cy'])**2) ** 0.5 < (c['r']+k['r'])*0.7 for k in kept):
            continue
        kept.append(c)
    circles = kept

    print(f"Found {len(circles)} candidate circles in {img_path}")

    if debug_out:
        dbg = img.copy()
        for c in circles:
            cv2.circle(dbg, (int(c['cx']), int(c['cy'])), int(c['r']), (0, 0, 255), 2)
        cv2.imwrite(debug_out, dbg)

    return circles, gray_full


if __name__ == '__main__':
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    detect_circles(path, out)
