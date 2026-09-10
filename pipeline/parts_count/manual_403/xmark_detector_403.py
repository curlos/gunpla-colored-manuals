"""
Unused-part ("x") mark detector for 403.pdf's Parts List page.

403.pdf's Parts List header explicitly says "(x印は使用しないパーツです。)"
("the x mark means this part isn't used") - and, after visually inspecting
every runner on this page (crop-and-zoom, the same discipline used for the
circle detector's false positives), there turns out to be exactly ONE
visual convention for it here: a small bare "x" glyph (no circle around it,
unlike a used-part marker) sitting immediately next to the last circled
instance of a repeated part number, on the <PC-132AB> polycap runner. No
other runner (A-G) has any x mark, and there is no "large x drawn across an
entire molded part's outline" convention on this manual either (949.pdf's
second style) - confirmed by inspecting the full page at native resolution.

Calibration story (two earlier general approaches were tried this session
and abandoned - both are documented in 949's README as having been tried
there too, for the same reasons):

1. Plain contour circularity/fill, scanned over the whole page: sampling
   the two known real x marks gives fill~0.21-0.23, circularity~0.10 -
   looks like a clean signature in isolation. But scanning the *whole*
   page's content area (excluding only the header/legend, same
   CONTENT_TOP_Y cutoff as the circle detector) for anything of similar
   bbox size (14-30px) and a loose version of that signature
   (fill 0.12-0.40, circularity<0.35) turns up 87 candidates. Almost all
   of them are either: (a) a real number circle's own inner digit-stroke
   contour (RETR_LIST returns the digit glyph as its own low-circularity
   contour, nested inside the circle's outer boundary - these have a
   centroid essentially coincident with an already-detected circle's own
   center, distance ~0px), or (b) ordinary sprue-diagram line art (a
   right-angle frame-bar junction, a corner of a molded part's outline,
   Kanji strokes in the material-composition text) that happens to share
   the same rough size/fill/circularity band. This is the same failure
   mode 949's README describes for its own abandoned contour-circularity
   attempt.

2. Adding "must be near a real number circle, but not so near it's that
   circle's own digit stroke" (this page's actual x-mark convention: the
   mark sits right where the *next* circled instance in a repeating
   sequence would otherwise go) is a huge improvement - down to 3
   candidates page-wide: the 2 real x marks (distance ~25-28px from the
   nearest circle) and exactly one leftover false positive on D, at almost
   the same distance (~28px): a right-angle "+"-shaped sprue-frame bar
   junction next to D's 10/11/12 circles. Distance-to-nearest-circle alone
   can't separate a 28px-away real x from a 28px-away frame junction.

The feature that actually separates them: an "x" is two *diagonal*
crossing strokes; a sprue-frame bar junction is two strokes at 0/90
degrees ("+"-shaped), even though both have near-identical bounding-box
size, fill ratio and circularity. Binning each candidate's dark pixels
(within its own padded bbox, normalized to [-1, 1]) into a "near either
diagonal" band vs a "near either central axis" band gives a clean,
large-margin split:
    real x marks:        diag_score ~0.97-1.00, axis_score ~0.33-0.34
    the D frame-junction: diag_score ~0.58,      axis_score ~0.65
This is the key new signal for this manual - 949's session never needed
it, because 949's abandoned attempts didn't get far enough to hit this
specific frame-junction failure mode.

Net result: contour fill/circularity prefilter -> exclude a candidate
that's really just the nearest circle's own digit stroke -> require it be
within a real runner's own component blob (reuses the same connected-
component assignment as circle-to-runner clustering - this alone excludes
the header, legend, labels, and material-line text) -> cap distance to the
nearest circle at 40px (this convention's marks always sit immediately
next to a circle) -> diagonal-vs-axis stroke check. This lands on exactly
the 2 real x marks with a wide margin on every threshold, but it has only
ever been calibrated against those 2 known positives and 1 known false
positive - see the README's confidence caveats before trusting this on a
different manual.
"""
import cv2
import numpy as np
import sys

sys.path.insert(0, '..')
from cluster_geom import build_runner_components, match_labels_to_components
from detect_circles_403 import detect_circles, CONTENT_TOP_Y
from detect_labels_403 import detect_labels

MIN_BBOX = 14
MAX_BBOX = 30
ASPECT_MIN, ASPECT_MAX = 0.6, 1.6
FILL_MIN, FILL_MAX = 0.15, 0.30
CIRC_MAX = 0.20
MAX_DIST_TO_CIRCLE = 40
DIAG_MIN = 0.85
AXIS_MAX = 0.50

PC_LABEL = {'code': 'PC', 'mult': 1, 'box_l': 1720, 'box_t': 1745,
            'box_r': 2010, 'box_b': 1800, 'anchor_x': 1865, 'anchor_y': 1800}
MIN_COMPONENT_HEIGHT = 150


def _diag_vs_axis_score(gray, cx, cy, bw, bh):
    pad = 3
    x0, y0 = int(cx - bw / 2) - pad, int(cy - bh / 2) - pad
    x1, y1 = int(cx + bw / 2) + pad, int(cy + bh / 2) + pad
    sub = gray[max(0, y0):y1, max(0, x0):x1]
    if sub.size == 0:
        return 0.0, 1.0
    _, mask = cv2.threshold(sub, 128, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 1.0
    h, w = sub.shape
    nx = (xs / w) * 2 - 1
    ny = (ys / h) * 2 - 1
    band = 0.35
    diag_score = float(np.mean((np.abs(nx - ny) < band) | (np.abs(nx + ny) < band)))
    axis_score = float(np.mean((np.abs(nx) < band * 0.7) | (np.abs(ny) < band * 0.7)))
    return diag_score, axis_score


def detect_xmarks(img_path, debug_out=None):
    circles, gray = detect_circles(img_path)
    labels = detect_labels(img_path)
    labels.append(dict(PC_LABEL))
    comps, comp_img, (px, py) = build_runner_components(gray, labels)
    comps = [c for c in comps if c['h'] > MIN_COMPONENT_HEIGHT]
    keep_ids = {c['id'] for c in comps}
    comp_to_label = match_labels_to_components(labels, comps)
    comp_to_code = {cid: lb['code'] for cid, lb in comp_to_label.items()}

    h, w = gray.shape
    _, mask = cv2.threshold(gray[CONTENT_TOP_Y:h, :], 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    xmarks = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if not (MIN_BBOX <= bw <= MAX_BBOX and MIN_BBOX <= bh <= MAX_BBOX):
            continue
        aspect = bw / float(bh)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        area = cv2.contourArea(c)
        fill = area / (bw * bh) if bw * bh else 0
        perim = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
        if not (FILL_MIN <= fill <= FILL_MAX and circularity < CIRC_MAX):
            continue

        gx, gy = x + bw / 2.0, y + CONTENT_TOP_Y + bh / 2.0

        # must sit inside a real runner's own component blob (not header/
        # legend/label/material text - all outside any runner blob)
        lx, ly = int(gx - px), int(gy - py)
        cid = comp_img[ly, lx] if 0 <= ly < comp_img.shape[0] and 0 <= lx < comp_img.shape[1] else 0
        code = comp_to_code.get(cid) if cid in keep_ids else None
        if code is None:
            continue

        # distance to the nearest detected number circle: must be close
        # (this convention's x sits right next to a circle) but not so
        # close it's actually that circle's own digit-stroke contour
        best_d, best_r = None, None
        for circ in circles:
            d = ((gx - circ['cx']) ** 2 + (gy - circ['cy']) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_d, best_r = d, circ['r']
        if best_d is None or best_d < best_r + 3 or best_d > MAX_DIST_TO_CIRCLE:
            continue

        diag_score, axis_score = _diag_vs_axis_score(gray, gx, gy, bw, bh)
        if not (diag_score >= DIAG_MIN and axis_score <= AXIS_MAX):
            continue

        xmarks.append({
            'cx': gx, 'cy': gy, 'w': bw, 'h': bh, 'fill': fill,
            'circularity': circularity, 'diag_score': diag_score,
            'axis_score': axis_score, 'dist_to_circle': best_d,
            'x': int(gx - bw / 2), 'y': int(gy - bh / 2),
        })

    print(f"Found {len(xmarks)} x-mark candidates in {img_path}")

    if debug_out:
        img = cv2.imread(img_path)
        for xm in xmarks:
            cv2.rectangle(img, (xm['x'], xm['y']), (xm['x'] + xm['w'], xm['y'] + xm['h']), (255, 0, 255), 3)
        cv2.imwrite(debug_out, img)

    return xmarks, gray


if __name__ == '__main__':
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    xm, _ = detect_xmarks(path, out)
    for m in xm:
        print(m)
