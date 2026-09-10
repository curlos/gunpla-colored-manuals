"""
Count USED parts for 403.pdf from the assembly-STEP pages instead of the
Parts List page.

Why this exists (see this session's chat log for the full back-and-forth):
the Parts List badges are tiny (~20px) and hard to read reliably even with
cloud OCR. The user pointed out that assembly-step diagrams already show
every part that gets used - each new part callout is a Latin letter
immediately followed by a black-filled circle badge with a white digit,
noticeably bigger than the Parts List badges (measured on this manual:
E12's step-1 badge is ~38px vs the Parts List's ~20px - though badge size
turns out to vary step to step, see CIRCLE constants below). Crucially,
counting this way needs NO unused/X-mark detection at all: an unused part
is by definition never inserted anywhere, so it can never appear in a step.

Verified against the real page (p2_right_full.png) before writing this:
  - Step 27: two real letter+badge callouts (G17, G6). The other numbers
    visible in that panel (25, 16, 18, 26) sit in SQUARE boxes, not
    circles - those are references back to earlier step numbers (parts
    already built in an earlier step, shown again for placement context),
    not new parts. Hough circle detection naturally never finds square
    boxes at all, so this distinction falls out for free from using
    circle detection specifically (not a generic "is there a number here"
    detector).
  - Step 29: a pink "x2" icon + "2個作る" ("make 2") text near the step
    number, and exactly 4 real letter+badge callouts (F3, PC4, PC2, F4)
    -> 8 parts used. Steps 30-32 carry the same x2 icon; 33-35 have none
    (implicit x1).

Pipeline:
  1. find_step_boxes - locate each step's bold step-number box (a
     rounded-square border, NOT a circle - this is what distinguishes a
     step-number reference from a part badge).
  2. find_multiplier - OCR near each step box for an "xN" pattern (icon
     or plain text).
  3. find_candidate_badges - Hough circles, wide radius net (badge size
     varies step to step - see CIRCLE_MIN/MAX_RADIUS), loose fill-ratio
     prefilter.
  4. verify_new_part - for each candidate badge, OCR the small region
     immediately to its LEFT for a Latin letter/code (A-Z, PC, MP, etc.).
     A real new-part callout always has one; a false-positive round
     mechanical detail (screw, joint, peg - the same false-positive class
     documented in detect_circles.py's docstring) never does. This is the
     main precision lever, playing the role fill-ratio/radius played on
     the Parts List page.
  5. assign_to_step - each verified badge and each multiplier belongs to
     the step panel whose bordered box contains it (panel boxes found the
     same way as detect_labels_403's find_label_boxes, just at panel
     scale instead of label scale).
  6. Tally per step: verified badges found x that step's multiplier
     (default 1) = parts used in that step. Sum across steps = total.

Confidence note: this is a first pass on a brand-new strategy, tested
against a small number of real step panels on ONE manual (verified
step 27, 29, 11 by eye against the source page - not a large sample).
Treat it the way every other threshold in this pipeline has been treated:
spot-check the debug overlay against the real page before trusting a
number from it.
"""
import cv2
import numpy as np
import pytesseract
import re
import sys

# Badge size varies noticeably step to step (measured directly: E12's
# step-1 badge ~38px diameter, G17's step-27 badge closer to ~24-28px) -
# unlike the Parts List page, there's no single tight radius range to
# calibrate against. Cast a wide net here and rely on the letter-adjacency
# check (verify_new_part) for precision instead of a tight radius/fill cut.
CIRCLE_MIN_RADIUS = 10
CIRCLE_MAX_RADIUS = 30
CIRCLE_MIN_DIST = 18
CIRCLE_PARAM2 = 16
CIRCLE_FILL_MIN = 0.45  # looser than the Parts List's 0.58 - just enough to reject obvious non-circles; verify_new_part does the real work

STEP_BOX_MIN_W, STEP_BOX_MAX_W = 30, 90
STEP_BOX_MIN_H, STEP_BOX_MAX_H = 30, 90

MULT_RE = re.compile(r'[x×X]\s*([0-9])')
CODE_RE = re.compile(r'\b([A-Z]{1,2})\b')


def _interior_dark_frac(gray, x, y, w, h, inset=6):
    """Fraction of dark pixels strictly inside a box, excluding its
    border. A real step-number box is mostly white inside with just the
    bold digit's strokes (measured: ~0.28 on a real step box) - the
    x2/xN multiplier icon is a SOLID dark square with reversed white text
    (measured: ~0.68) and would otherwise also pass the box-shape test
    below, since it's the same bordered-square silhouette. This is what
    separates the two."""
    sub = gray[y + inset:y + h - inset, x + inset:x + w - inset]
    if sub.size == 0:
        return 1.0
    _, dark = cv2.threshold(sub, 140, 255, cv2.THRESH_BINARY_INV)
    return cv2.countNonZero(dark) / dark.size


def _find_square_boxes(gray, min_w, max_w, min_h, max_h, max_interior_dark=0.80):
    """Shared box-shape finder: a bordered (mostly-hollow) square/
    rounded-square silhouette in the given size range. Used both for the
    per-panel step-number box (large size range) and for the smaller
    in-panel step-reference boxes like "25"/"16"/"18" in step 27 (small
    size range) - both are the same visual shape, just different sizes,
    and both need to be excluded from circle-candidate consideration
    (see find_candidate_badges)."""
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        aspect = w / float(h)
        if not (0.7 <= aspect <= 1.4):
            continue
        area = cv2.contourArea(c)
        rect_area = w * h
        fill = area / rect_area if rect_area else 0
        # a box's outer border contour fills nearly its whole bounding
        # rect (allowing for rounded corners); a circle badge's bounding-
        # rect fill is ~0.78 (pi/4) - use a higher cutoff to exclude
        # circles specifically
        if fill < 0.85:
            continue
        if _interior_dark_frac(gray, x, y, w, h) > max_interior_dark:
            continue  # solid icon (e.g. the xN multiplier badge), not a hollow number box
        boxes.append((x, y, w, h))
    return boxes


def find_step_boxes(gray):
    """Find each step's bold step-number box: a filled/bordered
    rounded-square containing a large bold digit, top-left of its panel."""
    boxes = _find_square_boxes(gray, STEP_BOX_MIN_W, STEP_BOX_MAX_W, STEP_BOX_MIN_H, STEP_BOX_MAX_H)
    # dedupe nested/duplicate contours (inner+outer border)
    boxes.sort()
    used = [False] * len(boxes)
    kept = []
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        cluster = [b]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if abs(boxes[j][0] - b[0]) < 8 and abs(boxes[j][1] - b[1]) < 8:
                cluster.append(boxes[j])
                used[j] = True
        kept.append(max(cluster, key=lambda t: t[2] * t[3]))
    return kept


def ocr_digit_only(gray, x, y, w, h, pad=4, scale=4):
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    sub = gray[y0:y1, x0:x1]
    big = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # psm 7 (single line) unreliably returns empty on these crops - likely
    # confused by the step box's own border lines touching the digit.
    # psm 8 (single word) reads them cleanly instead - confirmed directly
    # on a real failing crop before making this the default.
    text = pytesseract.image_to_string(bw, config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()
    digits = ''.join(ch for ch in text if ch.isdigit())
    return digits


def find_multiplier(gray, box, search_w=90, search_h=140):
    """The xN icon/text sits just below (sometimes beside) the step box -
    see step 29-32's pink "x2" icon in the verified example. Search a
    generous region below the step box for an x/X/× followed by a digit,
    OR the Japanese "N個作る" ("make N") phrasing seen as a redundant cue
    on the same pages."""
    x, y, w, h = box
    x0, y0 = max(0, x - 10), y
    x1, y1 = min(gray.shape[1], x + search_w), min(gray.shape[0], y + search_h)
    sub = gray[y0:y1, x0:x1]
    big = cv2.resize(sub, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(bw, lang='jpn+eng', config='--psm 6').strip()
    m = MULT_RE.search(text.replace(' ', ''))
    if m:
        return int(m.group(1))
    m2 = re.search(r'([0-9])個作', text)
    if m2:
        return int(m2.group(1))
    return 1


def find_panels(gray, min_area=20000):
    """Segment the page into its actual bordered panel cells using the
    grid lines themselves (same morphological-line technique as
    cluster_geom.py's runner-blob segmentation, applied to the opposite
    purpose here: cluster_geom removes border lines to find ink blobs
    *between* them; this extracts the border lines specifically to find
    the panel cells *enclosed by* them). Direct bordered-rectangle-contour
    detection (the approach that works for the small step-number boxes)
    doesn't work at this scale - adjacent panels share a wall, so a
    single panel's border isn't a closed contour on its own, only the
    outer page frame is. Confirmed empirically: contour-based panel
    detection found 4 panels; this line-based approach finds the correct
    9 for a 9-step page."""
    _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel),
    )
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    non_line = cv2.bitwise_not(lines)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(non_line, connectivity=4)
    panels = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if x == 0 and y == 0:
            continue  # the sparse page-margin region outside the grid, not a real panel - solidity is low but cheaper to just exclude by position
        solidity = area / (w * h) if w * h else 0
        if solidity < 0.3:
            continue
        panels.append((int(x), int(y), int(w), int(h)))
    return panels


def find_candidate_badges(gray, square_boxes):
    """`square_boxes` is every bordered-square shape found on the page
    (step-number boxes AND smaller in-panel step-reference boxes like
    "25"/"16"/"18"/"26" in step 27's own panel - both are the same visual
    shape, just resized, and both must be excluded here). Confirmed bug,
    fixed by this exclusion: Hough fits a rough circle to a square's
    rounded corners often enough that "25", "16", "18", "26" (step-
    reference numbers, not part badges) were showing up as verified
    candidates in the very first version of this script."""
    blur = cv2.medianBlur(gray, 3)
    circ = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=CIRCLE_MIN_DIST,
                             param1=80, param2=CIRCLE_PARAM2,
                             minRadius=CIRCLE_MIN_RADIUS, maxRadius=CIRCLE_MAX_RADIUS)
    if circ is None:
        return []
    _, dark_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    out = []
    for x, y, r in circ[0]:
        if any(bx - 4 <= x <= bx + bw + 4 and by - 4 <= y <= by + bh + 4 for (bx, by, bw, bh) in square_boxes):
            continue
        xi, yi, ri = int(round(x)), int(round(y)), int(round(r))
        m = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(m, (xi, yi), max(1, int(ri * 0.85)), 255, -1)
        region = cv2.bitwise_and(dark_mask, m)
        total = cv2.countNonZero(m)
        dark = cv2.countNonZero(region)
        fill = dark / total if total > 0 else 0
        if fill < CIRCLE_FILL_MIN:
            continue
        out.append({'cx': float(x), 'cy': float(y), 'r': float(r), 'fill': fill})
    out.sort(key=lambda c: -c['fill'])
    kept = []
    for c in out:
        if any(((c['cx'] - k['cx']) ** 2 + (c['cy'] - k['cy']) ** 2) ** 0.5 < (c['r'] + k['r']) * 0.7 for k in kept):
            continue
        kept.append(c)
    return kept


def verify_new_part(gray, cx, cy, r):
    """A real new-part callout has a Latin letter/code immediately to the
    LEFT of the badge circle (e.g. "G(6)", "PC(4)"). Search a tight region
    just left of the circle for OCR-readable Latin text; reject if none
    found - this is the main false-positive filter (rejects plain round
    mechanical details: screws, pegs, ball joints - the same class of
    false positive documented in detect_circles.py)."""
    x0 = max(0, int(cx - r * 4.6))  # wide enough for a 2-char code like "PC"/"MP", not just a single letter
    x1 = max(0, int(cx - r * 0.6))
    y0 = max(0, int(cy - r * 1.3))
    y1 = int(cy + r * 1.3)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    # a real letter has real ink in this crop; a near-blank crop (empty
    # space, or just a thin stray line/arrow) has almost none - gate on
    # this BEFORE running OCR, since Tesseract will otherwise still
    # confidently hallucinate a single letter (usually "I" or "L", both
    # visually close to any thin vertical stroke) out of near-nothing.
    # Confirmed bug this fixed: false "codes" like "LI"/"GG" on candidates
    # that had no real adjacent letter at all.
    _, ink = cv2.threshold(sub, 140, 255, cv2.THRESH_BINARY_INV)
    ink_frac = cv2.countNonZero(ink) / ink.size if ink.size else 0
    if ink_frac < 0.04:
        return None
    big = cv2.resize(sub, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(bw, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
    m = CODE_RE.search(text.replace(' ', ''))
    return m.group(1) if m else None


def process_step_page(img_path, debug_out=None):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    step_boxes = find_step_boxes(gray)
    # also find smaller in-panel reference boxes (e.g. "25"/"16"/"18"/"26"
    # in step 27's own panel) purely so find_candidate_badges can exclude
    # them from circle consideration - these are never treated as steps
    all_square_boxes = step_boxes + _find_square_boxes(gray, 15, STEP_BOX_MIN_W - 1, 15, STEP_BOX_MIN_H - 1)

    steps = []
    for (x, y, w, h) in step_boxes:
        num = ocr_digit_only(gray, x, y, w, h)
        if not num:
            continue  # not a real, readable step-number box - drop it rather than pollute the table with a blank row
        mult = find_multiplier(gray, (x, y, w, h))
        steps.append({'num': num, 'box': (x, y, w, h), 'mult': mult, 'badges': []})

    candidates = find_candidate_badges(gray, all_square_boxes)
    verified = []
    for c in candidates:
        code = verify_new_part(gray, c['cx'], c['cy'], c['r'])
        if code:
            c['code'] = code
            verified.append(c)

    # Assign each verified badge to its actual enclosing panel (found via
    # the page's own grid lines - see find_panels), then match each panel
    # to the step box whose top-left corner falls inside/near it. This
    # replaced an earlier "nearest step box" proximity heuristic that
    # leaked badges across panel boundaries into the wrong step - proper
    # containment via the real panel geometry fixes that.
    panels = find_panels(gray)

    def panel_for_point(px, py):
        for (bx, by, bw, bh) in panels:
            if bx - 5 <= px <= bx + bw + 5 and by - 5 <= py <= by + bh + 5:
                return (bx, by, bw, bh)
        return None

    panel_to_step = {}
    for s in steps:
        sx, sy, sw, sh = s['box']
        p = panel_for_point(sx + sw / 2.0, sy + sh / 2.0)
        if p is not None:
            panel_to_step[p] = s

    for c in verified:
        p = panel_for_point(c['cx'], c['cy'])
        step = panel_to_step.get(p) if p is not None else None
        if step is not None:
            step['badges'].append(c)

    if debug_out:
        dbg = img.copy()
        for s in steps:
            x, y, w, h = s['box']
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(dbg, f"#{s['num']} x{s['mult']}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        for c in verified:
            cv2.circle(dbg, (int(c['cx']), int(c['cy'])), int(c['r']), (0, 0, 255), 3)
            cv2.putText(dbg, c['code'], (int(c['cx'] - c['r']), int(c['cy'] - c['r'] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        cv2.imwrite(debug_out, dbg)

    return steps


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python3 steps_used_parts.py page1.png [page2.png ...]")
        return
    grand_total = 0
    all_steps = []
    for i, p in enumerate(paths):
        steps = process_step_page(p, debug_out=f'debug/steps_debug_{i}.png')
        all_steps.extend(steps)

    all_steps.sort(key=lambda s: (int(s['num']) if s['num'].isdigit() else 9999))
    print(f"\n{'Step':>6s} {'New parts found':>16s} {'x':>4s} {'Subtotal':>9s}")
    for s in all_steps:
        n = len(s['badges'])
        sub = n * s['mult']
        grand_total += sub
        codes = ','.join(c['code'] for c in s['badges'])
        print(f"{s['num']:>6s} {n:>16d} {'x' + str(s['mult']):>4s} {sub:>9d}   [{codes}]")
    print(f"\n=== TOTAL USED PARTS (from steps): {grand_total} ===")


if __name__ == '__main__':
    main()
