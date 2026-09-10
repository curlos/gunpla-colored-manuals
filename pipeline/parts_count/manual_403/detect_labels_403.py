"""
Runner-label detector for 403.pdf's Parts List page.

403.pdf's label style is genuinely different from 949.pdf's (not just a
different scan resolution), so this does not reuse detect_labels.py's
box-contour logic directly - it re-derives the same *idea* (find a bordered
box, OCR it, erase the border first) for this page's actual layout:

- 949.pdf: one wide bordered box per runner containing the runner code, its
  kana name, and its `xN` multiplier all together, with the material line
  below it as a separate OCR'd text line.
- 403.pdf: a small bordered box (~150x40px) around *just* the 1-2 character
  runner code (e.g. "A", "パーツ" - literally "[A][parts]"), with the `(xN)`
  multiplier (only on D and F here) and the material-composition line both
  sitting as plain, unboxed text beside/below the box - see the README for
  a side-by-side crop.

There are exactly 7 of these boxed labels on this page (runners A-G). The
8th "runner", <PC-132AB> (a polycap bag), uses a distinct bracket-style
header with no box at all and is handled as its own special case in
final_report_403.py, since it's the only one of its kind on this manual.
"""
import cv2
import numpy as np
import pytesseract
import re
import sys

# OCR reliably reads the runner code immediately followed by "パ" (the
# start of "パーツ" = "parts"), e.g. "Aパーツ" - a plain \b([A-G])\b word
# boundary regex doesn't fire here since Python's re treats the adjacent
# kana as a \w character too, so anchor on the following パ instead.
CODE_RE = re.compile(r'([A-G])パ')


def find_label_boxes(gray):
    """Locate the 7 small bordered boxes around each runner's 1-letter code
    (calibrated on this page: hollow-rectangle contours ~130-170px wide,
    30-50px tall, with polygon-area/bbox-area > 0.85 - looser candidates
    with a lower ratio here are physical part-diagram shapes, not text
    boxes, unlike 949 where 0.8 alone was a clean cut)."""
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (130 < w < 170 and 30 < h < 50):
            continue
        area = cv2.contourArea(c)
        ratio = area / (w * h)
        if ratio > 0.85:
            raw.append((x, y, w, h))
    raw.sort()
    used = [False] * len(raw)
    boxes = []
    for i, b in enumerate(raw):
        if used[i]:
            continue
        cluster = [b]
        used[i] = True
        for j in range(i + 1, len(raw)):
            if used[j]:
                continue
            if abs(raw[j][0] - b[0]) < 8 and abs(raw[j][1] - b[1]) < 8:
                cluster.append(raw[j])
                used[j] = True
        best = max(cluster, key=lambda t: t[2] * t[3])
        boxes.append(best)
    return boxes


def ocr_region(gray, x0, y0, x1, y1, box=None):
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    sub = gray[y0:y1, x0:x1]
    scale = 3
    big = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if box is not None:
        bx, by, bw_, bh_ = box
        thick = 14
        left_x = (bx - x0) * scale
        top_y = (by - y0) * scale
        right_x = (bx + bw_ - x0) * scale
        bot_y = (by + bh_ - y0) * scale
        cv2.line(bw, (left_x, 0), (left_x, bw.shape[0]), 255, thick)
        cv2.line(bw, (0, top_y), (bw.shape[1], top_y), 255, thick)
        cv2.line(bw, (right_x, 0), (right_x, bw.shape[0]), 255, thick)
        cv2.line(bw, (0, bot_y), (bw.shape[1], bot_y), 255, thick)
    texts = []
    for psm in (6, 7, 11):
        t = pytesseract.image_to_string(bw, lang='jpn+eng', config=f'--psm {psm}').strip()
        texts.append(t)
    return texts


def pick_code(texts):
    for text in texts:
        m = CODE_RE.search(text.replace(' ', ''))
        if m:
            return m.group(1)
    return None


def pick_mult(texts):
    for text in texts:
        mm = re.search(r'[x×X]\s*([0-9])', text)
        if mm:
            return int(mm.group(1))
    return 1


def detect_labels(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = find_label_boxes(gray)

    labels = []
    for (bx, by, bw_, bh_) in boxes:
        # OCR the box itself (border-erased) for the code, and a generous
        # region to the right + below for the material line / multiplier -
        # this page's material line sits beside the box for some runners
        # and below it for others (see module docstring), so cover both.
        code_texts = ocr_region(gray, bx - 5, by - 5, bx + bw_ + 5, by + bh_ + 5, box=(bx, by, bw_, bh_))
        context_texts = ocr_region(gray, bx - 10, by - 10, bx + bw_ + 520, by + bh_ + 90)
        code = pick_code(code_texts) or pick_code(context_texts)
        mult = pick_mult(context_texts)
        labels.append({
            'code': code, 'mult': mult,
            'raw': ' || '.join(code_texts + context_texts),
            'box_l': bx, 'box_t': by, 'box_r': bx + bw_, 'box_b': by + bh_,
            'anchor_x': bx + bw_ / 2.0, 'anchor_y': by + bh_,
        })

    # This page has exactly 7 lettered runners (A-G), always laid out in
    # the same reading order (2 rows: A,B,C,D then E,F,G, left to right).
    # OCR reliably reads 6 of 7 (one runner - "B" at the time this was
    # checked - sometimes comes back too garbled for the regex to match
    # anything at all). Rather than hand-fixing one specific letter the
    # way 949's fix_labels() does, fill any single remaining gap by
    # elimination against the known A-G set in reading order - this only
    # fires when exactly one code is missing, so it can't silently paper
    # over a real multi-box OCR failure.
    all_codes = list('ABCDEFG')
    missing_idx = [i for i, lb in enumerate(sorted(labels, key=lambda d: (d['box_t'], d['box_l']))) if lb['code'] is None]
    if len(missing_idx) == 1 and len(labels) == 7:
        found_codes = {lb['code'] for lb in labels if lb['code']}
        remaining = [c for c in all_codes if c not in found_codes]
        if len(remaining) == 1:
            ordered = sorted(labels, key=lambda d: (d['box_t'], d['box_l']))
            ordered[missing_idx[0]]['code'] = remaining[0]

    return labels


if __name__ == '__main__':
    labels = detect_labels(sys.argv[1])
    for lb in sorted(labels, key=lambda d: (d['box_t'], d['box_l'])):
        print(lb['code'], 'x%d' % lb['mult'], '| box=(%d,%d)' % (lb['box_l'], lb['box_t']), '| raw:', repr(lb['raw']))
