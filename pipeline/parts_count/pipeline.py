import sys
import re
import cv2
import numpy as np
from collections import defaultdict

from detect_circles import detect_circles
from detect_labels import detect_labels, CODE_RE


def cluster_circles_to_labels(circles, labels):
    # only labels with a resolved code participate as cluster targets
    valid_labels = [l for l in labels if l['code']]
    MAX_DX = 360   # runner cards are compact; a part this far sideways from
    MAX_DY = 420   # its label (or this far below it) is almost certainly noise
    for c in circles:
        best_i = None
        best_d = 1e18
        for i, lb in enumerate(valid_labels):
            if lb['anchor_y'] > c['cy'] + 5:  # label must be above (or barely level with) the circle
                continue
            dx = c['cx'] - lb['anchor_x']
            dy = c['cy'] - lb['anchor_y']
            if abs(dx) > MAX_DX or dy > MAX_DY:
                continue
            d = dx * dx * 2.5 + dy * dy  # weight x distance more (columns matter)
            if d < best_d:
                best_d = d
                best_i = i
        c['label_idx'] = best_i
    return valid_labels


def read_digit(gray_full, circle, debug=False):
    x, y, w, h = circle['x'], circle['y'], circle['w'], circle['h']
    pad = max(2, int(0.35 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = x + w + pad, y + h + pad
    sub = gray_full[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    scale = 8
    big = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # circle is dark fill with lighter/white digit; binarize then invert so digit is black-on-white
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure background (majority) is white: count corner pixels
    corners = [bw[0, 0], bw[0, -1], bw[-1, 0], bw[-1, -1]]
    if sum(1 for v in corners if v < 128) >= 3:
        bw = 255 - bw
    txt = pytesseract_digit(bw)
    return txt


import pytesseract

def pytesseract_digit(bw_img):
    cfg = '--psm 10 -c tessedit_char_whitelist=0123456789'
    txt = pytesseract.image_to_string(bw_img, config=cfg).strip()
    txt = re.sub(r'[^0-9]', '', txt)
    return txt if txt else None


def run_page(img_path, verbose=True):
    circles, gray_full = detect_circles(img_path)
    labels, pagebox = detect_labels(img_path)

    # drop false-positive circles that land on the label/material-composition text
    # itself (e.g. a Kanji radical in "樹脂" occasionally fill-matches a real circle)
    def in_label_text_zone(c):
        for lb in labels:
            if (lb['box_l'] - 5 <= c['cx'] <= lb['box_r'] + 5 and
                    lb['box_t'] - 5 <= c['cy'] <= lb['box_b'] + 10):
                return True
        return False
    before_lbl = len(circles)
    circles = [c for c in circles if not in_label_text_zone(c)]
    if verbose:
        print(f"Dropped {before_lbl - len(circles)} circles overlapping label text")

    valid_labels = cluster_circles_to_labels(circles, labels)

    for c in circles:
        c['digit'] = read_digit(gray_full, c)

    per_runner = defaultdict(list)
    unassigned = 0
    for c in circles:
        if c['label_idx'] is None:
            unassigned += 1
            continue
        lb = valid_labels[c['label_idx']]
        per_runner[lb['code']].append(c)

    mult_map = {lb['code']: lb['mult'] for lb in valid_labels}

    if verbose:
        print(f"=== {img_path} ===")
        print(f"Total circles detected: {len(circles)}  (unassigned: {unassigned})")
        for code in sorted(per_runner.keys()):
            digs = sorted(int(d) for d in (c['digit'] for c in per_runner[code]) if d and d.isdigit())
            mult = mult_map.get(code, 1)
            n = len(per_runner[code])
            print(f"  {code} (x{mult}): {n} parts detected | numbers read: {digs}")
    return circles, valid_labels, per_runner, mult_map


if __name__ == '__main__':
    run_page(sys.argv[1])
