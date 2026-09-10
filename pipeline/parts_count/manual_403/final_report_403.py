"""
Full used/unused parts report for 403.pdf (HGUC 1/144 MSN-04 Sazabi),
Parts List page.

Ties together:
  - detect_circles_403.detect_circles - used-part (numbered circle) detection,
    tuned for this page's native resolution (see that module's docstring).
  - detect_labels_403.detect_labels - the 7 lettered runners' (A-G) codes
    and xN multipliers.
  - cluster_geom.build_runner_components / match_labels_to_components
    (reused as-is from the 949.pdf pipeline - the physical-blob clustering
    approach generalizes cleanly here) - assigns each circle to its runner.
  - xmark_detector_403.detect_xmarks - NEW for this manual: finds the bare
    "x" glyphs marking unused polycap positions on <PC-132AB> (see that
    module's docstring for how this was calibrated).

<PC-132AB> (the polycap bag) is not one of the 7 boxed A-G runners - it has
its own bracket-style header with no box - so it's handled as a special
case: its label position is hardcoded (verified against the actual page
image, see README) and a small min-component-height filter drops a stray
text-ink blob near D's label that would otherwise get mismatched to D's
component (see README's Clustering fix note).
"""
import sys
import os
from collections import defaultdict

sys.path.insert(0, '..')
from cluster_geom import build_runner_components, match_labels_to_components
from detect_circles_403 import detect_circles
from detect_labels_403 import detect_labels
from xmark_detector_403 import detect_xmarks

import cv2

# <PC-132AB>'s own header ("<PC-132AB> (ポリエチレン:PE)") has no bordered
# box (unlike the 7 lettered runners) so detect_labels_403 can't find it -
# this page has exactly one of these, so its position is hardcoded here
# rather than building general bracket-header detection for a single
# instance. Verified against the actual page image (see README).
PC_LABEL = {'code': 'PC', 'mult': 1, 'box_l': 1720, 'box_t': 1745,
            'box_r': 2010, 'box_b': 1800, 'anchor_x': 1865, 'anchor_y': 1800}

# Real runner-content blobs on this page are consistently tall (h=455-542
# for A-G, h=292 for PC). One label (D)'s multiplier "(x2)" + material-line
# text sits close enough above its real content to form its own small
# (h=105) connected component that then wins the "closest below" label
# match ahead of D's real, much taller, content blob - the same class of
# bug as 949's wide-runner label-stealing, different trigger. A height
# filter cleanly separates the two populations on this page (spurious ~105
# vs real 292+) without needing cluster_geom.py's shared logic changed.
MIN_COMPONENT_HEIGHT = 150


def process_page(img_path):
    circles, gray_full = detect_circles(img_path)
    labels = detect_labels(img_path)
    labels.append(dict(PC_LABEL))

    comps, comp_img, (px, py) = build_runner_components(gray_full, labels)
    comps = [c for c in comps if c['h'] > MIN_COMPONENT_HEIGHT]
    comp_to_label = match_labels_to_components(labels, comps)
    comp_to_code = {cid: lb['code'] for cid, lb in comp_to_label.items()}
    mult_map = {lb['code']: lb['mult'] for lb in comp_to_label.values()}

    def assign_point(cx, cy):
        lx, ly = int(cx - px), int(cy - py)
        if 0 <= ly < comp_img.shape[0] and 0 <= lx < comp_img.shape[1]:
            cid = comp_img[ly, lx]
            if cid != 0 and cid in comp_to_code:
                return comp_to_code[cid]
        best, best_d = None, 1e18
        for comp in comps:
            cx0, cy0 = comp['x'], comp['y']
            cx1, cy1 = comp['x'] + comp['w'], comp['y'] + comp['h']
            dx = max(cx0 - cx, 0, cx - cx1)
            dy = max(cy0 - cy, 0, cy - cy1)
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best = comp['id']
        if best is not None and best_d < 40 * 40 and best in comp_to_code:
            return comp_to_code[best]
        return None

    unassigned_circles = 0
    for c in circles:
        c['code'] = assign_point(c['cx'], c['cy'])
        if c['code'] is None:
            unassigned_circles += 1

    xmarks, _ = detect_xmarks(img_path)
    unassigned_xmarks = 0
    for xm in xmarks:
        xm['code'] = assign_point(xm['cx'], xm['cy'])
        if xm['code'] is None:
            unassigned_xmarks += 1

    used = defaultdict(int)
    unused = defaultdict(int)
    for c in circles:
        if c['code']:
            used[c['code']] += 1
    for xm in xmarks:
        if xm['code']:
            unused[xm['code']] += 1

    return used, unused, mult_map, unassigned_circles, unassigned_xmarks, circles, xmarks, labels, comps, comp_to_label, gray_full


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'full/parts_list.png'
    used, unused, mult_map, unassigned_c, unassigned_x = process_page(img_path)[:5]

    order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'PC']
    grand_used = grand_unused = grand_total = 0
    print(f"\n{'Runner':7s} {'x':4s} {'Used':>6s} {'Unused':>6s} {'Total/run':>10s} {'Used*N':>7s} {'Unused*N':>9s} {'Total*N':>8s}")
    for code in order:
        u = used.get(code, 0)
        x = unused.get(code, 0)
        mult = mult_map.get(code, 1)
        tot = u + x
        used_n, unused_n, total_n = u * mult, x * mult, tot * mult
        grand_used += used_n
        grand_unused += unused_n
        grand_total += total_n
        print(f"{code:7s} x{mult:<3d} {u:6d} {x:6d} {tot:10d} {used_n:7d} {unused_n:9d} {total_n:8d}")

    print(f"\nUnassigned: {unassigned_c} circle(s), {unassigned_x} x-mark(s)")
    print(f"\n=== GRAND TOTAL: {grand_total} parts ({grand_used} used + {grand_unused} unused) ===")


if __name__ == '__main__':
    main()
