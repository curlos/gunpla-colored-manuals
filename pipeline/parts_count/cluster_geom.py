import cv2
import numpy as np
from detect_circles import find_page_bbox


def build_runner_components(gray_full, labels, min_area=5000):
    """Segment the page into physical runner-card blobs via connected components
    on the ink mask (with page border/rule lines stripped out first), then match
    each blob to the label anchored just above it. This is far more robust than
    nearest-label-by-distance for wide/irregular runner layouts (e.g. a runner
    whose label sits left-of-center over an unusually wide diagram)."""
    px, py, pw, ph = find_page_bbox(gray_full)
    crop = gray_full[py:py+ph, px:px+pw]
    _, ink = cv2.threshold(crop, 240, 255, cv2.THRESH_BINARY_INV)

    # blank out the page's instruction/header line (e.g. "X marks = unused
    # parts", "パーツリスト" title oval) sitting above the runner cards -
    # otherwise its decorative border/ink can dilate-merge into the first
    # real runner's blob right below it. Blank the whole strip above the
    # topmost *real* (non-header) label, full page width, to be safe (the
    # header's own oval border can extend beyond its OCR text bbox).
    real_tops = [lb['box_t'] for lb in labels if lb['code'] and lb['code'] != 'X']
    if real_tops:
        cutoff = min(real_tops) - py - 15
        if cutoff > 0:
            ink[0:cutoff, :] = 0

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(h_lines, v_lines)
    ink_clean = cv2.bitwise_and(ink, cv2.bitwise_not(lines))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dil = cv2.dilate(ink_clean, kernel)
    n, comp_img, stats, centroids = cv2.connectedComponentsWithStats(dil, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append({'id': i, 'x': x + px, 'y': y + py, 'w': w, 'h': h, 'area': int(area)})

    return comps, comp_img, (px, py)


def match_labels_to_components(labels, comps):
    """Assign each label to the component whose bounding box the label's anchor
    sits just above/inside (label box bottom-center should fall within, or just
    above, the component x-range and at/near its top)."""
    assignments = {}  # comp_id -> label
    for lb in labels:
        if not lb['code'] or lb['code'] == 'X':
            continue
        best = None
        best_score = 1e18
        ax = (lb['box_l'] + lb['box_r']) / 2.0
        ay = lb['box_b']
        for c in comps:
            cx0, cy0, cx1, cy1 = c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']
            if not (cx0 - 20 <= ax <= cx1 + 20):
                continue
            if ay > cy1 + 10:
                continue
            # prefer the component whose top is closest below the label, and
            # whose horizontal center is closest to the label's center
            dy = max(0, cy0 - ay)
            dx = abs((cx0 + cx1) / 2.0 - ax)
            score = dy * 3 + dx
            if score < best_score:
                best_score = score
                best = c['id']
        if best is not None:
            # keep the closer label if two labels claim the same component
            if best in assignments:
                prev = assignments[best]
                prev_ax = (prev['box_l'] + prev['box_r']) / 2.0
                prev_ay = prev['box_b']
                prev_c = next(c for c in comps if c['id'] == best)
                prev_score = max(0, prev_c['y'] - prev_ay) * 3 + abs((prev_c['x'] + prev_c['w']/2.0) - prev_ax)
                if best_score >= prev_score:
                    continue
            assignments[best] = lb
    return assignments


def cluster_circles_by_geometry(circles, labels, gray_full):
    comps, comp_img, (px, py) = build_runner_components(gray_full, labels)
    comp_to_label = match_labels_to_components(labels, comps)

    # map from comp id -> code
    comp_to_code = {cid: lb['code'] for cid, lb in comp_to_label.items()}
    mult_map = {lb['code']: lb['mult'] for lb in comp_to_label.values()}

    unassigned = 0
    for c in circles:
        lx, ly = int(c['cx'] - px), int(c['cy'] - py)
        code = None
        if 0 <= ly < comp_img.shape[0] and 0 <= lx < comp_img.shape[1]:
            cid = comp_img[ly, lx]
            if cid != 0:
                code = comp_to_code.get(cid)
        if code is None:
            # fall back: nearest component by bbox distance (circle might sit
            # in a small gap the dilation didn't bridge)
            best = None
            best_d = 1e18
            for comp in comps:
                cx0, cy0, cx1, cy1 = comp['x'], comp['y'], comp['x']+comp['w'], comp['y']+comp['h']
                dx = max(cx0 - c['cx'], 0, c['cx'] - cx1)
                dy = max(cy0 - c['cy'], 0, c['cy'] - cy1)
                d = dx*dx + dy*dy
                if d < best_d:
                    best_d = d
                    best = comp['id']
            if best is not None and best_d < 40*40:
                code = comp_to_code.get(best)
        c['code'] = code
        if code is None:
            unassigned += 1

    return comps, comp_to_label, mult_map, unassigned, comp_img, (px, py)
