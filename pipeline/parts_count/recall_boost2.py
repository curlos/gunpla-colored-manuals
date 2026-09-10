import cv2
import numpy as np


def masked_clahe_rescan(gray_full, comp, comp_img, px, py, existing_circles):
    """Re-scan a single runner's own connected-component blob (not just its
    bounding box) with local CLAHE contrast enhancement, to recover circles
    that render at unusually low native contrast on some runners (seen on
    "T", a dense polyethylene/rubber-parts sprue). Earlier attempts scoped
    only to the component's *bounding box* still leaked in background,
    neighboring-runner edges, and label text that the rectangular crop
    didn't exclude - that's what caused the flood of false positives
    elsewhere on the page. This version blanks every pixel outside the
    component's actual dilated blob mask to white before enhancing/scanning,
    so Hough only ever sees this one runner's own ink."""
    pad = 20
    x0 = max(0, comp['x'] - px - pad)
    y0 = max(0, comp['y'] - py - pad)
    x1 = min(comp_img.shape[1], comp['x'] - px + comp['w'] + pad)
    y1 = min(comp_img.shape[0], comp['y'] - py + comp['h'] + pad)

    crop_full = gray_full[y0 + py:y1 + py, x0 + px:x1 + px].copy()
    mask_crop = (comp_img[y0:y1, x0:x1] == comp['id'])
    # dilate the mask a couple px so circle anti-aliased edges aren't clipped
    mask_u8 = mask_crop.astype(np.uint8) * 255
    mask_u8 = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    crop_full[mask_u8 == 0] = 255  # blank everything outside this runner's own blob

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(crop_full)
    blur = cv2.medianBlur(enhanced, 3)
    _, dark_mask = cv2.threshold(enhanced, 190, 255, cv2.THRESH_BINARY_INV)

    circ = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=7,
                             param1=80, param2=10, minRadius=5, maxRadius=9)
    if circ is None:
        return []

    new_circles = []
    for x, y, r in circ[0]:
        xi, yi, ri = int(round(x)), int(round(y)), int(round(r))
        m = np.zeros(crop_full.shape, dtype=np.uint8)
        cv2.circle(m, (xi, yi), max(1, int(ri * 0.85)), 255, -1)
        region = cv2.bitwise_and(dark_mask, m)
        total = cv2.countNonZero(m)
        dark = cv2.countNonZero(region)
        fill = dark / total if total > 0 else 0
        if fill < 0.63:
            continue
        gx, gy = x + px + x0, y + py + y0
        if any(abs(gx - c['cx']) < 8 and abs(gy - c['cy']) < 8 for c in existing_circles):
            continue

        # a real number marker has a digit cut into it: on a *normal-
        # contrast* runner that's usually a fragmented, relatively small
        # lighter region, while a plain round plastic part (ball joint,
        # wheel, washer) that CLAHE also makes pass the dark-fill test above
        # tends to have one big, simple, roughly circular lighter "hole"
        # instead - calibrated against R1's false positives (0.49-1.00,
        # commonly a single contour) vs its real digits (0.14-0.42, commonly
        # 2+ fragmented contours). Measured on the raw (pre-CLAHE) crop,
        # matching how it was calibrated - CLAHE's local contrast stretch
        # changes this ratio enough to matter.
        #
        # NOT a universal signal, though: on a runner that's low-contrast
        # in its *native* scan (e.g. "T"), even its real digits show a big
        # light_frac after CLAHE stretches the whole circle's contrast, so
        # this would wrongly reject them. It's returned on every candidate
        # but left for the caller to decide whether to threshold on
        # per-runner (see LIGHT_FRAC_CHECK_RUNNERS in final_report.py).
        xi_n, yi_n = int(round(x)), int(round(y))
        raw_sub = crop_full[max(0, yi_n - ri - 2):yi_n + ri + 2, max(0, xi_n - ri - 2):xi_n + ri + 2]
        light_frac = None
        if raw_sub.size > 0:
            _, raw_dark = cv2.threshold(raw_sub, 190, 255, cv2.THRESH_BINARY_INV)
            raw_light = 255 - raw_dark
            lm = np.zeros(raw_sub.shape, dtype=np.uint8)
            cv2.circle(lm, (raw_sub.shape[1] // 2, raw_sub.shape[0] // 2), max(1, int(ri * 0.85)), 255, -1)
            light_in = cv2.bitwise_and(raw_light, lm)
            circle_px = cv2.countNonZero(lm)
            light_frac = cv2.countNonZero(light_in) / circle_px if circle_px else 0

        w = h = r * 2
        new_circles.append({
            'cx': float(gx), 'cy': float(gy), 'r': float(r), 'fill': fill,
            'light_frac': light_frac,
            'x': int(gx - r), 'y': int(gy - r), 'w': int(w), 'h': int(h),
        })
    return new_circles
