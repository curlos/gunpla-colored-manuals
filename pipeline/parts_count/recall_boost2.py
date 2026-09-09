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
        w = h = r * 2
        new_circles.append({
            'cx': float(gx), 'cy': float(gy), 'r': float(r), 'fill': fill,
            'x': int(gx - r), 'y': int(gy - r), 'w': int(w), 'h': int(h),
        })
    return new_circles
