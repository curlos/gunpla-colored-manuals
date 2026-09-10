import cv2
import numpy as np
import sys


def find_page_bbox(gray):
    """Locate the white page area within the source image.

    949.pdf's embedded spreads have visible grey margin around the white
    page, so the page is found as the largest bright connected component.
    403.pdf's embedded page image has NO grey margin at all - the raw
    image *is* already exactly the white page, edge to edge (border pixels
    are ~254-255). In that case the "largest bright blob" approach
    actually picks the wrong thing (the page content, e.g. runner
    diagrams/text, fragments the white area into many disconnected
    regions, so the largest single blob can be a small blank patch, not
    the whole page) - detected by checking the image border like this
    across both manuals. If the border is already essentially white,
    skip the blob search and just use the full image.
    """
    h, w = gray.shape
    edge = min(gray[:5, :].min(), gray[-5:, :].min(), gray[:, :5].min(), gray[:, -5:].min())
    if edge > 245:
        return 0, 0, w, h
    _, bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, area = stats[idx]
    return x, y, w, h


def detect_circles(img_path, debug_out=None):
    img = cv2.imread(img_path)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    px, py, pw, ph = find_page_bbox(gray_full)
    crop = gray_full[py:py+ph, px:px+pw]
    blur = cv2.medianBlur(crop, 3)

    circles_raw = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=7,
        param1=80, param2=10, minRadius=5, maxRadius=9,
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
            # a real part-marker circle is a solid black disk with a lighter digit
            # cut into it -> high dark-fill ratio. Text glyphs (false positives from
            # Japanese characters / sidebar labels) are thin dark strokes on white
            # background -> low dark-fill ratio.
            if fill < 0.63:
                continue
            if x < 105:  # left sidebar index tab column ("PARTS LIST", "BODY", ...)
                continue
            gx, gy = x + px, y + py
            w = h = r * 2
            circles.append({
                'cx': float(gx), 'cy': float(gy), 'r': float(r), 'fill': fill,
                'x': int(gx - r), 'y': int(gy - r), 'w': int(w), 'h': int(h),
            })

    # de-duplicate near-identical detections (Hough occasionally fires twice
    # for the same physical circle at slightly different center/radius)
    circles.sort(key=lambda c: -c['fill'])
    kept = []
    for c in circles:
        if any(abs(c['cx'] - k['cx']) < 8 and abs(c['cy'] - k['cy']) < 8 for k in kept):
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
