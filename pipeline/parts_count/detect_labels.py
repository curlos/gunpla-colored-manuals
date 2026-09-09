import cv2
import numpy as np
import pytesseract
import re
from collections import defaultdict

CODE_RE = re.compile(r'(MP\d|[A-Z]{1,3}\d{0,2})')
MATERIAL_CODES = {'PS', 'PE', 'ABS', 'PP', 'PC', 'POM', 'PVC'}


def find_page_bbox(gray):
    _, bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, area = stats[idx]
    return x, y, w, h


def find_box_contours(crop_gray):
    _, mask = cv2.threshold(crop_gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (100 < w < 450 and 15 < h < 45):
            continue
        area = cv2.contourArea(c)
        ratio = area / (w * h)
        if ratio > 0.8:
            raw.append((x, y, w, h))
    # dedupe nested pairs (inner/outer border contour) by clustering close boxes
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
            if abs(raw[j][0] - b[0]) < 6 and abs(raw[j][1] - b[1]) < 6:
                cluster.append(raw[j])
                used[j] = True
        # take the largest (outer) box in the cluster
        best = max(cluster, key=lambda t: t[2] * t[3])
        boxes.append(best)
    return boxes


def find_text_lines(crop_gray):
    data = pytesseract.image_to_data(crop_gray, lang='jpn+eng', config='--psm 11', output_type=pytesseract.Output.DICT)
    n = len(data['text'])
    lines = defaultdict(list)
    for i in range(n):
        t = data['text'][i].strip()
        if not t:
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        lines[key].append((data['left'][i], data['top'][i], data['width'][i], data['height'][i], t))
    out = []
    for key, toks in lines.items():
        toks.sort()
        text = ''.join(t[4] for t in toks)
        if ('ツ' in text or 'ーツ' in text) and '印' not in text:
            l = min(t[0] for t in toks)
            t_ = min(t[1] for t in toks)
            r = max(t[0] + t[2] for t in toks)
            b = max(t[1] + t[3] for t in toks)
            out.append((l, t_, r, b))
    return out


def ocr_region(crop_gray, box, extend_right=170, pad=8, erase_border=False):
    bx, by, bw_, bh_ = box
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(crop_gray.shape[1], bx + bw_ + extend_right)
    y1 = min(crop_gray.shape[0], by + bh_ + pad)
    sub = crop_gray[y0:y1, x0:x1]
    scale = 3
    big = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if erase_border:
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
    for psm in (6, 7):
        t = pytesseract.image_to_string(bw, lang='jpn+eng', config=f'--psm {psm}').strip()
        texts.append(t)
    return texts, (x0, y0, x1, y1)


def boxes_overlap(a, b, margin=10):
    ax0, ay0, ax1, ay1 = a[0]-margin, a[1]-margin, a[0]+a[2]+margin, a[1]+a[3]+margin
    bx0, by0, bx1, by1 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def pick_code(texts):
    for text in texts:
        codes = CODE_RE.findall(text.replace(' ', ''))
        for c in codes:
            if c.upper() not in MATERIAL_CODES and len(c) <= 4:
                return c.upper()
    return None


def pick_mult(texts):
    for text in texts:
        mm = re.search(r'[x×X]\s*([0-9])', text)
        if mm:
            return int(mm.group(1))
    return 1


def detect_labels(img_path):
    img = cv2.imread(img_path)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    px, py, pw, ph = find_page_bbox(gray_full)
    crop = gray_full[py:py + ph, px:px + pw]

    boxes = find_box_contours(crop)
    lines = find_text_lines(crop)

    regions = [{'box': b, 'erase_border': True} for b in boxes]
    for (l, t_, r, b) in lines:
        line_box = (l, t_, r - l, b - t_)
        if not any(boxes_overlap(bx, line_box) for bx in boxes):
            regions.append({'box': line_box, 'erase_border': False})

    labels = []
    for reg in regions:
        texts, (x0, y0, x1, y1) = ocr_region(crop, reg['box'], erase_border=reg['erase_border'])
        code = pick_code(texts)
        mult = pick_mult(texts)
        bx, by, bw_, bh_ = reg['box']
        labels.append({
            'code': code, 'mult': mult, 'raw': ' || '.join(texts),
            'anchor_x': px + bx + bw_ / 2.0, 'anchor_y': py + by + bh_,
            'box_l': px + bx, 'box_t': py + by, 'box_r': px + bx + bw_, 'box_b': py + by + bh_,
        })

    return labels, (px, py, pw, ph)


if __name__ == '__main__':
    import sys
    labels, pagebox = detect_labels(sys.argv[1])
    for lb in sorted(labels, key=lambda d: (d['box_t'], d['box_l'])):
        print(lb['code'], 'x%d' % lb['mult'], '| anchor=(%d,%d)' % (lb['anchor_x'], lb['anchor_y']), '| raw:', repr(lb['raw']))
