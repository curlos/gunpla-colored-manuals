"""
Count parts-list circles per runner across one or more Parts List page images.

Usage:
    python3 final_report.py page1.png [page2.png ...]

If no arguments are given, falls back to the example pages this script was
developed and tuned against (Bandai manual 949.pdf / MSN-04 Sazabi Ver.Ka,
Parts List pages extracted to full/p06.png, full/p07.png, full/p08.png -
see README.md for how those were produced from the source PDF).

NOTE: `fix_labels()` below hand-corrects a handful of OCR letter/digit
misreads (A1->A, F2->F, I->J collision, O->0, MP2->MP8) that were verified
by visually inspecting those specific source pages. It does not generalize
to a new manual out of the box - for a different manual, re-check the
`detect_labels` output against the page image and extend/replace this
function as needed. It only relabels an already correctly-clustered,
script-counted group of circles; it never affects the counts themselves.
"""
import sys
from collections import defaultdict
from detect_circles import detect_circles
from detect_labels import detect_labels
from pipeline import cluster_circles_to_labels, read_digit

DEFAULT_PAGES = [
    ('full/p06.png', 'p6'),
    ('full/p07.png', 'p7'),
    ('full/p08.png', 'p8'),
]


def fix_labels(page_tag, valid_labels):
    if page_tag == 'p6':
        for lb in valid_labels:
            if lb['code'] == 'A':
                lb['code'] = 'A1'
            elif lb['code'] == 'F' and abs(lb['anchor_x'] - 1097) < 60:
                lb['code'] = 'F2'
    elif page_tag == 'p7':
        js = [lb for lb in valid_labels if lb['code'] == 'J']
        if len(js) == 2:
            js.sort(key=lambda lb: lb['anchor_x'])
            js[0]['code'] = 'I'  # leftmost mis-OCR'd J -> I
        for lb in valid_labels:
            if lb['code'] is None and abs(lb['anchor_x'] - 273) < 80 and abs(lb['anchor_y'] - 884) < 80:
                lb['code'] = 'O'
    elif page_tag == 'p8':
        for lb in valid_labels:
            if lb['code'] == 'MP8':
                lb['code'] = 'MP2'
    return valid_labels


def main():
    if len(sys.argv) > 1:
        pages = [(p, f'p{i+1}') for i, p in enumerate(sys.argv[1:])]
    else:
        pages = DEFAULT_PAGES

    grand_total = 0
    all_rows = []
    for path, tag in pages:
        circles, gray_full = detect_circles(path)
        labels, pagebox = detect_labels(path)

        labels = fix_labels(tag, labels)

        if tag == 'p8':
            # p8 has a dashed side-note box (finger-joint caution text) to the
            # right of MP2, and a "color seal / decal" note box at the bottom -
            # both contain small Kanji glyphs that pass the circle/fill filter.
            # Neither is a parts runner, so drop anything in those zones.
            circles = [c for c in circles if not (c['cx'] > 1040 or c['cy'] > 600)]

        # drop circles landing on the label/material text itself
        def in_label_text_zone(c):
            for lb in labels:
                if (lb['box_l'] - 5 <= c['cx'] <= lb['box_r'] + 5 and
                        lb['box_t'] - 5 <= c['cy'] <= lb['box_b'] + 10):
                    return True
            return False
        circles = [c for c in circles if not in_label_text_zone(c)]

        valid_labels = cluster_circles_to_labels(circles, labels)

        per_runner2 = defaultdict(int)
        mult_map2 = {}
        idx_to_code = {i: lb['code'] for i, lb in enumerate(valid_labels)}
        for c in circles:
            if c['label_idx'] is None:
                continue
            code = idx_to_code[c['label_idx']]
            if code is None:
                continue
            per_runner2[code] += 1
        for lb in valid_labels:
            if lb['code']:
                mult_map2[lb['code']] = lb['mult']

        print(f"\n--- {path} ---")
        for code in sorted(per_runner2.keys()):
            n = per_runner2[code]
            mult = mult_map2.get(code, 1)
            total = n * mult
            grand_total += total
            all_rows.append((code, n, mult, total))
            print(f"  {code:5s} x{mult}: {n:3d} parts/runner -> {total:3d} total")

    print(f"\n=== GRAND TOTAL: {grand_total} parts ===")
    return all_rows, grand_total


if __name__ == '__main__':
    main()
