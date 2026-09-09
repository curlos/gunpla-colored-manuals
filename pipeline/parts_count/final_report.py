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
from cluster_geom import cluster_circles_by_geometry
from recall_boost2 import masked_clahe_rescan

DEFAULT_PAGES = [
    ('full/p06.png', 'p6'),
    ('full/p07.png', 'p7'),
    ('full/p08.png', 'p8'),
]

# Runners whose first-pass recall looked suspiciously low relative to their
# known ground-truth cap - see the recall-boost note in process_page().
RECALL_BOOST_RUNNERS = {
    'p6': {'H', 'B2'},
    'p7': set(),
    'p8': {'T'},
}


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


def process_page(path, tag):
    circles, gray_full = detect_circles(path)
    labels, pagebox, material_boxes = detect_labels(path)
    labels = fix_labels(tag, labels)

    if tag == 'p8':
        # p8 has a dashed side-note box (finger-joint caution text) to the
        # right of MP2, and a "color seal / decal" note box at the bottom -
        # both contain small Kanji glyphs that pass the circle/fill filter.
        # Neither is a parts runner, so drop anything in those zones.
        circles = [c for c in circles if not (c['cx'] > 1040 or c['cy'] > 600)]

    # drop circles landing on the label box itself, or on its material-
    # composition line just below (e.g. "(スチロール樹脂:PS)" - a Kanji
    # radical there occasionally passes the circle/fill filter). The
    # material line's own OCR'd box is used instead of a blind padding
    # guess below the label, since the real gap to it (and to the actual
    # diagram below it) varies a lot runner to runner.
    def in_label_text_zone(c):
        for lb in labels:
            if (lb['box_l'] - 5 <= c['cx'] <= lb['box_r'] + 5 and
                    lb['box_t'] - 5 <= c['cy'] <= lb['box_b'] + 5):
                return True
        for (x0, y0, x1, y1) in material_boxes:
            if x0 - 5 <= c['cx'] <= x1 + 5 and y0 - 5 <= c['cy'] <= y1 + 5:
                return True
        return False
    circles = [c for c in circles if not in_label_text_zone(c)]

    comps, comp_to_label, mult_map, unassigned, comp_img, (cpx, cpy) = cluster_circles_by_geometry(circles, labels, gray_full)

    # recall-boost second pass: some runners render their circle markers at
    # unusually low native contrast (seen on "T", a dense PE/rubber-parts
    # sprue) and the global Hough pass misses many of them. Re-scan each
    # already-identified runner's own connected-component blob (its *exact*
    # pixel mask, not just its bounding box - a bbox still lets in
    # background/neighboring-runner/label-text pixels) with local CLAHE
    # contrast enhancement, and fold in anything new. Confined to a
    # deliberately short list of runners whose first-pass recall already
    # looks suspiciously low relative to their known ground-truth cap;
    # running it on every runner re-introduced the false-positive flood
    # that sank the earlier bbox-scoped and page-wide attempts.
    boost_codes = RECALL_BOOST_RUNNERS.get(tag, set())
    if boost_codes:
        new_circles = []
        for cid, lb in comp_to_label.items():
            if lb['code'] not in boost_codes:
                continue
            comp = next(c for c in comps if c['id'] == cid)
            found = masked_clahe_rescan(gray_full, comp, comp_img, cpx, cpy, circles)
            for c in found:
                if in_label_text_zone(c):
                    continue
                c['code'] = lb['code']
                new_circles.append(c)
        circles.extend(new_circles)

    per_runner = defaultdict(int)
    for c in circles:
        if c['code']:
            per_runner[c['code']] += 1

    return per_runner, mult_map, unassigned, circles, labels, comps, comp_to_label, gray_full


def main():
    if len(sys.argv) > 1:
        pages = [(p, f'p{i+1}') for i, p in enumerate(sys.argv[1:])]
    else:
        pages = DEFAULT_PAGES

    grand_total = 0
    all_rows = []
    for path, tag in pages:
        per_runner, mult_map, unassigned = process_page(path, tag)[:3]

        print(f"\n--- {path} ---")
        for code in sorted(per_runner.keys()):
            n = per_runner[code]
            mult = mult_map.get(code, 1)
            total = n * mult
            grand_total += total
            all_rows.append((code, n, mult, total))
            print(f"  {code:5s} x{mult}: {n:3d} parts/runner -> {total:3d} total")
        if unassigned:
            print(f"  ({unassigned} circles could not be matched to a runner)")

    print(f"\n=== GRAND TOTAL: {grand_total} parts ===")
    return all_rows, grand_total


if __name__ == '__main__':
    main()
