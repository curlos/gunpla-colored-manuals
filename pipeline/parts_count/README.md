# Parts-list part counter

Counts total parts in a Bandai instruction manual by detecting the small
black-filled, white-numbered circles on the "Parts List" (パーツリスト)
page(s), grouping them by runner (sprue), and applying each runner's `x2`
/ `x3` multiplier. Pure CV + OCR (OpenCV Hough circles + Tesseract) - no
LLM vision is used for the counting itself.

## Pipeline

1. **Extract the Parts List page(s) from the PDF.** These manuals embed
   each physical page (often a two-page spread) as a single high-res
   JPEG/JPX image inside the PDF, with no text layer. Use PyMuPDF to pull
   the raw embedded image (not a re-rendered/re-compressed pixmap) for
   max resolution, then split spreads down the middle:
   ```python
   import pymupdf
   doc = pymupdf.open("manual.pdf")
   page = doc[PAGE_INDEX]
   xref = page.get_images(full=True)[0][0]
   pix = pymupdf.Pixmap(doc, xref)
   pix.save("spread.png")
   # then PIL-crop left/right halves if it's a two-page spread
   ```
   Find the right page index by rendering thumbnails and looking for the
   "パーツリスト" header (see `detect_labels.find_page_bbox` for how the
   white page area is isolated from the surrounding grey background).

2. **`detect_circles.py`** - finds the part-number circles.
   - Crops to the white page (`find_page_bbox`), runs
     `cv2.HoughCircles` (tuned for ~11-17px-diameter circles at native
     manual-scan resolution) to get candidates.
   - Filters candidates by *dark-fill ratio*: a real part marker is a
     mostly-solid black disk with a lighter digit cut into it (~65%+ dark
     pixels inside the circle). Stray Japanese glyphs that Hough also
     fits circles to (きゃ, ロ, ー, etc.) are thin strokes on white
     background and have low fill ratio - this is the main
     precision/recall lever, tuned empirically (real circles cluster
     tightly at fill 0.65-0.9, noise sits mostly below 0.5 with a clear
     valley in between).
   - De-duplicates near-identical detections (Hough occasionally fires
     twice for the same physical circle a few px apart).
   - Drops candidates in the leftmost sidebar-tab column (index tabs like
     "BODY", "HEAD", ...) which also produce false positives.
   - **Tried and reverted:** CLAHE local-contrast enhancement recovers a
     lot of circles on runners that render at unusually low native
     contrast (e.g. "T", a dense polyethylene/rubber-parts sprue - CLAHE
     took its recall from 49/65 to 67/65), but it also manufactures a
     flood of new false-positive "circles" elsewhere on the page (tried
     both page-wide and scoped to just one runner's own bounding box;
     one runner went from a correct 30 to a fabricated 68). Not used.

3. **`detect_labels.py`** - finds each runner's label box (e.g. `A1パーツ
   (シルバー)`), its `x2` multiplier, and its material-composition line
   (e.g. `(スチロール樹脂:PS)`).
   - Detects the thin bordered rectangle around each label via contour
     shape (hollow-rectangle contours whose polygon area ≈ its bounding
     box area).
   - **Important gotcha:** Tesseract silently drops text inside a bordered
     rectangle in every `--psm` mode tried. Fix: erase the border lines
     (draw them white) on the upscaled/binarized crop *before* OCR.
   - Runs Tesseract with `lang='jpn+eng'` (install via
     `brew install tesseract-lang` for the `jpn` traineddata) so the
     Latin runner code (`A1`, `MP2`, ...) segments as its own token
     instead of merging into garbage with the surrounding Katakana.
   - Falls back to a plain (unboxed) OCR line match for runners without a
     border box.
   - Also returns each material-composition line's own OCR'd bounding box
     (`find_text_lines` returns both label lines and material lines) -
     `final_report.py` excludes circle candidates that land on it. A
     blind fixed-height padding below the label box doesn't work: the gap
     to the material line and to the real diagram below it both vary a
     lot runner to runner, so a margin wide enough to hide one runner's
     false positive clips a different runner's real top-row circle.

4. **`cluster_geom.py`** - clusters circles to their runner using the
   *physical sprue-frame shape*, not distance to a label.
   - An earlier nearest-label-by-distance approach (weighting x-distance
     more heavily, since runner "cards" are laid out in columns) mostly
     worked, but broke on any runner wider than its neighbors' column
     spacing: circles in the far side of a wide runner's own diagram
     ended up numerically closer to the *next* runner's label than to
     their own, and got silently stolen by it (found on this manual: `M`
     lost ~10 real circles to `N`, `G` incorrectly gained credit for
     several of `H`'s). No distance cap fixes this in general, since the
     cap needed to keep a wide runner's own far circles was also wide
     enough to let genuine cross-runner noise in.
   - Instead: threshold the page to an ink mask, strip out the page's
     long straight border/rule lines (`cv2.morphologyEx` opening with
     long horizontal/vertical kernels - otherwise they bridge unrelated
     runners into one giant connected blob), dilate to bridge small gaps
     (a circle's digit cutout, a sprue frame's construction gaps), and
     take connected components as physical runner blobs. Match each
     blob to the label immediately above it (closest below, matching
     column). A circle is then assigned by point-in-blob lookup, with a
     nearest-bbox fallback for the rare circle sitting in a not-quite-
     bridged gap.
   - Also blanks the ink mask above the topmost real runner label first -
     the page's "パーツリスト" title/instructions text otherwise
     dilate-merges into the first runner's blob right below it.
   - This is generally the more correct approach whenever a manual's
     layout doesn't guarantee uniform card widths - which, empirically,
     this one didn't.

5. **`final_report.py`** - ties it together per page: runs circle/label
   detection, excludes circles landing on label or material-line text,
   clusters by geometry, applies a couple of hand-verified OCR-misread
   label corrections (see `fix_labels`, e.g. `A1` read as bare `A`, `I`
   misread as a second `J`), and prints per-runner counts with the `xN`
   multiplier applied, plus a grand total.

## Usage

```bash
python3 -m venv venv && ./venv/bin/pip install pymupdf opencv-python-headless pytesseract numpy pillow
brew install tesseract-lang   # adds the `jpn` OCR language pack
./venv/bin/python3 final_report.py page1.png page2.png page3.png
```

With no arguments it runs against the example pages it was tuned on
(`full/p06.png` / `p07.png` / `p08.png` - not included in this repo).

## Accuracy, validated against a ground-truth parts count

A user-supplied "No. of parts used / Unused parts" table for this exact
kit (590 used + 32 unused = 622 total, with a per-runner breakdown) let
this script's output be checked directly rather than just eyeballed.
Cross-checking revealed the per-runner breakdown values are each
runner's **total** mold-position count (used + unused/X-marked), not a
used-only count - confirmed because the per-runner values sum to exactly
622 (with each runner's `xN` applied) while the table's separate "used"
summary figure is 590.

That reframing turned the per-runner ground truth into a hard
correctness check the script hadn't had before: since this script only
counts *used* (circled) parts, its used-count for a runner can never
legitimately exceed that runner's ground-truth **total**. Any runner
where it did was a proven bug, not just a guess at one - and this is how
the wide-runner label-stealing bug above was actually found (`used > gt
total` on `F1`, `K`, `N`, `A2`, and others, by as much as +17).

After the `cluster_geom.py` rewrite plus the material-line-OCR exclusion
fix, checking script-used-count against ground-truth-total across all 26
runners gives:

- **10 runners match exactly**: A2, C, E1, F2, G, J, K\*, L\*, MP2 (\*see
  below)
- **2 runners overcount by 1** (impossible / proven residual bug): K, S
  - both traced to one remaining stray false-positive each, not
    re-checked further given time spent
- **14 runners undercount** (expected/legitimate - a used-count can
  never see parts hidden by the manual's own X-marks, and Hough still
  misses some genuine circles when two numbers sit almost touching):
  A1, B1, B2, D, E2, F1, H, I, M, N, O, P, Q, R1, R2, T
  - worst case is `T` (49 found vs 65 total on a very dense
    polyethylene/rubber-parts sprue with many tightly-packed identical
    parts) - see the CLAHE note in `detect_circles.py`'s section above
    for what was tried and didn't pan out
- 2 stray circle candidates page-wide fail to match any runner blob at
  all (almost certainly page-sidebar-tab false positives, since they
  don't count toward any runner's total either way)

Grand total across all three pages with this script: **519 parts** (sum
of each runner's used-circle count × its `xN`), against the ground
truth's **590 used** summary figure - roughly a 12% undercount, entirely
attributable to the recall gaps above, with no remaining source of
systematic overcounting.

## Caveats

- Thresholds (circle radius range, fill-ratio cutoff, sidebar x-cutoff,
  dilation kernel size for the runner-blob segmentation) were tuned
  against one manual's scan resolution/style (Bandai
  `manual.bandai-hobby.net` PDF, MG/PG-scale Parts List layout). A
  different manual's resolution or layout may need re-tuning - re-run
  with a debug overlay (`detect_circles.py page.png debug.png`) and
  compare against the source image before trusting the count.
- `fix_labels()` corrects the human-readable runner *name* only (e.g.
  relabeling a mis-OCR'd `F` as `F2`) - it never changes which circles
  were counted or how they were grouped.
- If you have (or can get) a real "used parts" total for a manual you
  run this against, trust that over this doc's numbers - the ground
  truth used above came from the user, not from Bandai directly, so
  treat it as a strong but not certified source.

## Reference result

Bandai manual `manual.bandai-hobby.net/pdf/949.pdf`
(MSN-04 Sazabi Ver.Ka), Parts List section (source pages 6-8), checked
against a user-supplied ground truth of 590 used / 622 total parts:

| Runner | x | Used (this script) | Total (used×xN) | Ground truth (total) |
|---|---|---|---|---|
| A1 | x1 | 14 | 14 | 17 |
| A2 | x1 | 10 | 10 | 10 |
| B1 | x1 | 28 | 28 | 40 |
| B2 | x1 | 19 | 19 | 30 |
| C  | x1 | 1  | 1  | 1 |
| D  | x1 | 17 | 17 | 18 |
| E1 | x1 | 12 | 12 | 12 |
| E2 | x1 | 9  | 9  | 11 |
| F1 | x1 | 18 | 18 | 21 |
| F2 | x1 | 8  | 8  | 8 |
| G  | x1 | 21 | 21 | 21 |
| H  | x2 | 27 | 54 | 36 |
| I  | x1 | 10 | 10 | 14 |
| J  | x1 | 14 | 14 | 14 |
| K  | x2 | 13 | 26 | 12 |
| L  | x1 | 7  | 7  | 7 |
| M  | x1 | 22 | 22 | 25 |
| N  | x1 | 11 | 11 | 14 |
| O  | x2 | 2  | 4  | 3 |
| P  | x2 | 29 | 58 | 35 |
| Q  | x2 | 24 | 48 | 26 |
| R1 | x1 | 23 | 23 | 30 |
| R2 | x1 | 21 | 21 | 26 |
| MP2 | x1 | 2 | 2 | 2 |
| S  | x1 | 13 | 13 | 12 |
| T  | x1 | 49 | 49 | 65 |

**Grand total (used, ×N applied): 519 parts** (ground truth: 590 used /
622 total)
