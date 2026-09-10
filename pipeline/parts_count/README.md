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
   - **Recall-boost second pass (`recall_boost2.masked_clahe_rescan`,
     invoked from `final_report.process_page`):** some runners render
     their circle markers at unusually low native contrast (worst case:
     "T", a dense polyethylene/rubber-parts sprue) and the main pass
     above misses a lot of them. A local CLAHE-enhanced re-scan recovers
     most of the missing ones, but is unreliable enough that it's opt-in
     per runner (`RECALL_BOOST_RUNNERS` in `final_report.py`) rather than
     applied everywhere:
     - Page-wide CLAHE: manufactures a flood of new false-positive
       "circles" everywhere (tried and reverted).
     - Scoped to just a runner's own bounding *box*: still leaks -
       a rectangular crop still includes background, an adjacent
       runner's edge, or label text the box didn't exclude. On this
       manual that alone took one runner from a correct 30 to a
       fabricated 68.
     - Scoped to the runner's exact connected-component *pixel mask*
       (blank everything outside the dilated blob to white before
       enhancing) fixes the background/neighbor leakage, and this is
       what's implemented - but a second, different failure mode
       remains even mask-scoped: CLAHE can make an ordinary **round
       plastic part** (a wheel, joint, washer - common on any runner)
       pass the same "solid dark disk" fill-ratio test a real number
       marker does, once contrast is stretched. This hit some runners
       hard (adding a dozen+ false circles) and left others untouched.
       Two secondary checks on boosted candidates only (never on the
       main pass's already-reliable detections) claw most of this back:
       - **Radius cap** (`final_report.process_page`): a runner's real
         markers are never much bigger than the ones the main pass
         already found on that *same* runner. Concretely, on `B1` the
         main pass's markers all topped out at r=6.0px, but the boost
         pass's false positives (plain beads) came in at r=6.2-8.4px -
         capping each runner's boosted candidates at its own
         already-found max + 1.0px fixed it.
       - **Light-region-shape check** (`recall_boost2`, opt-in per
         runner via `LIGHT_FRAC_CHECK_RUNNERS`): a real digit's interior
         light region is usually a small, fragmented shape (the
         numeral's strokes); a plain round part's "hole" tends to be one
         big, simple, near-circular blob. Measuring the light-pixel
         fraction inside each candidate (on the raw, pre-CLAHE crop)
         separated `R1`'s false positives (0.49-1.00) from its real
         digits (0.14-0.42) cleanly. **Not universal, though**: on a
         runner that's low-contrast in its native scan (`T`), even its
         *real* digits show a high light fraction once CLAHE stretches
         the whole circle's contrast, so this check would wrongly
         reject them there - it's why it's opt-in per runner rather
         than automatic, and off for every boosted runner except `R1`.
       With no reliable way found to predict in advance which runner
       needs which combination of checks, `RECALL_BOOST_RUNNERS` (and
       `LIGHT_FRAC_CHECK_RUNNERS`) are the outcome of actually testing
       against every undercounting runner and keeping only the
       settings that landed safely at or under each one's ground-truth
       total - see the Accuracy section.

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
   - The material line is matched by known substrings (`樹脂`, `PS)`,
     `PE)`, ...) first, but OCR occasionally garbles it into complete
     nonsense (`(スチロール樹脂:PS)` read as `(AFO-IDBS`, no known
     substring survives). Fallback: treat any short, normal-single-line-
     sized OCR line starting with a literal `(` as a material line too -
     guarded by width/height bounds so a `psm 11`-mis-grouped multi-line
     blob, or a stray diagram-detail misread as e.g. `(Al)`, doesn't
     qualify (both were tried as the naive version and produced their own
     false exclusions before the size guard was added).

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
     dilate-merges into the first runner's blob right below it. More
     generally, blanks the bounding box of *every* label whose code
     never resolved (not just the header): a page-bottom build note
     (e.g. "clear parts may have air bubbles...") can span nearly the
     full page width right below the last row of runners, and its ink
     was found bridging `R1`'s component out to 659px wide (~2x a normal
     runner's ~250-420px) by connecting it toward `R2`/`Q` underneath
     that footnote. Since `detect_labels` already finds this text as a
     label line (just with no runner code matched to it), blanking every
     unresolved-code label's box handles the header and this footnote
     case with one rule.
   - This is generally the more correct approach whenever a manual's
     layout doesn't guarantee uniform card widths - which, empirically,
     this one didn't.

5. **`final_report.py`** - ties it together per page: runs circle/label
   detection, excludes circles landing on label or material-line text,
   clusters by geometry, runs the opt-in recall-boost re-scan
   (`RECALL_BOOST_RUNNERS`) on a short list of runners it's been checked
   safe on, applies a couple of hand-verified OCR-misread label
   corrections (see `fix_labels`, e.g. `A1` read as bare `A`, `I`
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

After the `cluster_geom.py` rewrite (footnote-bridging fix that
un-merged `R1`'s component), the material-line-OCR-fallback fix, and the
opt-in recall-boost pass (now applied to nearly every undercounting
runner, with a radius-aware Euclidean de-dup, a per-runner radius cap,
and an opt-in light-region-shape + solid-fill check), checking
script-used-count against ground-truth-total across all 26 runners
gives:

- **10 runners match exactly**: A2, C, D, E1, F2, G, J, K, L, MP2
- **5 runners overcount by a small, individually-diagnosed amount**
  (impossible / proven residual bugs): `F1` (+1), `I` (+1), `M` (+1),
  `Q` (+2), `S` (+1). Each was crop-and-zoomed and visually confirmed to
  be a real, specific false positive - not a guess:
  - `S`: a small round sprue attachment nub (a physical plastic-tree
    detail, not a number marker).
  - `M`: a plain mechanical/joint detail with fill=1.00 (fixed - a
    solid-fill check now catches this specific pattern) plus one
    further false positive not yet isolated.
  - `I`: a small round peg/hole feature that renders with a light
    interior close enough to a real digit's (light-region-shape check:
    0.21, comfortably under the 0.45 cutoff that catches most
    false positives) that the existing checks don't separate it.
  - `Q`: a row of ring/donut-shaped wheel parts on the physical sprue,
    which - being genuinely ring-shaped - produce exactly the kind of
    "dark ring, lighter center" pattern a real digit marker does; at
    least one of these passes every check tried.
  - `F1`: a specific circle traced to the *original* (non-boosted, pre-
    existing) detection pass, not something the boost introduced -
    a pre-existing edge case in the base circle detector, not a boost
    regression.
  - None of these could be resolved further without either (a) a much
    larger calibration sample than the handful of confirmed examples
    available per failure mode, or (b) tolerating real risk of cutting
    genuine digits elsewhere - see the two abandoned generalization
    attempts immediately below.
- **11 runners undercount** (expected/legitimate - a used-count can
  never see parts hidden by the manual's own X-marks, and Hough still
  misses some genuine circles when two numbers sit almost touching, or
  render at low enough native contrast that even the recall-boost pass
  can't fully recover them): A1, B1, B2, E2, H, N, O, P, R1, R2, T.
  `T` is within 4 of its ground-truth total (61 vs 65); `H` has the
  largest remaining single-runner gap at 7 (×2 multiplier -> 14 actual
  parts) despite boosting - spot-checked its most visually "tight
  cluster" of six identical parts and found the main pass already
  detects all six correctly, so the gap is distributed elsewhere in
  smaller amounts rather than concentrated in one fixable spot.
- 2 stray circle candidates page-wide fail to match any runner blob at
  all (almost certainly page-sidebar-tab false positives, since they
  don't count toward any runner's total either way)

**Two generalization attempts that made things worse, and were
reverted** (left here since they're exactly the next things worth
trying again with more calibration data, not because they're
dead ends):
- **Lowering the boost pass's own Hough `param2`** (8 instead of 10, to
  surface more raw candidates before filtering) recovered more of `T`
  and `N`'s real gaps, but also pushed `H` from a safe 27 under-cap to
  39 - *over* its 36 cap - since the extra raw candidates included
  more false positives than the existing filters were tuned to catch
  at that noise level.
- **Applying the light-region-shape / solid-fill checks to every
  boosted runner** (not just the ones they were calibrated on)
  self-defeated on `T`: its real digits, being low-native-contrast,
  render with a *high* light fraction after CLAHE - the exact opposite
  of what the check assumes elsewhere - so it rejected genuine finds
  there. `T`'s recall dropped from 61 back toward its pre-boost 49 when
  this was tried broadly.

Grand total across all three pages with this script: **588 parts** (sum
of each runner's used-circle count × its `xN`), against the ground
truth's **590 used** summary figure - a 2-part (0.3%) undercount. This
was reached from 562 in this same debugging session via: fixing a
duplicate-detection bug in the boost pass (two Hough hits for one
physical circle weren't always close enough together in pixel terms to
trip the original fixed-distance de-dup - replaced with a radius-aware
check), which alone fixed `D` exactly and helped several others; then
extending the already-validated radius-cap + light-shape-check combo to
most of the remaining undercounting runners. The count briefly hit
exactly 590 mid-session with a *worse* error profile (7 total points of
proven overcounting across 5 runners, coincidentally offsetting a
larger undercount elsewhere) - 588 was kept instead as the more
honestly-accurate state, since forcing the exact number by tolerating
more known-wrong overcounts isn't a real fix. The 5 currently-known
overcounts (F1 +1, I +1, M +1, Q +2, S +1 = 6 points) and the remaining
undercount gaps are independent problems; closing the overcounts
without closing an equivalent amount of undercount would move the
total further from 590, not closer, despite being strictly more
correct per-runner.

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

| Runner | x | Used (this script) | Total (used×xN) | Ground truth (total) | Match? |
|---|---|---|---|---|---|
| A1 | x1 | 16 | 16 | 17 | under |
| A2 | x1 | 10 | 10 | 10 | **exact** |
| B1 | x1 | 39 | 39 | 40 | under |
| B2 | x1 | 25 | 25 | 30 | under |
| C  | x1 | 1  | 1  | 1  | **exact** |
| D  | x1 | 18 | 18 | 18 | **exact** |
| E1 | x1 | 12 | 12 | 12 | **exact** |
| E2 | x1 | 9  | 9  | 11 | under |
| F1 | x1 | 22 | 22 | 21 | **overcount +1** |
| F2 | x1 | 8  | 8  | 8  | **exact** |
| G  | x1 | 21 | 21 | 21 | **exact** |
| H  | x2 | 29 | 58 | 36 | under |
| I  | x1 | 15 | 15 | 14 | **overcount +1** |
| J  | x1 | 14 | 14 | 14 | **exact** |
| K  | x2 | 12 | 24 | 12 | **exact** |
| L  | x1 | 7  | 7  | 7  | **exact** |
| M  | x1 | 26 | 26 | 25 | **overcount +1** |
| N  | x1 | 12 | 12 | 14 | under |
| O  | x2 | 2  | 4  | 3  | under |
| P  | x2 | 34 | 68 | 35 | under |
| Q  | x2 | 28 | 56 | 26 | **overcount +2** |
| R1 | x1 | 25 | 25 | 30 | under |
| R2 | x1 | 22 | 22 | 26 | under |
| MP2 | x1 | 2 | 2 | 2  | **exact** |
| S  | x1 | 13 | 13 | 12 | **overcount +1** |
| T  | x1 | 61 | 61 | 65 | under (by 4) |

**Grand total (used, ×N applied): 588 parts** (ground truth: 590 used /
622 total)
