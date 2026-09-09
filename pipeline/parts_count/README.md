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
     precision/recall lever, tuned empirically (see fill-ratio histogram
     approach: real circles cluster tightly at 0.65-0.9, noise sits
     mostly below 0.5 with a clear valley in between).
   - Drops candidates in the leftmost sidebar-tab column (index tabs like
     "BODY", "HEAD", ...) which also produce false positives.

3. **`detect_labels.py`** - finds each runner's label box (e.g. `A1パーツ
   (シルバー)`) and its `x2` multiplier.
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

4. **`pipeline.py`** - clusters circles to their nearest label.
   - A circle is assigned to the nearest label whose anchor is above it
     (x-distance weighted more heavily than y, since runner "cards" are
     laid out in columns), subject to a max-distance cap. The cap matters:
     without it, a page's *last* label with nothing below/beside it to
     compete for "nearest" will vacuum up unrelated false-positive
     circles from footnote text arbitrarily far down the page.
   - Per-circle digit OCR (`read_digit`) is informational only (shown in
     the debug report) - it was not reliable enough at this circle size
     to use as a hard include/exclude filter.

5. **`final_report.py`** - ties it together per page, applies a couple of
   hand-verified OCR-misread label corrections (see `fix_labels`, e.g.
   `A1` read as bare `A`, `I` misread as a second `J`), and prints
   per-runner counts with the `xN` multiplier applied, plus a grand total.

## Usage

```bash
python3 -m venv venv && ./venv/bin/pip install pymupdf opencv-python-headless pytesseract numpy pillow
brew install tesseract-lang   # adds the `jpn` OCR language pack
./venv/bin/python3 final_report.py page1.png page2.png page3.png
```

With no arguments it runs against the example pages it was tuned on
(`full/p06.png` / `p07.png` / `p08.png` - not included in this repo).

## Caveats

- Thresholds (circle radius range, fill-ratio cutoff, sidebar x-cutoff,
  clustering distance cap) were tuned against one manual's scan
  resolution/style (Bandai `manual.bandai-hobby.net` PDF, MG/PG-scale
  Parts List layout). A different manual's resolution or layout may need
  re-tuning - re-run with a debug overlay (`detect_circles.py page.png
  debug.png`) and compare against the source image before trusting the
  count.
- Estimated accuracy on the reference manual (see below) is roughly
  ±3-5%: a small number of real circles are missed when two part numbers
  sit almost touching each other, and a small number of Kanji glyphs in
  the "material composition" line under each label occasionally pass the
  fill-ratio filter. These two error sources partially offset each other.
- `fix_labels()` corrects the human-readable runner *name* only (e.g.
  relabeling a mis-OCR'd `F` as `F2`) - it never changes which circles
  were counted or how they were grouped.

## Reference result

Bandai manual `manual.bandai-hobby.net/pdf/949.pdf`
(MSN-04 Sazabi Ver.Ka), Parts List section (source pages 6-8):

| Runner | x | Parts/runner | Total |
|---|---|---|---|
| A1 | x1 | 11 | 11 |
| A2 | x1 | 12 | 12 |
| B1 | x1 | 31 | 31 |
| B2 | x1 | 20 | 20 |
| C  | x1 | 1  | 1 |
| D  | x1 | 11 | 11 |
| E1 | x1 | 13 | 13 |
| E2 | x1 | 11 | 11 |
| F1 | x1 | 26 | 26 |
| F2 | x1 | 9  | 9 |
| G  | x1 | 14 | 14 |
| H  | x2 | 33 | 66 |
| I  | x1 | 11 | 11 |
| J  | x1 | 12 | 12 |
| K  | x2 | 16 | 32 |
| L  | x1 | 8  | 8 |
| M  | x1 | 13 | 13 |
| N  | x1 | 21 | 21 |
| O  | x2 | 2  | 4 |
| P  | x2 | 30 | 60 |
| Q  | x2 | 21 | 42 |
| R1 | x1 | 27 | 27 |
| R2 | x1 | 21 | 21 |
| MP2 | x1 | 2 | 2 |
| S  | x1 | 13 | 13 |
| T  | x1 | 49 | 49 |

**Grand total: 540 parts**
