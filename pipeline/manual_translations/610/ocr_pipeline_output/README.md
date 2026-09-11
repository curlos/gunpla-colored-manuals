# Non-LLM OCR + Translation Pipeline — Test Run

This is a real, scriptable, non-LLM pipeline (`pipeline/ocr_translate.py`), run against
the same two manual pages (page 3 "Parts List" and page 8 "Painting Guide") that were
previously translated by hand by Claude reading the images directly.

**Stack (no LLM anywhere):**
1. OpenCV — region detection (see below) + per-region grayscale/2x-upscale/threshold
2. Tesseract OCR (`pytesseract`, `lang=jpn`) — text + bounding boxes
3. Google Translate via `deep-translator` (`GoogleTranslator`, ja → en), with a
   `MyMemoryTranslator` fallback + retry/backoff — plain MT web APIs, not an LLM

Run it yourself:
```
cd pipeline
uv run ocr_translate.py <image_path> <output_dir> --crop x,y,w,h --psm 6
uv run ocr_translate.py <image_path> <output_dir> --crop x,y,w,h --auto   # region-detect mode
```

## Region detection (`--auto`)

The single-crop mode above requires *you* to manually find good crop coordinates for
every label on the page (see the `page3_title` / `page3_runner_a_label` trial-and-error
below). `--auto` instead detects candidate text-block regions automatically, so the
whole page can run unattended. Getting this from 0 useful regions to ~40 well-aligned
ones took three real fixes, all in `detect_text_regions()`:

1. **Global Otsu threshold → 0 regions.** The maroon PAINTING box has light text on a
   dark background, opposite of the rest of the (light-background, dark-text) page. One
   global threshold picks one polarity and the other type either vanishes or swallows
   the whole box as one giant false-positive blob.
2. **Dual-polarity adaptive threshold → regions appear, but box borders fuse everything.**
   Switching to per-neighborhood adaptive thresholding (run both polarities, OR them
   together) fixed the light-on-dark problem — but every box's border rectangle and every
   divider line also count as "ink," and since they all touch/nearly-touch across the
   layout, dilation fused huge swaths of the page into single useless blobs.
3. **Strip long straight lines before dilating → real regions.** Running a morphological
   opening with long horizontal/vertical kernels first removes the border/divider strokes
   (they're the only things long and straight enough to survive it) while leaving actual
   glyph strokes alone. *After* that, closing + anisotropic dilation to fuse a label's
   wrapped lines gives boxes that visibly track the real labels — see
   `page8_painting_box_auto/detected_regions.png`.

Even after all that, it's a heuristic (classic connected-components/contours), not a
trained text-detector model — expect noise and occasional merged/split boxes, not
pixel-perfect per-label segmentation.

## A second, unrelated failure: getting rate-limited mid-run

Running `--auto` across all 42 detected regions fires ~80+ translation calls in a tight
loop. Partway through, Google's free translate endpoint (the one `GoogleTranslator`
scrapes) started silently refusing *every* request — even a single trivial word like
"こんにちは" failed in isolation afterward, with no distinguishable error from a genuine
"no translation" case. Fixed by adding retry-with-backoff and a fallback to
`MyMemoryTranslator` (different backend/quota) in `translate_text()`. Good to know before
trusting this for a full-manual batch job: the free backend needs throttling and a
fallback, not naive per-line calls.

## Results

| Folder | Crop | Result |
|---|---|---|
| `page8_single_swatch/` | One isolated, single-column label ("本体などの塗装色 モンザレッド(65%)+レッド(30%)+ホワイト(5%)") | **Works reasonably.** OCR read the kanji correctly; translation came out understandable ("Paint color of main body etc." / "Monza Red (659%%)" / "+Red (3096)+White (596)-"). Note the `%` sign gets consistently misread as extra digits (`65%` → `659%%`) — a known Tesseract/jpn quirk with the percent glyph. |
| `page8_painting_box/` | The full 3-column PAINTING chart, uncropped-per-cell | **Mostly garbage.** PSM 6 treats the 3 side-by-side color columns as one text block and interleaves them mid-sentence (e.g. one "line" merges "body color" + "weapon color" + "thruster color" text together). Only 1 of 23 detected lines translated successfully — the one full-width footnote sentence that wasn't in a multi-column layout. |
| `page3_runner_a_label/` | One isolated line: "Aパーツ（ブラック）（スチロール樹脂：PS" | **Partially works, with a funny failure.** OCR misread "パーツ" (parts) as "バーツ" (a one-character dakuten/handakuten mistake — パ vs バ). Google Translate then correctly translated "バーツ" as **"Baht"** — the Thai currency — because that's a real, common Japanese word. Output: *"LA baht (black) (styrene resin: PS"*. This is a good concrete example of why naive OCR+MT breaks on domain-specific technical text: a single misread diacritic silently produces a fluent, wrong, unrelated translation instead of failing loudly. |
| `page3_title/` | The page's title banner, inside a rounded box outline | **Fails.** The box's border line and adjacent column text bleed into the OCR region, and PSM 7 (single line) can't segment around the shape. Output was unusable garbage; translation correctly refused it ("No translation was found"). |
| `page8_painting_box_auto/` | Same 3-column PAINTING chart as `page8_painting_box/`, but with `--auto` region detection instead of one big crop | **Much better, still imperfect.** 42 regions detected, 84 text lines, translations mostly land close to correct meaning now that each swatch's label is its own region instead of 3 columns interleaved. E.g. `よりリアルに仕上けたい方は、下の基本色をご発くたさい。` → *"If you want a more realistic finish, please use the basic colors below."* — right meaning even with two OCR typos in the Japanese. But quality is inconsistent: some regions still merge unrelated fragments across nearby boxes (e.g. `頂、振などの浴徹折` → *"Thorough bathing such as peak and swing"*, garbled from "肩、襟などの塗装色" / shoulders-collar-etc.), and one region OCR'd `肛の挫徹名` from what should have been `顔の塗装色` (face color) — translated, hilariously and wrongly, as *"name of anus"*. |

## Takeaway

Naive single-crop OCR does **not** work on these manual pages — confirmed by `page8_painting_box/` above. Automated **region detection** (the `--auto` mode) gets meaningfully closer: going from 0 usable regions to ~40 well-aligned ones took three real fixes (dual-polarity thresholding, then stripping border lines before dilating — see above), and the resulting translations are mostly readable and roughly correct. But it's still a classical CV heuristic, not a trained text-detector model, so a meaningful fraction of regions merge, split, or misread text badly enough that translation output ranges from "close enough" to "confidently wrong" (a few genuinely funny OCR-driven mistranslations turned up, like "name of anus" for "face paint color"). Isolated, single-column, boundary-free text crops remain the one case that worked cleanly without any of this machinery.

The full manual (LLM-read) translations remain in `../page3_parts_list_translated.md` for comparison.
