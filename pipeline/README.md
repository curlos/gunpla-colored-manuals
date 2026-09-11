# Pipeline

Python processing pipeline — runs as a background/batch job, not a live server.

Scrapes kit metadata + manual PDFs (NewType.us) and processes manuals: badge/digit OCR, part-shape isolation, recoloring, and unused-parts (X-mark) detection.

## `ocr_translate.py`

Non-LLM OCR + translation of manual page images: OpenCV region detection ->
per-region preprocessing -> Tesseract OCR (`lang=jpn`) -> Google Translate
(`deep-translator`, with a MyMemory fallback). No LLM involved at any stage.

```
uv run ocr_translate.py <image_path> <output_dir> --crop x,y,w,h --psm 6   # one manual crop
uv run ocr_translate.py <image_path> <output_dir> --crop x,y,w,h --auto    # auto region detection
```

Tested against `manual_translations/610/` — see
`manual_translations/610/ocr_pipeline_output/README.md` for results and known limits
(region detection is a classical CV heuristic, not a trained text detector — expect
noisy/merged regions and occasional bad OCR->mistranslation combos, not
production-ready accuracy).
