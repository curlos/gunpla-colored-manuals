# OCR + Translation: page8_painting_guide.png

Pipeline: OpenCV preprocessing -> Tesseract OCR (`lang=jpn`) -> Google Translate (`deep-translator`). No LLM involved.

- **JA:** 本体などの塗装色
  **EN:** Paint color of main body etc.
  _(conf: 89.2, bbox: [371, 45, 767, 122])_
- **JA:** モンザレッド(659%%)
  **EN:** Monza Red (659%%)
  _(conf: 87.2, bbox: [372, 107, 814, 179])_
- **JA:** +レッド(3096)+ホワイト(596)ー
  **EN:** +Red (3096)+White (596)-
  _(conf: 82.6, bbox: [371, 167, 1409, 237])_