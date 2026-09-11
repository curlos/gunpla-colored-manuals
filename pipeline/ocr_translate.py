"""Non-LLM OCR + translation pipeline for Japanese Gunpla manual pages.

Pipeline: image -> [optional] OpenCV region detection -> per-region OpenCV
preprocessing -> Tesseract OCR (lang=jpn) -> line grouping -> Google
Translate (via deep-translator) ja->en.

No LLM is used at any stage. This is purpose-built to be compared against
(and eventually replace) manual/LLM-based page translation.

Usage:
    python ocr_translate.py <image_path> <output_dir> [--crop x,y,w,h] [--psm 6]
    python ocr_translate.py <image_path> <output_dir> --crop x,y,w,h --auto
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from deep_translator import GoogleTranslator, MyMemoryTranslator


def load_region(image_path: str, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """Load the image (BGR) and apply the optional crop, unscaled."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if crop:
        x, y, w, h = crop
        img = img[y : y + h, x : x + w]
    return img


def to_ocr_image(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Upscale small text before thresholding — Tesseract does much better
    # on Japanese glyphs above ~30px cap height.
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


def detect_text_regions(
    bgr: np.ndarray,
    min_area: int = 2500,
    max_area_frac: float = 0.10,
    min_w: int = 40,
    min_h: int = 15,
    dilate_kernel: tuple[int, int] = (11, 9),
    dilate_iterations: int = 2,
    line_removal_len: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Find candidate text-block bounding boxes without any deep model.

    Classic connected-components approach, tuned against the manual's
    boxed/multi-column layout through several failed attempts:

    1. Binarize in BOTH polarities and OR them together. A single global
       Otsu threshold can't handle a page that mixes dark text on a light
       background (most of the page) with light text on a dark background
       (e.g. the maroon PAINTING box) — one polarity swallows the other as
       one giant false-positive blob covering ~75% of the image. Per-pixel
       adaptive thresholding (local neighborhood mean, not one global value)
       run in both directions and OR'd together catches both cases.
    2. Strip long straight strokes (via morphological opening with long
       horizontal/vertical kernels) before dilating. Without this, the
       boxes' own border rectangles and divider lines are picked up as
       "ink," and dilation fuses them with neighboring boxes' borders into
       one blob spanning most of the page — since the border lines all
       touch/nearly-touch each other across the layout.
    3. Close small gaps within/between glyphs, then dilate anisotropically
       to fuse a label's wrapped lines (e.g. a 2-3 line color/percentage
       caption under one swatch) into one region, without bridging the
       (usually wider) gap to the next swatch/column over.

    Still a heuristic, not a trained text-detector model (e.g. CRAFT/EAST)
    — expect noisy/fragmented boxes, not perfect per-label segmentation.
    """
    h_img, w_img = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    block_size, C = 21, 8
    dark_on_light = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C
    )
    light_on_dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, -C
    )
    binary = cv2.bitwise_or(dark_on_light, light_on_dark)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_removal_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_removal_len))
    lines_mask = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )
    lines_mask = cv2.dilate(
        lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    text_only = cv2.bitwise_and(binary, cv2.bitwise_not(lines_mask))

    closed = cv2.morphologyEx(
        text_only, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(closed, kernel, iterations=dilate_iterations)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = max_area_frac * w_img * h_img
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if w < min_w or h < min_h:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def draw_debug_boxes(
    bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    debug = bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            debug,
            str(i),
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return debug


def ocr_lines(img: np.ndarray, psm: int) -> list[dict]:
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        img, lang="jpn", config=config, output_type=pytesseract.Output.DICT
    )

    lines: dict[tuple[int, int, int], dict] = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(float(data["conf"][i])) if data["conf"][i] != "-1" else -1
        if not text or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        entry = lines.setdefault(
            key,
            {
                "text": "",
                "confidences": [],
                "left": data["left"][i],
                "top": data["top"][i],
                "right": data["left"][i] + data["width"][i],
                "bottom": data["top"][i] + data["height"][i],
            },
        )
        entry["text"] += text
        entry["confidences"].append(conf)
        entry["left"] = min(entry["left"], data["left"][i])
        entry["top"] = min(entry["top"], data["top"][i])
        entry["right"] = max(entry["right"], data["left"][i] + data["width"][i])
        entry["bottom"] = max(entry["bottom"], data["top"][i] + data["height"][i])

    results = []
    for entry in lines.values():
        avg_conf = sum(entry["confidences"]) / len(entry["confidences"])
        results.append(
            {
                "text_ja": entry["text"],
                "confidence": round(avg_conf, 1),
                "bbox": [entry["left"], entry["top"], entry["right"], entry["bottom"]],
            }
        )
    results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return results


def translate_text(text: str, delay: float = 0.5) -> str:
    """Translate ja->en with a primary + fallback backend and light retries.

    The free Google Translate endpoint that `GoogleTranslator` scrapes will
    start silently refusing every request (even single trivial words) after
    a burst of calls in a short window — observed firsthand running this
    pipeline across ~80 regions in one page. There's no clean way to detect
    "rate limited" vs "genuinely no translation" from its exception, so on
    any failure we retry a couple times with backoff, then fall back to
    MyMemoryTranslator (a different free backend/quota) rather than give up.
    """
    time.sleep(delay)
    last_error = None
    for attempt in range(3):
        try:
            return GoogleTranslator(source="ja", target="en").translate(text)
        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))

    try:
        return MyMemoryTranslator(source="ja-JP", target="en-US").translate(text)
    except Exception as e:
        return f"[translation failed: google={last_error}; mymemory={e}]"


def translate_lines(lines: list[dict]) -> list[dict]:
    for line in lines:
        line["text_en"] = translate_text(line["text_ja"])
    return lines


def run_ocr_translate(bgr_region: np.ndarray, psm: int) -> list[dict]:
    processed = to_ocr_image(bgr_region)
    lines = ocr_lines(processed, psm)
    return translate_lines(lines)


def write_report(out_dir: Path, image_name: str, lines: list[dict], mode_note: str):
    (out_dir / "ocr_translation.json").write_text(
        json.dumps(lines, ensure_ascii=False, indent=2)
    )

    md_lines = [f"# OCR + Translation: {image_name}\n"]
    md_lines.append(
        "Pipeline: OpenCV preprocessing -> Tesseract OCR (`lang=jpn`) -> "
        f"Google Translate (`deep-translator`). No LLM involved. {mode_note}\n"
    )
    for line in lines:
        prefix = f"[region {line['region']}] " if "region" in line else ""
        md_lines.append(f"- {prefix}**JA:** {line['text_ja']}")
        md_lines.append(f"  **EN:** {line['text_en']}")
        md_lines.append(f"  _(conf: {line['confidence']}, bbox: {line['bbox']})_")
    (out_dir / "ocr_translation.md").write_text("\n".join(md_lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop region as x,y,w,h in original image pixels, applied before "
        "everything else (including --auto region detection)",
    )
    parser.add_argument(
        "--psm", type=int, default=6, help="Tesseract page segmentation mode"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect candidate text-block regions (via contour/dilate "
        "heuristic) within the image/crop and OCR+translate each separately, "
        "instead of treating the whole image/crop as one block.",
    )
    parser.add_argument(
        "--min-area", type=int, default=2500, help="--auto: min region area (px^2)"
    )
    parser.add_argument(
        "--max-area-frac",
        type=float,
        default=0.10,
        help="--auto: max region area as a fraction of the (cropped) image area",
    )
    args = parser.parse_args()

    crop = None
    if args.crop:
        crop = tuple(int(v) for v in args.crop.split(","))
        if len(crop) != 4:
            sys.exit("--crop must be x,y,w,h")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    region = load_region(args.image_path, crop)

    if not args.auto:
        cv2.imwrite(str(out_dir / "preprocessed.png"), to_ocr_image(region))
        lines = run_ocr_translate(region, args.psm)
        write_report(out_dir, Path(args.image_path).name, lines, mode_note="")
        print(f"Found {len(lines)} text lines.")
    else:
        boxes = detect_text_regions(
            region, min_area=args.min_area, max_area_frac=args.max_area_frac
        )
        cv2.imwrite(
            str(out_dir / "detected_regions.png"), draw_debug_boxes(region, boxes)
        )

        all_lines = []
        for i, (x, y, w, h) in enumerate(boxes):
            sub = region[y : y + h, x : x + w]
            sub_lines = run_ocr_translate(sub, args.psm)
            for line in sub_lines:
                # offset bbox back into region-relative coordinates
                line["bbox"] = [
                    x + line["bbox"][0] // 2,
                    y + line["bbox"][1] // 2,
                    x + line["bbox"][2] // 2,
                    y + line["bbox"][3] // 2,
                ]
                line["region"] = i
                all_lines.append(line)

        write_report(
            out_dir,
            Path(args.image_path).name,
            all_lines,
            mode_note=f"Auto-detected {len(boxes)} candidate regions "
            "(see detected_regions.png) and OCR'd each separately.",
        )
        print(f"Detected {len(boxes)} candidate regions.")
        print(f"Found {len(all_lines)} text lines across all regions.")
        print(f"Wrote {out_dir / 'detected_regions.png'}")

    print(f"Wrote {out_dir / 'ocr_translation.json'}")
    print(f"Wrote {out_dir / 'ocr_translation.md'}")


if __name__ == "__main__":
    main()
