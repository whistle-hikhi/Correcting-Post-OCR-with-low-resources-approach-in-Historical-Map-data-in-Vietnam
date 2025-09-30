from paddleocr import PaddleOCR
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import json, csv, math
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR processor using PaddleOCR with tiling support"""
    
    def __init__(self, min_score: float = 0.2, iou_merge: float = 0.2):
        self.min_score = min_score
        self.iou_merge = iou_merge
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    def compute_dynamic_tiling(self, img_w: int, img_h: int, max_tiles: int = 100) -> Tuple[int, int]:
        """
        Compute dynamic tile_size and overlap based on image dimensions.
        Ensures fewer than max_tiles tiles while keeping overlap safe.
        """
        # estimate tile size so that total tiles <= max_tiles
        est_tile = int(max(img_w, img_h) / math.sqrt(max_tiles))
        # keep it in a safe range
        tile_size = max(512, min(est_tile, 4000))   # clamp between 512 and 4000 px
        # overlap = 10% of tile_size, but at least 100 px
        overlap = max(100, int(tile_size * 0.1))
        return tile_size, overlap

    def split_image(self, img: Image.Image, max_tiles: int = 100) -> List[Tuple[Tuple[int, int, int, int], Image.Image]]:
        """Split image into tiles for processing"""
        w, h = img.size
        tile_size, overlap = self.compute_dynamic_tiling(w, h, max_tiles=max_tiles)
        logger.info(f"📐 Image {w}x{h} → tile_size={tile_size}, overlap={overlap}")

        step = tile_size - overlap
        tiles = []
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
                tiles.append(((x0, y0, x1, y1), img.crop((x0, y0, x1, y1))))
        return tiles

    def exif_safe_open(self, path: str) -> Image.Image:
        """Safely open image with EXIF orientation handling"""
        img = Image.open(path)
        return ImageOps.exif_transpose(img).convert("RGB")

    def is_normalized(self, poly_list: List[List[List[float]]]) -> bool:
        """Check if coordinates are normalized (0..1)"""
        mx = 0.0
        for p in poly_list:
            for (x, y) in p:
                mx = max(mx, float(x), float(y))
        return mx <= 2.0

    def poly_to_aabb(self, poly: List[List[float]]) -> List[float]:
        """Convert polygon to axis-aligned bounding box"""
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return [min(xs), min(ys), max(xs), max(ys)]

    def iou(self, a: List[float], b: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        # a,b are [x0,y0,x1,y1]
        ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
        iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0: return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def nms_merge(self, objs: List[Dict], iou_thr: float = None) -> List[Dict]:
        """Non-Maximum Suppression to merge overlapping detections"""
        if iou_thr is None:
            iou_thr = self.iou_merge
            
        # objs: list of dicts with "bbox" (poly), "score"
        # simple NMS on AABBs, keep highest score
        objs = sorted(objs, key=lambda d: d["score"], reverse=True)
        kept = []
        aabbs = []
        for o in objs:
            a = self.poly_to_aabb(o["bbox"])
            drop = False
            for kb, ka in zip(kept, aabbs):
                if self.iou(a, ka) >= iou_thr:
                    drop = True
                    break
            if not drop:
                kept.append(o)
                aabbs.append(a)
        return kept

    def draw_overlay(self, base_img: Image.Image, results: List[Dict], out_path: str = "overlay.png"):
        """Draw OCR results overlay on image"""
        draw = ImageDraw.Draw(base_img, "RGBA")
        for r in results:
            poly = [(int(x), int(y)) for (x,y) in r["bbox"]]
            # polygon outline + translucent fill
            draw.polygon(poly, outline=(255,0,0,255))
            # optional label
            # draw.text(poly[0], r["text"][:20], fill=(255,0,0,255))
        base_img.save(out_path)

    def process_image(self, image_path: str) -> List[Dict]:
        """Process a single image through OCR with tiling"""
        img = self.exif_safe_open(image_path)
        tiles = self.split_image(img)

        ocr_output = []
        W, H = img.size

        for (x0, y0, x1, y1), tile in tiles:
            tile_np = np.array(tile)  # Paddle expects numpy or str
            th, tw = tile_np.shape[0], tile_np.shape[1]

            results = self.ocr.predict(tile_np)  # returns list[dict] for one image

            if not results:
                continue
            r = results[0]

            polys = r.get("dt_polys", [])
            texts = r.get("rec_texts", [])
            scores = r.get("rec_scores", [])

            if len(polys) == 0:
                continue

            # If detector returns normalized coords (0..1), rescale to tile pixels
            norm = self.is_normalized(polys)
            for bbox, text, score in zip(polys, texts, scores):
                if score is None or float(score) < self.min_score:
                    continue

                # bbox is a list/array of 4 points [[x,y],...]
                pts = []
                for (px, py) in bbox:
                    if norm:
                        gx = float(px) * tw + x0
                        gy = float(py) + y0
                    else:
                        gx = float(px) + x0
                        gy = float(py) + y0
                    pts.append([gx, gy])

                ocr_output.append({
                    "bbox": pts,                # polygon in full-image coords
                    "text": text,
                    "score": float(score)
                })

        # Deduplicate boxes from overlaps
        ocr_output = self.nms_merge(ocr_output, iou_thr=self.iou_merge)

        # Round coordinates for output stability
        for r in ocr_output:
            r["bbox"] = [[int(round(x)), int(round(y))] for (x,y) in r["bbox"]]

        logger.info(f"✅ OCR completed! Found {len(ocr_output)} text regions")
        return ocr_output