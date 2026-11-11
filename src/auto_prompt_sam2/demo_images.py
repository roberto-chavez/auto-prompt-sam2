"""Sentinel-2 chip input and manual prompts, without torch or label access."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

S2_LAYOUT = "GeoTIFF Sen1Floods11: bandas 4/3/2 (DN)"
RGB_LAYOUT = "GeoTIFF de 3 bandas: R/G/B (DN)"
DISPLAY_LAYOUT = "PNG/JPEG: imagen o visualización preparada"
AUTO_LAYOUT = "Automático (según extensión)"
LAYOUTS = [AUTO_LAYOUT, S2_LAYOUT, RGB_LAYOUT, DISPLAY_LAYOUT]


@dataclass
class DemoImage:
    rgb: np.ndarray
    valid: np.ndarray
    metadata: dict

    def preview(self):
        return np.rint(np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)


def load_image(path, layout, params, max_upload_mb=32):
    path = Path(path)
    if path.stat().st_size > max_upload_mb * 1024**2:
        raise ValueError(f"El archivo supera {max_upload_mb} MB")
    if layout not in LAYOUTS:
        raise ValueError("Seleccionar un formato de entrada")
    if layout == AUTO_LAYOUT:
        layout = DISPLAY_LAYOUT if path.suffix.lower() in (".png", ".jpg", ".jpeg") else S2_LAYOUT
    if layout == DISPLAY_LAYOUT:
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Este modo requiere PNG/JPEG")
        with Image.open(path) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("PNG/JPEG demasiado grande: máximo 16 millones de píxeles")
            if source.mode not in ("RGB", "RGBA", "L", "LA", "P", "1"):
                raise ValueError("Usar una visualización PNG/JPEG de 8 bits, RGB o gris")
            oriented = ImageOps.exif_transpose(source)
            original_wh = oriented.size
            rgba = oriented.convert("RGBA")
            resized = ImageOps.contain(rgba, (512, 512), Image.Resampling.BILINEAR)
            left, top = (512 - resized.width) // 2, (512 - resized.height) // 2
            canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            canvas.paste(resized, (left, top))
            pixels = np.asarray(canvas)
            valid = pixels[:, :, 3] > 0
            # Composite partial transparency over black; padding remains invalid.
            rgb = pixels[:, :, :3].astype(np.float32) / 255.0
            rgb *= pixels[:, :, 3:4].astype(np.float32) / 255.0
        metadata = {"radiometry": "rendered_RGB_not_equivalent_to_raw_E2", "georeferenced": False,
                    "original_wh": list(original_wh), "resized_wh": list(resized.size),
                    "padding_left_top": [left, top], "coordinate_space": "prepared_512x512_canvas",
                    "input_note": "PNG/JPEG exploratorio: no identifica sensor ni elimina títulos o bordes"}
    else:
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("Este modo requiere un GeoTIFF con DN int16/uint16")
        import rasterio
        with rasterio.open(path) as ds:
            if (ds.height, ds.width) != (512, 512):
                raise ValueError("Recortar un chip de 512×512; no se remuestrea la escena")
            if (layout == S2_LAYOUT and ds.count != 13) or (layout == RGB_LAYOUT and ds.count != 3):
                raise ValueError("Sen1Floods11 requiere 13 bandas; RGB requiere 3. Elegir el orden correcto")
            bands = [4, 3, 2] if layout == S2_LAYOUT else [1, 2, 3]
            raw = ds.read(bands)
            if raw.dtype not in (np.dtype("int16"), np.dtype("uint16")):
                raise ValueError("Se esperan DN int16/uint16; no reflectancia float ni TIFF RGB de 8 bits")
            valid = (ds.read_masks(bands) > 0).all(axis=0) & np.isfinite(raw).all(axis=0)
            # Same order of float32 operations as preprocessing.rgb_unit in phase 1.
            reflectance = raw.astype(np.float32) / params["config"]["reflectance_divisor"]
            unit = np.clip(reflectance, 0, params["upper_reflectance"]) / params["upper_reflectance"]
            unit[:, ~valid] = params["config"]["invalid_rgb_fill_before_normalize"]
            rgb = np.moveaxis(unit, 0, -1).copy()
            metadata = {"radiometry": "fixed_train_E2", "bands_one_based": bands,
                        "upper_dn": params["upper_dn"], "georeferenced": ds.crs is not None,
                        "crs": str(ds.crs), "transform": list(ds.transform)[:6]}
    if not valid.any():
        raise ValueError("El chip no contiene píxeles RGB válidos")
    metadata.update({"shape_hw": [512, 512], "valid_pixels": int(valid.sum()),
                     "input_layout": layout})
    return DemoImage(rgb, valid, metadata)


def validate_points(points, image, max_points=64, *, require_positive=True):
    if image is None:
        raise ValueError("Primero carga un chip")
    if len(points) > max_points:
        raise ValueError(f"Máximo {max_points} puntos")
    seen = set()
    for point in points:
        if len(point) != 3 or any(type(v) is not int for v in point):
            raise ValueError("Cada punto requiere x, y y etiqueta enteros")
        x, y, label = point
        if label not in (0, 1) or not (0 <= x < 512 and 0 <= y < 512):
            raise ValueError("Punto fuera de la imagen o etiqueta inválida")
        if not image.valid[y, x]:
            raise ValueError("El punto cae sobre un píxel sin datos")
        if (x, y) in seen:
            raise ValueError("Ya existe un punto en ese píxel; deshaz el anterior")
        seen.add((x, y))
    if require_positive and not any(p[2] == 1 for p in points):
        raise ValueError("Añade al menos un punto positivo sobre el agua")
    return points


def draw_points(image, points, mask=None):
    pixels = image.preview()
    if mask is not None:
        visible = np.asarray(mask, dtype=bool) & image.valid
        pixels[visible] = np.rint(.55 * pixels[visible] + .45 * np.array([30, 145, 255])).astype(np.uint8)
    rendered = Image.fromarray(pixels)
    draw = ImageDraw.Draw(rendered)
    for x, y, label in points:
        color = (35, 220, 95) if label else (255, 75, 80)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
        draw.line((x - 3, y, x + 3, y), fill="black", width=1)
        if label:
            draw.line((x, y - 3, x, y + 3), fill="black", width=1)
    return np.asarray(rendered)
