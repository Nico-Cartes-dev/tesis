"""
Tests para src/preprocessing/image_enhancer.py
Cubre:
  - process() no cambia resolución (w, h, canales) — requisito básico
  - pipeline completo no crashea con imágenes uniformes / de baja calidad
  - (opcional) corrección gamma opcional
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from preprocessing.image_enhancer import (  # noqa: E402
    ImageEnhancer,
    PreprocessConfig,
)


class TestImageEnhancer(unittest.TestCase):
    def test_shape_preservado(self):
        """process() debe devolver la misma resolución y 3 canales."""
        formas = [(480, 640, 3), (720, 1280, 3), (100, 200, 3)]
        for forma in formas:
            with self.subTest(shape=forma):
                img = np.random.randint(0, 256, size=forma, dtype=np.uint8)
                enhancer = ImageEnhancer()
                out = enhancer.process(img)
                self.assertEqual(out.shape, forma)
                self.assertEqual(out.dtype, np.uint8)

    def test_clahe_no_crashea_con_imagen_uniforme(self):
        """CLAHE tiene casos borde con imágenes planas (0 varianza)."""
        img = np.full((480, 640, 3), fill_value=128, dtype=np.uint8)
        cfg = PreprocessConfig(
            denoise_activo=False,
            clahe_activo=True,
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=(8, 8),
        )
        enhancer = ImageEnhancer(cfg)
        out = enhancer.process(img)
        self.assertEqual(out.shape, img.shape)

    def test_solo_denoise_shape_ok(self):
        """Pipeline sin CLAHE, solo denoise."""
        img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        cfg = PreprocessConfig(
            denoise_activo=True,
            denoise_fuerza=5,
            clahe_activo=False,
        )
        enhancer = ImageEnhancer(cfg)
        out = enhancer.process(img)
        self.assertEqual(out.shape, img.shape)

    def test_gamma_opcional_activo_shape_ok(self):
        """Corrección gamma activada NO cambia el shape."""
        img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        cfg = PreprocessConfig(
            denoise_activo=False,
            clahe_activo=False,
            gamma_activo=True,
            gamma_valor=0.7,  # aclarar
        )
        enhancer = ImageEnhancer(cfg)
        out = enhancer.process(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_gamma_valor_1_igual_que_desactivado(self):
        """gamma=1.0 con LUT activa == gamma desactivada (casi idéntico)."""
        img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        cfg_on = PreprocessConfig(denoise_activo=False, clahe_activo=False,
                                  gamma_activo=True, gamma_valor=1.0)
        cfg_off = PreprocessConfig(denoise_activo=False, clahe_activo=False,
                                   gamma_activo=False)
        out_on = ImageEnhancer(cfg_on).process(img)
        out_off = ImageEnhancer(cfg_off).process(img)
        np.testing.assert_array_equal(out_on, out_off)


if __name__ == "__main__":
    unittest.main()
