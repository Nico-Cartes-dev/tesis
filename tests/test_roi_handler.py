"""
Tests para src/preprocessing/roi_handler.py
Cubre:
  - Construcción de ROI rectángulo (coords normalizadas → píxeles correctos)
  - Construcción de ROI polígono (cerrado, puntos consistentes)
  - Dibujar ROI sobre imagen no crashea
  - Recorte: bounding box dentro del área de la imagen
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from preprocessing.roi_handler import ROI, ROIHandler  # noqa: E402


class TestROI(unittest.TestCase):
    def test_roi_rectangulo_puntos_correctos(self):
        """ROI rectángulo 10%-60% x / 20%-80% y sobre 1000x500 px
        → esquinas deben ser (100,100), (600,100), (600,400), (100,400)."""
        roi = ROI(
            id="prueba",
            tipo="rectangulo",
            clase_objetivo="producto",
            puntos_normalizados=[
                (0.10, 0.20), (0.60, 0.20),
                (0.60, 0.80), (0.10, 0.80),
            ],
        )
        pts = roi.a_pixeles(ancho=1000, alto=500)
        esperado = np.array([[100, 100], [600, 100], [600, 400], [100, 400]], dtype=np.int32)
        np.testing.assert_array_equal(pts, esperado)

    def test_roi_poligono_se_mantiene_dentro(self):
        """Ninguna coordenada del ROI (normalizada [0,1]) puede irse
        fuera de la imagen al traducir a píxeles."""
        W, H = 1920, 1080
        roi = ROI(
            id="poligono",
            tipo="poligono",
            clase_objetivo="yogur",
            puntos_normalizados=[
                (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
            ],
        )
        pts = roi.a_pixeles(W, H)
        self.assertTrue(np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= W))
        self.assertTrue(np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= H))

    def test_dibujar_roi_no_crashea(self):
        """dibujar_roi() debe devolver una imagen con el mismo shape que la entrada."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi = ROI(
            id="r",
            tipo="rectangulo",
            clase_objetivo="x",
            puntos_normalizados=[(0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)],
        )
        handler = ROIHandler.__new__(ROIHandler)  # no necesita yaml
        out = handler.dibujar_roi(img, roi)
        self.assertEqual(out.shape, img.shape)

    def test_recortar_roi_tiene_ancho_y_alto_mayor_a_cero(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(
            id="r",
            tipo="poligono",
            clase_objetivo="x",
            puntos_normalizados=[
                (0.2, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7),
            ],
        )
        handler = ROIHandler.__new__(ROIHandler)
        recorte = handler.recortar_roi(img, roi)
        self.assertGreater(recorte.shape[0], 0)
        self.assertGreater(recorte.shape[1], 0)
        self.assertEqual(recorte.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
