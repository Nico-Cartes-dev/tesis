"""
Manejo de Zonas de Interés (ROI) que varían de tamaño/forma entre cámaras.

Problema que resuelve: cada cámara tiene su repisa en un lugar distinto del
frame, y a veces no es un rectángulo perfecto (repisa vista en ángulo).

Solución: coordenadas SIEMPRE normalizadas (0-1) guardadas en configs/cameras.yaml,
así el mismo ROI sirve sin importar la resolución real de la cámara, y se
soporta tanto rectángulo simple como polígono arbitrario.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
import yaml


@dataclass
class ROI:
    id: str
    tipo: str  # "rectangulo" | "poligono"
    clase_objetivo: str
    puntos_normalizados: List[Tuple[float, float]]  # siempre como lista de (x, y)

    def a_pixeles(self, ancho: int, alto: int) -> np.ndarray:
        """Convierte coordenadas normalizadas a píxeles reales de la imagen actual."""
        pts = [(int(x * ancho), int(y * alto)) for x, y in self.puntos_normalizados]
        return np.array(pts, dtype=np.int32)


class ROIHandler:
    def __init__(self, config_path: str = "configs/cameras.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def get_rois(self, camera_id: str) -> List[ROI]:
        camara = self.config["cameras"].get(camera_id)
        if camara is None:
            raise ValueError(f"Cámara '{camera_id}' no encontrada en config")

        rois = []
        for roi_cfg in camara["rois"]:
            if roi_cfg["tipo"] == "rectangulo":
                x_min, y_min, x_max, y_max = roi_cfg["bbox"]
                puntos = [
                    (x_min, y_min), (x_max, y_min),
                    (x_max, y_max), (x_min, y_max),
                ]
            else:  # poligono
                puntos = [tuple(p) for p in roi_cfg["puntos"]]

            rois.append(ROI(
                id=roi_cfg["id"],
                tipo=roi_cfg["tipo"],
                clase_objetivo=roi_cfg["clase_objetivo"],
                puntos_normalizados=puntos,
            ))
        return rois

    def recortar_roi(self, imagen: np.ndarray, roi: ROI) -> np.ndarray:
        """
        Recorta la zona del ROI de la imagen.
        Para polígonos no rectangulares, aplica una máscara y recorta
        al bounding box del polígono (rellena fuera de la máscara con negro).
        """
        alto, ancho = imagen.shape[:2]
        pts_px = roi.a_pixeles(ancho, alto)

        if roi.tipo == "rectangulo":
            x_min, y_min = pts_px.min(axis=0)
            x_max, y_max = pts_px.max(axis=0)
            return imagen[y_min:y_max, x_min:x_max]

        # Polígono: máscara + recorte al bounding box
        mask = np.zeros((alto, ancho), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_px], 255)
        recorte_completo = cv2.bitwise_and(imagen, imagen, mask=mask)

        x_min, y_min = pts_px.min(axis=0)
        x_max, y_max = pts_px.max(axis=0)
        return recorte_completo[y_min:y_max, x_min:x_max]

    def dibujar_roi(self, imagen: np.ndarray, roi: ROI, color=(0, 255, 0)) -> np.ndarray:
        """Dibuja el contorno del ROI sobre la imagen (para debug/visualización)."""
        alto, ancho = imagen.shape[:2]
        pts_px = roi.a_pixeles(ancho, alto)
        imagen_out = imagen.copy()
        cv2.polylines(imagen_out, [pts_px], isClosed=True, color=color, thickness=2)
        return imagen_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python roi_handler.py <camera_id> <ruta_imagen>")
        sys.exit(1)

    camera_id, ruta_imagen = sys.argv[1], sys.argv[2]

    handler = ROIHandler(config_path="configs/cameras.yaml")
    imagen = cv2.imread(ruta_imagen)

    for roi in handler.get_rois(camera_id):
        recorte = handler.recortar_roi(imagen, roi)
        salida = f"roi_{roi.id}.jpg"
        cv2.imwrite(salida, recorte)
        print(f"ROI '{roi.id}' recortado -> {salida}")

    debug_img = imagen.copy()
    for roi in handler.get_rois(camera_id):
        debug_img = handler.dibujar_roi(debug_img, roi)
    cv2.imwrite("debug_rois.jpg", debug_img)
    print("Visualización de ROIs -> debug_rois.jpg")
