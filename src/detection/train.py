"""
Entrenamiento del modelo de detección de stock/vacíos en góndola.

Usa YOLOv8 (Ultralytics) por su buen balance velocidad/precisión,
clave dado el requisito de latencia de alerta (~1seg).

Requiere que el dataset esté en formato YOLO:

  data/
    processed/
      images/
        train/
        val/
      labels/
        train/    <- un .txt por imagen, formato: class x_center y_center width height (normalizado)
        val/

Y un archivo dataset.yaml (ver configs/dataset.yaml) que apunte a esas carpetas
y liste las clases (ej: "vacio", "producto" o las clases que definas).
"""

from ultralytics import YOLO
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent.parent

SEED = 42


def fijar_seeds(seed: int = SEED) -> None:
    """
    Fija seeds en todas las librerías aleatorias para máxima
    reproducibilidad de resultados (Requisito kickoff: "Resultados replicables").
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Operaciones deterministas en CPU/GPU cuando se pueda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Variables de entorno usadas por algunas libs
    os.environ["PYTHONHASHSEED"] = str(seed)


def entrenar(
    dataset_yaml: str = "configs/dataset.yaml",
    modelo_base: str = "yolov8n.pt",  # 'n' = nano, más liviano/rápido. Subir a 's' o 'm' si sobra precisión.
    epochs: int = 100,
    img_size: int = 640,
    batch: int = 16,
    nombre_run: str = "stock_detector_v1",
    seed: int = SEED,
):
    fijar_seeds(seed)
    modelo = YOLO(modelo_base)

    project_dir = str(BASE / "models" / "runs")
    resultados = modelo.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name=nombre_run,
        patience=20,          # early stopping si no mejora en 20 epochs
        augment=True,         # data augmentation integrado (flip, hsv, mosaic, etc.)
        project=project_dir,
        seed=seed,
        deterministic=True,
    )

    salida = Path(project_dir) / nombre_run / "weights" / "best.pt"
    print(f"\nEntrenamiento terminado. Mejor modelo guardado en:")
    print(f"  {salida}")
    print(f"  (seed usado = {seed}, deterministic=True)")

    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de detección de stock")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--modelo-base", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--nombre", default="stock_detector_v1")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed para reproducibilidad")
    args = parser.parse_args()

    entrenar(
        dataset_yaml=args.data,
        modelo_base=args.modelo_base,
        epochs=args.epochs,
        img_size=args.img_size,
        batch=args.batch,
        nombre_run=args.nombre,
        seed=args.seed,
    )
